import sys
sys.path.append('build')

import MatterSim

import csv
import base64
import numpy as np
import json
import math
import time
from tqdm import tqdm
import os

import networkx as nx

csv.field_size_limit(sys.maxsize)


class PretrainDataGenerator:
    def __init__(self, data_path=None, use_aug=True):
        self.json_data = []
        self.data_dict = {}

        if data_path is None:
            self.sim = MatterSim.Simulator()
            self.sim.setCameraResolution(640, 480)
            self.sim.setCameraVFOV(math.radians(60))
            self.sim.setRenderingEnabled(False)
            self.sim.setDiscretizedViewingAngles(True)

            self.train_data = self.load_datasets(['train'])
            self.val_seen_data = self.load_datasets(['val_seen'])
            self.val_unseen_data = self.load_datasets(['val_unseen'])

            # load connectivity graphs of scans in train_data
            self.scans = set([d['scan'] for d in self.train_data + self.val_unseen_data])
            self.graphs = self.load_nav_graphs(self.scans)
            self.calc_paths_and_distances()
            self.generate_json_data('target_train.json', self.train_data)
            self.generate_json_data('target_val_seen.json', self.val_seen_data)
            self.generate_json_data('target_val_unseen.json', self.val_unseen_data)
            if use_aug:
                self.aug_data = self.load_datasets(['aug_paths_full'])
                self.generate_json_data('target_aug_paths_full.json', self.aug_data)
        else:
            self.load_json_data(data_path)

        self.make_dict()

    def load_datasets(self, splits):
        """

        :param splits: A list of split.
            if the split is "something@5000", it will use a random 5000 data from the data
        :return:
        """
        import random
        data = []
        for split in splits:
            # Load Json
            # if split in ['train', 'val_seen', 'val_unseen', 'test',
            #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
            if "/" not in split:
                with open('tasks/R2R/data/R2R_%s.json' % split) as f:
                    new_data = json.load(f)
            else:
                with open(split) as f:
                    new_data = json.load(f)

            # Join
            data += new_data
        print('done reading splits: %s' % splits)
        print('total instructions: %d' % len(data))
        return data

    def load_nav_graphs(self, scans):
        ''' Load connectivity graph for each scan '''

        def distance(pose1, pose2):
            ''' Euclidean distance between two graph poses '''
            return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                    + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                    + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

        graphs = {}
        for scan in scans:
            with open('connectivity/%s_connectivity.json' % scan) as f:
                G = nx.Graph()  # 创建空的无向图
                positions = {}
                data = json.load(f)
                for i, item in enumerate(data):
                    if item['included']:
                        for j, conn in enumerate(item['unobstructed']):
                            if conn and data[j]['included']:
                                positions[item['image_id']] = np.array([item['pose'][3],
                                                                        item['pose'][7], item['pose'][11]]);
                                assert data[j]['unobstructed'][i], 'Graph should be undirected'
                                G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
                nx.set_node_attributes(G, values=positions, name='position')
                graphs[scan] = G
        return graphs

    def get_target_pos(self, scanId, src_vpId, tgt_vpId, back_vpId):
        def _loc_distance(loc):  # 计算相对距离（关于 heading 和 elevation）
            if loc is not None:
                return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
            else:
                return 1e8

        min_distance = 1e5
        back_min_distance = 1e5
        target_heading = 0
        target_elevation = 0
        target_viewId = 0
        back_target_heading = 0
        back_target_elevation = 0
        back_target_viewId = 0

        for viewId in range(36):
            if viewId == 0:  # 开一个新的 sim 实例
                self.sim.newEpisode(scanId, src_vpId, 0, math.radians(-30))
            elif viewId % 12 == 0:  # 向上、向右看，第一个参数 0 是 index，表示 viewpoint 不变
                self.sim.makeAction(0, 1.0, 1.0)
            else:  # 向右看
                self.sim.makeAction(0, 1.0, 0)

            state = self.sim.getState()

            for nb in state.navigableLocations[1:]:
                if nb.viewpointId == tgt_vpId:
                    distance = _loc_distance(nb)
                    if distance < min_distance:
                        min_distance = distance
                        target_heading = state.heading + nb.rel_heading
                        target_elevation = state.elevation + nb.rel_elevation
                        target_viewId = viewId
                elif nb.viewpointId == back_vpId:
                    distance = _loc_distance(nb)
                    if distance < back_min_distance:
                        back_min_distance = distance
                        back_target_heading = state.heading + nb.rel_heading
                        back_target_elevation = state.elevation + nb.rel_elevation
                        back_target_viewId = viewId

        return (target_viewId, target_heading, target_elevation), \
               (back_target_viewId, back_target_heading, back_target_elevation)

    def calc_paths_and_distances(self):
        self.paths = {}
        self.distances = {}
        print('start computing graphs')
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        print('done computing graphs')

    def generate_json_data(self, file_path='data.json', dataset=None):
        self.json_data = []
        print('generate json data and save to: tasks/R2R/data/%s' % file_path)
        with tqdm(total=len(dataset)) as pbar:
            assert dataset is not None
            for datum in dataset:
                scanId = datum['scan']
                path = datum['path']

                init_heading = datum['heading']
                instructions = datum['instructions']
                path_id = datum['path_id']

                for i in range(len(path)):
                    item = {}
                    cur_vpId = path[i]
                    next_vpId = path[i + 1] if i != len(path) - 1 else path[i]
                    back_vpId = path[i - 1] if i != 0 else path[i]
                    target_pos = self.get_target_pos(scanId, cur_vpId, next_vpId, back_vpId)
                    item['scan'] = scanId
                    item['viewpointId'] = cur_vpId
                    item['heading'] = init_heading if i == 0 else self.json_data[-1]['target_heading']
                    item['next_viewpointId'] = next_vpId
                    # 路径终点的 target 信息记为和上一个 viewpoint 的 target 一致（即假定 forward 进入场景就算到达目标）
                    item['target_viewId'] = target_pos[0][0] if i != len(path) - 1 else self.json_data[-1]['target_viewId']
                    item['target_heading'] = target_pos[0][1] if i != len(path) - 1 else self.json_data[-1]['target_heading']
                    item['target_elevation'] = target_pos[0][2] if i != len(path) - 1 else self.json_data[-1]['target_elevation']
                    # 起点的 back_target 信息先全部置 0
                    item['back_target_viewId'] = target_pos[1][0] if i != 0 else 0
                    item['back_target_heading'] = target_pos[1][1] if i != 0 else 0
                    item['back_target_elevation'] = target_pos[1][2] if i != 0 else 0
                    item['path_id'] = path_id
                    self.json_data.append(item)
                # 路径起点的 back_target 信息更新
                self.json_data[-len(path)]['back_target_viewId'] = self.json_data[-len(path) + 1]['back_target_viewId']
                self.json_data[-len(path)]['back_target_heading'] = self.json_data[-len(path) + 1]['back_target_heading']
                self.json_data[-len(path)]['back_target_elevation'] = self.json_data[-len(path) + 1]['back_target_elevation']
                pbar.update(1)

        with open('tasks/R2R/data/%s' % file_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)

    def load_json_data(self, data_path):
        if type(data_path) is list:
            for path in data_path:
                with open(path, 'r') as f:
                    self.json_data += json.load(f)
        elif type(data_path) is str:
            with open(data_path, 'r') as f:
                self.json_data = json.load(f)

    def make_dict(self):
        # not fully implemented yet
        if len(self.json_data) == 0:
            print('json_data is empty.')
            return
        for d in self.json_data:
            key = '%s_%s_%s' % (d['scan'], d['viewpointId'], d['path_id'])
            value = {
                'next_viewpointId': d['next_viewpointId'],
                'target_viewId': d['target_viewId'],
                'target_heading': d['target_heading'],
                'target_elevation': d['target_elevation'],
            }
            self.data_dict[key] = value
        # print('done generating data dict')

if __name__ == '__main__':
    pdg = PretrainDataGenerator()