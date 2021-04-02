import torch
from env import EnvBatch
from generate_pretrain_data import PretrainDataGenerator
from utils import load_datasets, get_all_point_angle_feature, new_simulator, angle_feature, load_nav_graphs, load_pretrain_datasets
import random
import numpy as np
import math
import networkx as nx
from param import args


class ValidBatch():
    def __init__(self, feature_store, batch_size=100, seed=10, splits=['val_seen'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        self.target_dict = {}
        for split in splits:
            # target = PretrainDataGenerator('tasks/R2R/data/target_%s.json' % split)
            # self.target_dict.update(target.data_dict)
            datasetLoader = load_pretrain_datasets if args.train == 'pretrain' else load_datasets
            for item in datasetLoader([split]):
                # Split multiple instructions into separate entries
                for j, instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:  # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self.angle_feature = get_all_point_angle_feature()
        self._load_nav_graphs()
        self.sim = new_simulator()
        self.buffered_state_dict = {}
        self.make_validation_data()

    def make_validation_data(self):
        self.validation_data = []
        for item in self.data:
            for i in range(len(item['path'])):
                new_item = {
                    'scan': item['scan'],
                    'viewpoint': item['path'][i],
                    'heading': 0,
                    'elevation': 0,
                    'target_viewId': item['target_viewId'],
                    'instructions': item['instructions'],
                    'instr_encoding': item['instr_encoding'],
                    'path': item['path'],
                    'path_id': item['path_id'],
                    'instr_id': item['instr_id'],
                    'teacher': item['path'][i + 1] if i < len(item['path']) - 1 else item['path'][i]
                }
                self.validation_data.append(new_item)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def reset_epoch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.validation_data)
        self.ix = 0

    def _next_minibatch(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        batch = self.validation_data[self.ix: self.ix+batch_size]
        self.ix += batch_size
        self.batch = batch

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30) # e.g. 30
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading  # relative agent heading e.g. 90 - 30
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading #  e.g. 60 + 30 (relative to current view pointidx)
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation) # relate, abs
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading, #  candiadate relateive to viewId
                            'elevation': loc_elevation, # abs evelation
                            "normalized_heading": state.heading + loc.rel_heading, # abs heading
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix, # current viewIdx
                            'distance': distance,
                            'idx': j + 1, # candidate index
                            'feature': np.concatenate((visual_feat, angle_feat), -1) # current view angle feature
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)  # abs agnle
            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scanId,  # scan
                'viewpoint': state.location.viewpointId,  # current viewpointId
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'candidate': candidate,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': item['teacher'],  # next viewpointId
                'path_id': item['path_id'],
                'target_viewId': item['target_viewId']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]

            obs[-1]['total_distance'] = self.distances[state.scanId][item['path'][0]][item['path'][-1]]
            obs[-1]['progress'] = 1 - (obs[-1]['distance'] / obs[-1]['total_distance'])
            obs[-1]['path'] = item['path']
            target_key = '%s_%s_%s' % (obs[-1]['scan'], obs[-1]['viewpoint'], obs[-1]['path_id'])
            obs[-1]['action_seq'], obs[-1]['last_action_seq'] = self.get_action_sequnce(
                obs[-1]['viewIndex'],
                obs[-1]['target_viewId'],
                isStart=obs[-1]['viewpoint'] == item['path'][0],
                isEnd=obs[-1]['viewpoint'] == item['path'][-1]
            )
            if target_key in self.target_dict:
                pass
                # obs[-1]['action_seq'], obs[-1]['last_action_seq'] = self.get_action_sequnce(
                #     obs[-1]['viewIndex'],
                #     obs[-1]['target_viewId'],
                #     isStart=obs[-1]['viewpoint'] == item['path'][0],
                #     isEnd=obs[-1]['viewpoint'] == item['path'][-1]
                # )

            else:
                if obs[-1]['viewpoint'] == item['path'][-1]:  # reach the goal
                    obs[-1]['target_viewId'] = -1
                    obs[-1]['target_heading'] = 0  # to do later
                    obs[-1]['target_elevation'] = 0  # to do later
                else:
                    min_dist = 100000  # max
                    for ix in range(36):
                        if ix == 0:
                            self.sim.newEpisode(obs[-1]['scan'], obs[-1]['viewpoint'], 0, math.radians(-30))
                        elif ix % 12 == 0:
                            self.sim.makeAction(0, 1.0, 1.0)
                        else:
                            self.sim.makeAction(0, 1.0, 0)

                        state = self.sim.getState()
                        assert state.viewIndex == ix
                        for j, loc in enumerate(state.navigableLocations[1:]):
                            if loc.viewpointId == obs[-1]['teacher']:
                                distance = _loc_distance(loc)
                                if distance < min_dist:
                                    obs[-1]['target_viewId'] = ix
                                    obs[-1]['target_heading'] = state.heading
                                    obs[-1]['target_elevation'] = state.elevation
                                    min_dist = distance
        return obs

    def get_valid_batch(self):
        for i in range(len(self.validation_data) // self.batch_size):
            self._next_minibatch()
            scanIds = [item['scan'] for item in self.batch]
            viewpointIds = [item['path'][0] for item in self.batch]
            self.env.newEpisodes(scanIds, viewpointIds, [0] * self.batch_size)
            yield self._get_obs()

    def get_action_sequnce(self, cur_viewId, tgt_viewId, isStart=False, isEnd=False):
        # calculate action sequence
        # return: action_seq, last_action_seq
        action_seq = []
        last_action_seq = []
        if isEnd:
            return ['<end>'], ['forward']
        elif isStart:
            last_action_seq.append('<start>')  # 当前点是路径的起始点
        else:
            last_action_seq.append('forward')  # 当前点是路径的中间节点，上一个动作应该是 foward

        # 计算上下
        tgt_elev = tgt_viewId // 12
        cur_elev = cur_viewId // 12
        up_down = tgt_elev - cur_elev
        if up_down > 0:     # 向上看
            action_seq += ['up'] * up_down
        elif up_down < 0:   # 向下看
            action_seq += ['down'] * (-up_down)
        cur_viewId += up_down * 12
        cur_elev += up_down
        assert cur_elev == tgt_elev
        # 计算左右
        tgt_head = tgt_viewId % 12
        cur_head = cur_viewId % 12

        turn_right = tgt_head - cur_head if tgt_head > cur_head else tgt_head + 12 - cur_head
        turn_left = cur_head + 12 - tgt_head if tgt_head > cur_head else cur_head - tgt_head

        if turn_right <= turn_left:      # 选择右转
            action_seq += ['right'] * turn_right
            cur_viewId += turn_right
            if cur_viewId // 12 > cur_elev:
                cur_viewId -= 12
        elif turn_left < turn_right:    # 选择左转
            action_seq += ['left'] * turn_left
            cur_viewId -= turn_left
            if cur_viewId // 12 < cur_elev:
                cur_viewId += 12

        if cur_viewId != tgt_viewId:
            print(cur_viewId, tgt_viewId, action_seq, last_action_seq)
        assert cur_viewId == tgt_viewId
        action_seq.append('forward')
        last_action_seq += action_seq[:-1]
        return action_seq, last_action_seq