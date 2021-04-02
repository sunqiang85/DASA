''' Batched Room-to-Room navigation environment '''

import sys

sys.path.append('buildpy36')
import MatterSim
import csv
import numpy as np
import math
import utils
import random
import networkx as nx
from param import args
from tqdm import tqdm

from utils import load_datasets, load_nav_graphs, load_pretrain_datasets
from generate_pretrain_data import PretrainDataGenerator

csv.field_size_limit(sys.maxsize)


class Depth_Features():
    def __init__(self):
        key_array = np.load(args.depth_index_file)
        value_array = np.load(args.depth_value_file)
        depth_map = {}
        for key, value in zip(key_array, value_array):
            depth_map["{}_{}".format(key[0], key[1])] = value
        self.depth_map = depth_map

depth_features = Depth_Features()

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:  # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]  # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def getDepthFeatures(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if args.dfeatures:
                feature = depth_features.depth_map[long_id]  # Get feature for
                feature_states.append(feature)
            else:
                feature_states.append(None)
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
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
            datasetLoader = load_pretrain_datasets if args.train == 'pretrain' else load_datasets
            data = datasetLoader([split])
            with tqdm(total=len(data), position=0, leave=True, ascii=True) as pbar:
                pbar.set_description('load instructions in [%s]' % split)
                for item in data:
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
                    pbar.update(1)
        print("len(self.data)", len(self.data))
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

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
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))


    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix + batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId  # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, dfeature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading = (viewId % 12) * math.radians(30)  # e.g. 30
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
                d_feat = dfeature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading  # e.g. 60 + 30 (relative to current view pointidx)
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)  # relate, abs
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,  # candiadate relateive to viewId
                            'elevation': loc_elevation,  # abs evelation
                            "normalized_heading": state.heading + loc.rel_heading,  # abs heading
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId,  # Next viewpoint id
                            'pointId': ix,  # current viewIdx
                            'distance': distance,
                            'idx': j + 1,  # candidate index
                            'feature': np.concatenate((visual_feat, angle_feat), -1),  # current view angle feature
                            'dfeature': np.concatenate((d_feat, angle_feat), -1)
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
                d_feat = dfeature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new['dfeature'] = np.concatenate((d_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):

        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        obs = []
        dfeatures = self.env.getDepthFeatures()
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, dfeatures[i], state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)  # abs agnle
            d_feature = np.concatenate((dfeatures[i], self.angle_feature[base_view_id]), -1)  # abs agnle

            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scanId,  # scan
                'viewpoint': state.location.viewpointId,  # current viewpointId
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'dfeature': d_feature,
                'candidate': candidate,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': self._shortest_path_action(state, item['path'][-1]),  # next viewpointId
                'back_teacher': self._shortest_path_action(state, item['path'][0]),
                'path_id': item['path_id'],
            })
            if 'target_viewId' in item:
                obs[-1]['target_viewId'] = item['target_viewId']
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]

            obs[-1]['total_distance'] = self.distances[state.scanId][item['path'][0]][item['path'][-1]]
            obs[-1]['progress'] = 1 - (obs[-1]['distance'] / (obs[-1]['total_distance']+1e-10))
            obs[-1]['path'] = item['path']


            if args.use_action_seq:
                target_key = '%s_%s_%s' % (obs[-1]['scan'], obs[-1]['viewpoint'], obs[-1]['path_id'])
                obs[-1]['action_seq'], obs[-1]['last_action_seq'] = self.get_action_sequnce(
                    obs[-1]['viewIndex'],
                    obs[-1]['target_viewId'],
                    isStart=obs[-1]['viewpoint'] == item['path'][0],
                    isEnd=obs[-1]['viewpoint'] == item['path'][-1]
                )


                if target_key in self.target_dict:
                    pass
                    # target = self.target_dict[target_key]
                    # obs[-1]['target_viewId'] = target['target_viewId']
                    # obs[-1]['progress'] = target['progress']
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

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:  # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:  # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:  # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def random_start_reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:  # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:  # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:  # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [random.choice(item['path']) for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

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

        assert cur_viewId == tgt_viewId
        action_seq.append('forward')
        last_action_seq += action_seq[:-1]
        return action_seq, last_action_seq
