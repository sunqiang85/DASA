import os
import sys
import re
sys.path.append('build')
import MatterSim
import string
import json
import time
import math
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from param import args
import torch.nn.functional as F
from param import args
from tqdm import tqdm


def dump_datasets(splits, scan_ids):
    """

    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if args.mini:
            number = 40
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            with open('tasks/R2R/data/R2R_%s.json' % split) as f:
                new_data = json.load(f)
        else:
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]


        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    print('read data from %s with %d items' % (splits, len(data)))

    filter_data = [c for c in new_data if c['scan'] in scan_ids][:100]
    print("filter_data", split, len(filter_data))
    with open('tasks/R2R/mini_data/R2R_%s.json' % split, 'w') as f:
        json.dump(filter_data, f, indent=1)
    return data


def read_img_features(feature_store, scan_ids):
    import csv
    import base64
    from tqdm import tqdm

    # print("Start loading the image feature")
    start = time.time()
    csv.field_size_limit(sys.maxsize)

    if "detectfeat" in args.features:
        views = int(args.features[10:])
    else:
        views = 36

    args.views = views
    print("input scan_ids", scan_ids)
    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
    features = {}
    features_index = []
    features_value = []
    with tqdm(total=10567, position=0, leave=True, ascii=True) as pbar:
        pbar.set_description("Start loading the image feature")
        with open(feature_store, "r") as tsv_in_file:     # Open the tsv file.
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:
                if item['scanId']  in scan_ids:
                    # print("item['scanId']",type(item['scanId']))
                    long_id = "{}_{}".format(item['scanId'], item['viewpointId'])
                    # print("long_id", long_id)
                    # print('scan_ids', scan_ids)

                    features_index.append(long_id)
                    ft= np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                  dtype=np.float32).reshape((views, -1))
                    features_value.append(ft)
    print("len(features", len(features))

    print("Finish Loading the image feature from %s in %0.4f seconds" % (feature_store, time.time() - start))
    np.save("tasks/R2R/mini_data/img_feature_index.npy", features_index)
    np.save("tasks/R2R/mini_data/img_feature_value.npy", features_value)
    index_set = set([c.split('_')[0] for c in features_index])
    print("len(index_set)", len(index_set))
    print(index_set)


    return features


def dump_depth_features(scan_ids):
    key_array = np.load(args.depth_index_file)
    value_array = np.load(args.depth_value_file)

    filtered_keys = []
    filtered_values = []
    for key, value in zip(key_array, value_array):
        if key[0] in scan_ids:
            filtered_keys.append(key)
            filtered_values.append(value)
    np.save("tasks/R2R/mini_data/viewpointIds.npy", np.array(filtered_keys))
    np.save("tasks/R2R/mini_data/ResNet-152-imagenet-depth.npy", np.array(filtered_values))



if __name__ == '__main__':
    print("start")
    scan_map = {}
    total_scan_ids = []
    for split in ['train', 'val_seen', 'val_unseen', 'test', 'aug_paths']:
        with open('tasks/R2R/data/R2R_%s.json' % split) as f:
            new_data = json.load(f)
            scan_map[split] = set([c['scan'] for c in new_data])

        for k,v in scan_map.items():
            print(k,len(v))
        scan_ids = list(scan_map[split])[:1]
        dump_datasets([split], scan_ids)
        total_scan_ids = total_scan_ids + scan_ids
    total_scan_ids = list(set(total_scan_ids))
    print("len(total_scan_ids)",len(total_scan_ids))
    print(total_scan_ids)

    feature_store = 'img_features/ResNet-152-imagenet.tsv'
    read_img_features(feature_store, total_scan_ids)
    dump_depth_features(total_scan_ids)