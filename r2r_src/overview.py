# from param import args
# import numpy as np
# class Depth_Features():
#     def __init__(self):
#         key_array = np.load(args.depth_index_file)
#         value_array = np.load(args.depth_value_file)
#         depth_map = {}
#         for key, value in zip(key_array, value_array):
#             depth_map["{}_{}".format(key[0], key[1])] = value
#         self.depth_map = depth_map
#
#     def get(self, scan_viewpointid):
#         return self.depth_map[scan_viewpointid]
#
# depth_feature = Depth_Features()
# print(list(depth_feature.depth_map)[:3])
#
# value1= depth_feature.get('17DRP5sb8fy_0e92a69a50414253a23043758f111cec')
# print(type(value1))
# print(value1.shape)

# import torch
# from fusion import MutanFusion
#
# opt1 = {
#     'dim_hv' : 8,
#     'dim_hq' : 8,
#     'dim_mm' :4,
#     'R': 2,
#     'dropout_hv': 0.5,
#     'dropout_hq': 0.5
# }
# mutan =  MutanFusion(opt1, False, False)
#
# v = torch.ones(2,8)
# q = torch.ones(2,8)
# z = mutan(v,q)
# print("z.shape", z.shape)

from utils import BTokenizer
tok = BTokenizer(encoding_length=80)
s = tok.split_sentence("left")
print(s)
s = tok.encode_sentence("left")
print(s)