import torch
import torchvision.models as models
import os,sys
import numpy as np
from  matplotlib import pyplot as plt
from tqdm import tqdm

pwd = os.path.abspath('.')
MP3D_build_path = os.path.join(pwd, 'MP3D_Sim', 'build')
DASA_path = os.path.join(pwd, 'DASA')
sys.path.append(MP3D_build_path)
os.chdir(DASA_path)

import MatterSim

VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 2048
BATCH_SIZE = 9
MODEL = 'data/resnet152.pth'
OUTFILE = 'data/ResNet-152-imagenet-depth'

# Simulator image parameters
WIDTH=640
HEIGHT=480
VFOV=60
GPU_ID = 1
vpids = np.load('viewpointIds.npy')

def normalizaiton(img):
    _range = np.max(img)-np.min(img)
    return (img-np.min(img))/(_range+1e-6)

def load_model():
    resnet152 = models.resnet152(pretrained=False)
    resnet152.load_state_dict(torch.load(MODEL))
    torch.cuda.set_device(GPU_ID)
    del resnet152.fc
    resnet152.fc=lambda x:x
    resnet152 = resnet152.cuda()
    return resnet152

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(np.radians(VFOV))
sim.setDepthEnabled(True)
sim.setDiscretizedViewingAngles(True)
sim.setBatchSize(1)
sim.initialize()

model = load_model()
feats = []
for vpid in tqdm(vpids):
    scanId = vpid[0]
    viewpointId = vpid[1]
    depth = []
    for ix in range(VIEWPOINT_SIZE):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [np.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix
        depth.append(normalizaiton(state.depth))

    input_depth = torch.from_numpy(np.array(depth)).float()
    input_depth = input_depth.repeat(1,1,1,3).permute(0,3,1,2).cuda()

    feat = []
    with torch.no_grad():
        for i in range(VIEWPOINT_SIZE//BATCH_SIZE):
            b_feat  = model(input_depth[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            feat.append(b_feat.cpu())
        feat = torch.stack(feat,dim=0).view(-1,2048)

    feats.append(feat.numpy())

np.save(OUTFILE,np.array(feats))