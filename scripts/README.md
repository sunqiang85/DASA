## Depth features

Download pre-extracted depth feature on [Google Drive](https://drive.google.com/file/d/1RqdQgTwJbH9BqoQoBFeKpQTBFrdKTomq/view?usp=sharing) (2.9G)



Or use following instructions to extract depth features from scratch (Not recommended since it's kinda tricky).

### Build newest Matterport3D Simulator

1. Clone the **newest** Matterport3D Simulator.
   Please make sure it does not overwrite the MP3D simulator used in training phase (which is the older version v0.1).

```bash
git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git MP3D_Sim
```

2. Follow the [instructions](https://github.com/ronghanghu/speaker_follower/blob/master/README_Matterport3DSimulator.md) to build the MP3D simulator.

### Download files

1. Download `matterport_skybox_images` and `undistorted_color_images` of [Matterport3D dataset](https://github.com/niessner/Matterport) and put them into `DASA/data/v1/scans` folders.

2. Download `viewpointIds.npy` on [Google Drive](https://drive.google.com/file/d/1XUz1VwrQfmI_7XdOaGojZykJ-sLz5evc/view?usp=sharing) (2.6M) and put it into `DASA/data` folder.

   This file contains all `[scan_ID, viewpoint_ID]` pairs of images appeared in the R2R dataset.

3. Download `resnet152.pth` from [Google Drive](https://drive.google.com/open?id=0B7fNdx_jAqhtMXU1N0VTZkN1dHc) (230M) and put it into `DASA/data` folder.

   For more information about this `pth` file: https://github.com/ruotianluo/pytorch-resnet

Now your file hierarchy should be like (only relative files are presented):

```
.
|-- MP3D_Sim
|-- DASA
	  -- connectivity/
		-- data/
		   -- resnet152.pth
		   -- v1
			 -- viewpointIds.npy
		-- scripts/
	     -- depth_feat_extractor.py
	     -- enable_depth.py
```

### Convert depth images to skybox

Run `python DASA/scripts/enable_depth.py`

This script concatenates depth images of each panoramic scene and converts them into a skybox representation which will be used in depth feature extraction.

Skybox images will be generated in `data/v1/scans/matterport_skybox_images/[scan_id]/` with file name of `[pano_id]_skybox_depth_small.png` where each `pano_id` indicates a viewpoint.

### Extract depth features

Run `python DASA/scripts/depth_feat_extractor.py`

The output file is generated in `DASA/data/ResNet-152-imagenet-depth.npy`