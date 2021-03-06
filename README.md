# DASA
## statement
Our code is based on the framework of [PREVALENT](https://github.com/weituo12321/PREVALENT_R2R) 
And modified according to the paper: Depth-guided AdaIN and Shift Attention Network for Vision-and-Language Navigation


## Envrionment Construction
Following the instructions in [EnvDrop](https://github.com/airsplay/R2R-EnvDrop) to build the envriroment. 
And use the r2r_src folder contains the main code of the project.


## Task data
Down load the [R2R task data](https://drive.google.com/file/d/1Pr9SnZOhFew4diMREbGtbqgj5DL_ux86/view?usp=sharing), and place in the folder: tasks/R2R/data


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



### train
Train the model without updating vilbert.
```bash
CUDA_VISIBLE_DEVICES=2 python  r2r_src/train.py --agent_type dg --adaIn_type channel --attn soft --train auglistener \
--mlWeight_org 0.4 \
--mlWeight_aug 1.2 \
--ab_type a --a_type sigmoid \
--d_vl_layers 3 \
--env_drop_stage after_adain \
--depth_drop \
--use_shift --shift_kernel_size 5 \
--warm_steps 1000 --decay_intervals 2000 --decay_start 4000 --lr_decay 0.2 \
--log_every 100 --val_every 2000 --use_lr_scheduler \
--selfTrain --aug tasks/R2R/data/aug_paths.json --speaker snap/speaker/state_dict/best_val_unseen_bleu \
--pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 \
--angleFeatSize 128 --accumulateGrad --featdropout 0.4 --feedback sample --subout max --optim rms --lr 0.0001 \
--iters 20000 --maxAction 35 --encoderType Dic --batchSize 20 --include_vision True --use_dropout_vision True \
--d_enc_hidden_size 1024 --critic_dim 1024 --name shift5_dga_sigmoid_vl3_ml2
```

### finetune
Finetune the whole model.
```bash
CUDA_VISIBLE_DEVICES=2 python  r2r_src/train.py --agent_type dg --adaIn_type channel --attn soft --train auglistener \
--load snap/shift5_dga_sigmoid_vl3_ml2/state_dict/LAST_iter20000 \
--d_update_add_layer True \
--mlWeight_org 0.4 \
--mlWeight_aug 1.2 \
--ab_type a --a_type sigmoid \
--d_vl_layers 3 \
--env_drop_stage after_adain \
--depth_drop \
--log_every 100 --val_every 1000 \
--use_shift --shift_kernel_size 5 \
--selfTrain --aug tasks/R2R/data/aug_paths.json --speaker snap/speaker/state_dict/best_val_unseen_bleu \
--pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 \
--angleFeatSize 128 --accumulateGrad --featdropout 0.4 --feedback sample --subout max --optim rms --lr 0.000002 \
--iters 30000 --maxAction 35 --encoderType Dic --batchSize 2 --include_vision True --use_dropout_vision True \
--d_enc_hidden_size 1024 --critic_dim 1024 --name shift5_dga_sigmoid_vl3_ml2_fine 
```


### validation
validate the model on the validation dataset. The result will be print out. The trained model can be downloaded from [here](https://drive.google.com/file/d/1EYGlT9uonY2MgY1giiJR4updRXKXbvXF/view?usp=sharing)
```bash
CUDA_VISIBLE_DEVICES=2 python  r2r_src/train.py --agent_type dg --adaIn_type channel --attn soft --train validlistener --submit \
--load snap/shift5_dga_sigmoid_vl3_ml2_fine/state_dict/best_val_unseen \
--d_update_add_layer True \
--mlWeight_org 0.4 \
--mlWeight_aug 1.2 \
--ab_type a --a_type sigmoid \
--d_vl_layers 3 \
--env_drop_stage after_adain \
--depth_drop \
--log_every 100 --val_every 1000 \
--use_shift --shift_kernel_size 5 \
--selfTrain --aug tasks/R2R/data/aug_paths.json --speaker snap/speaker/state_dict/best_val_unseen_bleu \
--pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 \
--angleFeatSize 128 --accumulateGrad --featdropout 0.4 --feedback sample --subout max --optim rms --lr 0.000002 \
--iters 30000 --maxAction 35 --encoderType Dic --batchSize 2 --include_vision True --use_dropout_vision True \
--d_enc_hidden_size 1024 --critic_dim 1024 --name shift5_dga_sigmoid_vl3_ml2_fine
```


