# train
CUDA_VISIBLE_DEVICES=3 python  r2r_src/train.py --agent_type dg --adaIn_type channel --attn soft --train auglistener \
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
--d_enc_hidden_size 1024 --critic_dim 1024 --name v3/shift5_dga_sigmoid_vl3_ml2 | tee snap/v3/shift5_dga_sigmoid_vl3_ml2/log.txt

# finetune
CUDA_VISIBLE_DEVICES=3 python  r2r_src/train.py --agent_type dg --adaIn_type channel --attn soft --train auglistener \
--load snap/v3/shift5_dga_sigmoid_vl3_ml2/state_dict/LAST_iter20000 \
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
--iters 50000 --maxAction 35 --encoderType Dic --batchSize 2 --include_vision True --use_dropout_vision True \
--d_enc_hidden_size 1024 --critic_dim 1024 --name v3/shift5_dga_sigmoid_vl3_ml2_fine | tee snap/v3/shift5_dga_sigmoid_vl3_ml2_fine/log.txt

# validation
CUDA_VISIBLE_DEVICES=3 python  r2r_src/train.py --agent_type dg --adaIn_type channel --attn soft --train validlistener --submit \
--load snap/icme/shift5_dga_sigmoid_vl3_ml2_fine/state_dict/best_val_unseen \
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
--iters 50000 --maxAction 35 --encoderType Dic --batchSize 2 --include_vision True --use_dropout_vision True \
--d_enc_hidden_size 1024 --critic_dim 1024 --name icme/shift5_dga_sigmoid_vl3_ml2_fine