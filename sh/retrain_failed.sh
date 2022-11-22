#!/bin/bash
conda activate torchreid

cd ..
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_4.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_6.yaml
