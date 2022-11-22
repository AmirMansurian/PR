#!/bin/bash
conda activate torchreid

cd ..
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_test.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_0.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_1.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_2.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_3.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_4.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_5.yaml
/home/vso/anaconda3/envs/torchreid/bin/python -u /home/vso/projects/deep-person-reid/scripts/main.py --config-file /home/vso/projects/deep-person-reid/configs/gridsearch/resnet50_market1501_pose_train_6.yaml