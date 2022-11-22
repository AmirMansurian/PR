#!/bin/bash

source ~/opt/anaconda3/etc/profile.d/conda.sh

conda activate torchreid

WORKING_DIR=$HOME/Code/deep-person-reid

python "$WORKING_DIR"/tools/visualize_actmap.py \
--root "$HOME"/datasets/reid \
-d synergy \
-m osnet_x1_0 \
--weights "$WORKING_DIR"/pretrained_models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth \
--save-dir "$WORKING_DIR"/log/visactmap_osnet_x1_0

python "$WORKING_DIR"/tools/visualize_actmap.py \
--root "$HOME"/datasets/reid \
-d market1501 \
-m osnet_x1_0 \
--weights "$WORKING_DIR"/pretrained_models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth \
--save-dir "$WORKING_DIR"/log/visactmap_osnet_x1_0

