#!/bin/bash

# TODO fix /home/vso/datasets/reid/Occluded_Duke_backup
# chmod +x /home/vso/projects/deep-person-reid/sh/dataset_migration.sh
# /home/vso/projects/deep-person-reid/sh/dataset_migration.sh /home/vso/datasets/reid/Occluded_Duke_backup "query bounding_box_test bounding_box_train"
set -x

DATASET_DIR=$1
SAMPLES_FOLDERS=$2

# backup dataset
cp -r $DATASET_DIR "${DATASET_DIR}_backup"

# fix dataset structure
echo "fix dataset structure"
mkdir $DATASET_DIR/images
find $DATASET_DIR/ -maxdepth 1 -mindepth 1 -not -name images -print0 | xargs -0 mv -t $DATASET_DIR/images/
mkdir $DATASET_DIR/masks
mkdir $DATASET_DIR/external_annotations

for FOLDER in $SAMPLES_FOLDERS; do
  mkdir -p $DATASET_DIR/masks/pifpaf/$FOLDER/
  find $DATASET_DIR/images/$FOLDER/ -name '*.npy' | xargs -n 1 -I path mv -t $DATASET_DIR/masks/pifpaf/$FOLDER/ path
done

# generate pifpaf annotations
echo "generate pifpaf annotations"
source /home/vso/.virtualenvs/openpifpaf/bin/activate

for FOLDER in $SAMPLES_FOLDERS; do
  mkdir -p $DATASET_DIR/external_annotations/pifpaf/$FOLDER
  python3 -m openpifpaf.predict $DATASET_DIR/images/$FOLDER/*.jpg \
  --image-output $DATASET_DIR/external_annotations/pifpaf/$FOLDER/ \
  --line-width 2 --dpi-factor 3 --figure-width 2 \
  --show-joint-confidences --show-frontier-order \
  --fields-output $DATASET_DIR/external_annotations/pifpaf/$FOLDER/ \
  --json-output $DATASET_DIR/external_annotations/pifpaf/$FOLDER/ \
  --checkpoint shufflenetv2k30w \
  --force-complete-pose --monocolor-connections
done


# generate detectron2 annotations
source ~/opt/anaconda3/etc/profile.d/conda.sh
source /home/vso/.virtualenvs/detectron2/bin/activate

for FOLDER in $SAMPLES_FOLDERS; do
  mkdir -p $DATASET_DIR/external_annotation/detectron_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv_conf_tresh_025/$FOLDER
  python /home/vso/projects/detectron2/demo/demo.py \
  --config-file /home/vso/projects/detectron2/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml \
  --input $DATASET_DIR/images/$FOLDER/*.jpg \
  --output $DATASET_DIR/external_annotation/detectron_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv_conf_tresh_025/$FOLDER \
  --confidence-threshold 0.25 \
  --opts MODEL.WEIGHTS detectron2://Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl
done
