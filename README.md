<!-- TODO
- add badges (see StrongSORT repo): arxiv + paper with code + license 



-->

# BPBReID
**A strong baseline for body part-based (occluded) person re-identification**


>**[StrongSORT: Make DeepSORT Great Again](https://arxiv.org/abs/2202.13514)**
>
>Yunhao Du, Yang Song, Bo Yang, Yanyun Zhao, Fei Su
>
>[*arxiv 2202.13514*](https://arxiv.org/abs/2202.13514)

## News

- [yyyy.mm.dd] 

## What's next
We plan on extending BPBReID in the near future, star/watch the repo to stay tuned:
- part-based video/tracklet reid
- reid for multi-object tracking
- ...

## Introduction
Welcome to the official repository for our WACV23 paper "_Body Part-Based Representation Learning for Occluded Person Re-Identification_".
In this work, we proposed BPBReID, a part-based method for person re-identification.
A person is represented as mutli
Please have a look at section XXX for for information about part-based ReID methods.
Simple yet effective architecture, can be used as a strong baseline for further research on part-based methods.
Three main components: 
- body part attention maps
- part-based features
- part-based visibility scores for occlusions
- part-based GiLt training loss 

## What to find in this repository
In this repository, we provide several adaptations to the official Torchreid framework to support part-based ReID methods: 
- A tool to visualize a query-gallery ranking, with pairwise body part distances and parts heatmaps
- A part-based engine which supports multiple features per image sample
- A fully configurable part-based loss to adopt any combination of triplet/id loss on part-based and holistic reid features
- Dataloader to load external information for training/inference, such as keypoints, human parsing, segmentation masks, ... 
- Evaluation code to compute query-gallery distance matrix based on part-based features and related visibility scores

## Installation
Make sure [conda](https://www.anaconda.com/distribution/) is installed.

    # clone this repository
    git clone xxx

    # create conda environment
    cd bpbreid/ # enter project folder
    conda create --name bpbreid python=3.7
    conda activate bpbreid
    
    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt
    
    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    
    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop

## Inference

## Training
Training configs for three datasets (Market-1501, Occluded-Duke and DukeMTMC-reID) and three backbones (ResNet-50, ResNet-50-IBN, HRNet-W32) are provided in the 'configs/bpbreid/' folder. A training procedure can be launched with:
conda activate bpbreid_wacv23;
python ./scripts/main.py --config-file configs/bpbreid/BPBreID_occluded_duke_resnet50_ibn_train.yam

## Download human parsing labels
Should be automatic when you run a test/training script for the first time
You can still download these labels manually at: ...
+ image with samples

## Download pre-trained models

## Tools
- visualization tool + sample image

## Documentation
Well documented model configuration in default_config.py and .yml, have a look at torchreid documentation for more information
Here we provide further details for the most important parts of the configuration:
- configure loss: cfg.loss.part_based
- visualization tool
- BPBReID model configuration: cfg.model.bpbreid

## Part-based methods for ReID
- what is part-based method + diagram (powerpoint slides)
- why part-based method useful
- list of other relevant part-based methods

## Torchreid
Our code is based on the popular [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) framework for person re-identification.
You can also have a look at the original [Torchreid README](Torchreid_original_README.rst) for additional information, such as documentation, how-to instructions, etc.
Difference with original Torchreid:
- Albumentation used for data augmentation
- Support for Wandb and other logger type in the loggger class
- Add an EngineState class to keep track of training epoch, etc
- New ranking visualization tool to display part heatmaps, local distance and other metrics
- ...

## Questions and suggestions
If you have any question or suggestion, please raise a GitHub issue in this repository, I'll be glab to help you as much as I can!

## Citation
If you use this repository for your research or wish to refer to our method BPBReID, please use the following BibTeX entry:
```
@article{
}
```

## Acknowledgement
This work is based on Torchreid
