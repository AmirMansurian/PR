import os
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import keyword

from scripts.default_config import get_default_config
from scripts.main import build_datamanager
from torchreid.utils import Writer, Logger
from torchreid.utils.engine_state import EngineState
from torchreid.utils.visualization.display_batch_triplets import show_triplet, show_triplet_grid
from torchreid.utils.visualization.feature_map_visualization import display_feature_maps


class TestVisualizationTool(unittest.TestCase):
    def test_show_triplet(self):
        base_path = "/Users/vladimirsomers/datasets/reid/synergy_small/train/5ec16ca355e4e150e0745be4/"

        anc_image_path = base_path + "t2p4/f1_bb12.png"
        anc_masks_path = base_path + "t2p4/f1_bb12.png.confidence_fields.npy"

        pos_image_path = base_path + "t2p4/f0_bb8.png"
        pos_masks_path = base_path + "t2p4/f0_bb8.png.confidence_fields.npy"

        neg_image_path = base_path + "t2p2/f0_bb6.png"
        neg_masks_path = base_path + "t2p2/f0_bb6.png.confidence_fields.npy"

        anc_image = self.load_image(anc_image_path)
        anc_masks = np.load(anc_masks_path)

        pos_image = self.load_image(pos_image_path)
        pos_masks = np.load(pos_masks_path)

        neg_image = self.load_image(neg_image_path)
        neg_masks = np.load(neg_masks_path)

        # (image, masks, id, body_part_name)
        anc = (anc_image, anc_masks[7], 5, 'left_elbow')
        pos = (pos_image, pos_masks[7], 5, 'left_elbow')
        neg = (neg_image, neg_masks[7], 12, 'left_elbow')

        pos_dist = 1.76544
        neg_dist = 3.65268

        show_triplet(anc, pos, neg, pos_dist, neg_dist)

    def test_show_triplet_grid(self):
        base_path = "/Users/vladimirsomers/datasets/reid/synergy_small/train/5ec16ca355e4e150e0745be4/"

        anc_image_path = base_path + "t2p4/f1_bb12.png"
        anc_masks_path = base_path + "t2p4/f1_bb12.png.confidence_fields.npy"

        pos_image_path = base_path + "t2p4/f0_bb8.png"
        pos_masks_path = base_path + "t2p4/f0_bb8.png.confidence_fields.npy"

        neg_image_path = base_path + "t2p2/f0_bb6.png"
        neg_masks_path = base_path + "t2p2/f0_bb6.png.confidence_fields.npy"

        anc_image = self.load_image(anc_image_path)
        anc_masks = np.load(anc_masks_path)

        pos_image = self.load_image(pos_image_path)
        pos_masks = np.load(pos_masks_path)

        neg_image = self.load_image(neg_image_path)
        neg_masks = np.load(neg_masks_path)

        pos_dist = 1.76544
        neg_dist = 3.65268

        # (image, masks, id, body_part_name)
        anc = (anc_image, anc_masks[7], 5, 'left_elbow')
        pos = (pos_image, pos_masks[7], 5, 'left_elbow')
        neg = (neg_image, neg_masks[7], 12, 'left_elbow')
        triplet = [pos, anc, neg, pos_dist, neg_dist]
        triplets = [triplet]*20 # display 20 triplets

        show_triplet_grid(triplets)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def test_embeddings(self):
        cfg = get_default_config()
        cfg.data.save_dir = "/Users/vladimirsomers/Code/logs/deep-person-reid/tests"
        cfg.project.logger.use_clearml = False
        cfg.project.logger.use_tensorboard = True
        cfg.project.logger.matplotlib_show = False
        logger = Logger(cfg)

        meta = []
        while len(meta) < 100:
            meta = meta + keyword.kwlist  # get some strings
        meta = meta[:100]

        for i, v in enumerate(meta):
            meta[i] = v + str(i)

        label_img = torch.rand(100, 3, 10, 32)
        for i in range(100):
            label_img[i] *= i / 100.0
        logger.add_embeddings("", torch.randn(100, 5), meta, label_img, 30)

    def test_display_feature_maps(self):

        # config
        cfg = get_default_config()
        cfg.project.logger.use_clearml = False
        cfg.project.logger.use_tensorboard = False
        cfg.project.logger.use_wandb = False
        cfg.project.logger.use_neptune = False
        cfg.project.logger.matplotlib_show = True
        cfg.use_gpu = False
        cfg.loss.name = 'part_based'
        cfg.data.root = "~/datasets/reid"
        cfg.data.sources = ['synergy']
        cfg.data.targets = ['synergy']

        # writer
        logger = Logger(cfg)
        writer = Writer(cfg)
        engine_state = EngineState(0, 60)
        writer.init_engine_state(engine_state, 36)
        writer.engine_state.epoch = writer.engine_state.max_epoch - 1
        writer.engine_state.batch = 0

        # dataset
        datamanager = build_datamanager(cfg)
        gallery_dataset = datamanager.test_loader["synergy"]['gallery'].dataset

        size = 25
        size = min(size, len(gallery_dataset))
        imgs_path = []
        body_part_masks = []
        pids = []
        for i in range(0, size):
            gallery_sample = gallery_dataset[i]
            qpid, qcamid, qimg_path, qmasks = gallery_sample['qpid'],  gallery_sample['qcamid'],  gallery_sample['qimg_path'],  gallery_sample['qmasks'],
            imgs_path.append(qimg_path)
            body_part_masks.append(qmasks)
            pids.append(qpid)

        body_part_masks = torch.stack(body_part_masks).unsqueeze(2) # torch.Size([9, 38, 1, 16, 8])

        d = 256
        body_parts_features = torch.randn((size, body_part_masks.shape[1], d)) # torch.Size([9, 38, 2048])
        spatial_features = torch.randn((size, 1, d, 16, 8)) # torch.Size([9, 1, 2048, 16, 8])
        display_feature_maps(body_parts_features, spatial_features, body_part_masks, imgs_path, pids)


    def test_sample_and_heatmaps_and_both(self):

        # config
        cfg = get_default_config()
        cfg.project.logger.use_clearml = False
        cfg.project.logger.use_tensorboard = False
        cfg.project.logger.use_wandb = False
        cfg.project.logger.use_neptune = False
        cfg.project.logger.matplotlib_show = True
        cfg.use_gpu = False
        cfg.loss.name = 'part_based'
        cfg.data.root = "~/datasets/reid"
        dataset = 'market1501'
        cfg.data.sources = [dataset]
        cfg.data.targets = [dataset]
        cfg.data.transforms = ['combine_pifpaf_into_four_body_masks_no_overlap'] # 'combine_pifpaf_into_four_body_masks_no_overlap', 'combine_pifpaf_into_four_body_masks'

        # writer
        logger = Logger(cfg)
        writer = Writer(cfg)
        engine_state = EngineState(0, 60)
        writer.init_engine_state(engine_state, 36)
        writer.engine_state.epoch = writer.engine_state.max_epoch - 1
        writer.engine_state.batch = 0

        # dataset
        datamanager = build_datamanager(cfg)
        gallery_dataset = datamanager.test_loader[dataset]['query'].dataset

        size = 15
        size = min(size, len(gallery_dataset))
        width, height = (128, 256)
        mask_width, mask_height = (8, 16)
        clip = True
        for i in range(0, size):
            _, pid, _, img_path, masks = gallery_dataset[i]
            masks = masks.numpy()
            img = cv2.imread(img_path)
            img = cv2.resize(img, (width, height))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            filename = os.path.basename(img_path)
            filename = "{}_{}".format(dataset, os.path.splitext(filename)[0])
            dir = os.path.join("/home/vso/experiments/sample_and_heatmaps_and_both/")
            Path(dir).mkdir(parents=True, exist_ok=True)

            img_file_path = os.path.join(dir, "{}_sample.jpg".format(filename))
            cv2.imwrite(img_file_path, img)
            for i in range(0, masks.shape[0]):
                original_mask = masks[i]

                # BIG MASK
                mask = cv2.resize(original_mask, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                if clip:
                    mask = np.clip(mask, 0, 1)
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
                mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

                mask_file_path = os.path.join(dir, "{}_mask_{}.jpg".format(filename, i))
                cv2.imwrite(mask_file_path, mask_color)

                # SMALL MASK
                small_mask = cv2.resize(original_mask, dsize=(mask_width, mask_height), interpolation=cv2.INTER_CUBIC)
                small_mask_high_res = cv2.resize(small_mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
                if clip:
                    small_mask = np.clip(small_mask, 0, 1)
                    small_mask = (small_mask * 255).astype(np.uint8)
                else:
                    small_mask = np.interp(small_mask, (small_mask.min(), small_mask.max()), (0, 255)).astype(np.uint8)
                small_mask_color = cv2.applyColorMap(small_mask, cv2.COLORMAP_JET)

                small_mask_file_path = os.path.join(dir, "{}_small_smask_{}.png".format(filename, i))
                cv2.imwrite(small_mask_file_path, small_mask_color)


                if clip:
                    small_mask_high_res = np.clip(small_mask_high_res, 0, 1)
                    small_mask_high_res = (small_mask_high_res * 255).astype(np.uint8)
                else:
                    small_mask_high_res = np.interp(small_mask_high_res, (small_mask_high_res.min(), small_mask_high_res.max()), (0, 255)).astype(np.uint8)
                small_mask_high_res_color = cv2.applyColorMap(small_mask_high_res, cv2.COLORMAP_JET)

                small_mask_high_res_file_path = os.path.join(dir, "{}_small_smask_high_res_{}.jpg".format(filename, i))
                cv2.imwrite(small_mask_high_res_file_path, small_mask_high_res_color)

                # MASKED IMG
                masked_img = cv2.addWeighted(img, 0.5, mask_color, 0.5, 0)
                masked_img_file_path = os.path.join(dir, "{}_masked_img__{}.jpg".format(filename, i))
                cv2.imwrite(masked_img_file_path, masked_img)




