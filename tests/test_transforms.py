import unittest
from scripts.default_config import get_default_config
from scripts.main import build_datamanager
from torchreid.data.transforms import build_transforms
from torchreid.utils import Writer, Logger, read_image, read_masks
from torchreid.utils.engine_state import EngineState
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from albumentations import (
    Resize, Compose, Normalize, ColorJitter, HorizontalFlip, CoarseDropout,
    DualTransform, RandomCrop, PadIfNeeded, functional  # HueSaturationValue for ColorJitter
)

class TestTransforms(unittest.TestCase):
    def test_transforms(self):
        # config
        cfg = get_default_config()
        cfg.project.logger.use_clearml = False
        cfg.project.logger.use_tensorboard = False
        cfg.project.logger.use_wandb = False
        cfg.project.logger.use_neptune = False
        cfg.project.logger.matplotlib_show = True
        cfg.use_gpu = False
        cfg.data.root = "~/datasets/reid"
        dataset = 'synergy'
        cfg.data.transforms = ['rc']
        # cfg.data.transforms = ['color_jitter', 'random_crop', 'random_flip', 'random_erase']
        cfg.data.sources = [dataset]
        cfg.train.batch_size = 6
        cfg.loss.name = 'part_based'

        # writer
        logger = Logger(cfg)
        writer = Writer(cfg)
        engine_state = EngineState(0, 60)
        writer.init_engine_state(engine_state, 36)
        writer.engine_state.epoch = writer.engine_state.max_epoch - 1
        writer.engine_state.batch = 0

        # dataset
        datamanager = build_datamanager(cfg)
        train_loader = datamanager.train_loader
        scales = [5, 8, 10, 15]
        for batch_idx, data in enumerate(train_loader):
            scale = scales[batch_idx]
            batch_imgs = data[0]
            batch_masks = data[4]
            img_paths = data[3]
            batch_size = batch_imgs.shape[0]
            fig, axs = plt.subplots(batch_size, batch_masks.shape[1]+2, figsize=(40, 8))
            for i in range(0, batch_size):
                base_img = read_image(img_paths[i])
                axs[i, 0].imshow(base_img, interpolation='nearest')

                img = batch_imgs[i]
                axs[i, 1].imshow(img.permute(1, 2, 0), interpolation='nearest')

                masks = batch_masks[i]
                base_masks = masks.numpy()
                masks = F.softmax(scale * masks.reshape((masks.shape[0], -1)), dim=1).reshape(masks.shape).numpy()
                axs[i, 0].set_axis_off()
                axs[i, 1].set_axis_off()
                for j in range(0, masks.shape[0]):
                    axs[i, j+2].imshow(masks[j], cmap=plt.get_cmap('jet'), interpolation='nearest')
                    axs[i, j+2].set_axis_off()
            plt.axis('off')
            # plt.show()
            # plt.waitforbuttonpress()
            plt.savefig("/Users/vladimirsomers/Downloads/myplot_transforms_transformed_masks_norm_scale_{}.jpg".format(scale))
            plt.close()
            break
            # batch_imgs = data[0]
            # batch_masks = data[4]
            # img_paths = data[3]
            # batch_size = batch_imgs.shape[0]
            # fig, axs = plt.subplots(batch_size, batch_masks.shape[1]+2, figsize=(40, 8))
            # for i in range(0, batch_size):
            #     base_img = read_image(img_paths[i])
            #     axs[i, 0].imshow(base_img, interpolation='nearest')
            #
            #     img = batch_imgs[i]
            #     axs[i, 1].imshow(img.permute(1, 2, 0), interpolation='nearest')
            #
            #     masks = read_masks(img_paths[i] + cfg.data.masks_suffix)
            #     masks = np.transpose(masks, (2, 0, 1))
            #     axs[i, 0].set_axis_off()
            #     axs[i, 1].set_axis_off()
            #     for j in range(0, masks.shape[0]):
            #         axs[i, j+2].imshow(masks[j], cmap=plt.get_cmap('jet'), interpolation='nearest')
            #         axs[i, j+2].set_axis_off()
            # plt.axis('off')
            # # plt.show()
            # # plt.waitforbuttonpress()
            # plt.savefig("/Users/vladimirsomers/Downloads/myplot_transforms_original_masks.jpg")
            # break

    def test_build_transforms(self):

        # config
        cfg = get_default_config()
        cfg.project.logger.use_clearml = False
        cfg.project.logger.use_tensorboard = False
        cfg.project.logger.use_wandb = False
        cfg.project.logger.use_neptune = False
        cfg.project.logger.matplotlib_show = True
        cfg.use_gpu = False
        # cfg.loss.name = 'part_based'
        cfg.data.root = "~/datasets/reid"
        dataset = 'synergy'
        cfg.data.sources = [dataset]
        cfg.data.ro.path = "/Users/vladimirsomers/datasets/other/VOCdevkit/VOC2012"
        cfg.data.masks_dir = "pifpaf"
        # cfg.data.targets = [dataset]
        # cfg.data.transforms = ['random_flip', 'random_erase'] # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
        # cfg.data.transforms = [] # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
        cfg.train.batch_size = 6
        cfg.loss.name = 'softmax'
        cfg.data.workers = 0

        # writer
        logger = Logger(cfg)
        writer = Writer(cfg)
        engine_state = EngineState(0, 60)
        writer.init_engine_state(engine_state, 36)
        writer.engine_state.epoch = writer.engine_state.max_epoch - 1
        writer.engine_state.batch = 0

        transform_tr, transform_te, parts_num = build_transforms(
            256,
            128,
            cfg,
            16,
            # transforms=[], # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
            transforms=['ro'], # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            use_masks=True
        )

        transform_tr = Compose(transform_tr[:2] + transform_tr[3:])  # remove normalize

        # dataset
        datamanager = build_datamanager(cfg)
        train_loader = datamanager.train_loader
        for self.batch_idx, data in enumerate(train_loader):
            imgs_path = data["img_path"]
            fig, axs = plt.subplots(len(imgs_path), 39, figsize=(40, 8))
            for i, img_path in enumerate(imgs_path):
                or_img = read_image(img_path)
                masks_suffix = datamanager.get_mask_suffix(dataset, True)
                masks_path = img_path + masks_suffix
                masks = read_masks(masks_path)
                result = transform_tr(image=or_img, mask=masks)
                masks = result['mask']
                # masks = [masks[i] for i in range(0, masks.shape[0])]
                # result = transform_tr(image=img, masks=masks)
                # masks = result['mask']
                img = result['image']

                axs[i, 0].imshow(or_img)
                # norm_mean = np.array([0.485, 0.456, 0.406])
                # norm_std = np.array([0.229, 0.224, 0.225])
                # unnormalize = Normalize(mean=(-norm_mean / norm_std).tolist(), std=(1.0 / norm_std).tolist())
                # img = unnormalize(image=img.permute(1, 2, 0).numpy())['image']
                # axs[i, 1].imshow(img)
                axs[i, 1].imshow(img.permute(1, 2, 0))
                axs[i, 0].set_axis_off()
                for j in range(0, masks.shape[0]):
                    axs[i, j+2].imshow(masks[j], cmap=plt.get_cmap('jet'))
                    axs[i, j+2].set_axis_off()
            plt.axis('off')
            # plt.show()
            # plt.waitforbuttonpress()
            plt.savefig("/Users/vladimirsomers/Downloads/myplot.jpg")
            break

    def test_random_occlusion(self):

        # config
        cfg = get_default_config()
        cfg.project.logger.use_clearml = False
        cfg.project.logger.use_tensorboard = False
        cfg.project.logger.use_wandb = False
        cfg.project.logger.use_neptune = False
        cfg.project.logger.matplotlib_show = True
        cfg.use_gpu = False
        cfg.data.ro.p = 1.
        cfg.data.ro.n = 1
        cfg.data.ro.min_overlap = 0.6
        cfg.data.ro.max_overlap = 0.9
        # cfg.loss.name = 'part_based'
        cfg.data.root = "~/datasets/reid"
        dataset = 'synergy'
        cfg.data.sources = [dataset]
        cfg.data.ro.path = "/Users/vladimirsomers/datasets/other/VOCdevkit/VOC2012"
        cfg.data.masks_dir = "pifpaf"
        # cfg.data.targets = [dataset]
        # cfg.data.transforms = ['random_flip', 'random_erase'] # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
        # cfg.data.transforms = [] # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
        cfg.train.batch_size = 10
        cfg.loss.name = 'softmax'
        cfg.data.workers = 0

        # writer
        logger = Logger(cfg)
        writer = Writer(cfg)
        engine_state = EngineState(0, 60)
        writer.init_engine_state(engine_state, 36)
        writer.engine_state.epoch = writer.engine_state.max_epoch - 1
        writer.engine_state.batch = 0

        transform_tr, transform_te, parts_num = build_transforms(
            256,
            128,
            cfg,
            16,
            # transforms=[], # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
            transforms=['ro'], # 'color_jitter', 'random_patch', 'random_crop', 'random_flip', 'random_erase'
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            use_masks=True
        )

        transform_tr = Compose(transform_tr[:2] + transform_tr[3:])  # remove normalize

        # dataset
        datamanager = build_datamanager(cfg)
        train_loader = datamanager.train_loader
        plt.figure(figsize=(40, 20))
        images_amout = 50
        count = 0
        for self.batch_idx, data in enumerate(train_loader):
            imgs_path = data["img_path"]
            for i, img_path in enumerate(imgs_path):
                count += 1
                or_img = read_image(img_path)
                masks_suffix = datamanager.get_mask_suffix(dataset, True)
                masks_path = img_path + masks_suffix
                masks = read_masks(masks_path)
                result = transform_tr(image=or_img, mask=masks)
                img = result['image']

                plt.subplot(5, 10, count)
                plt.imshow(img.permute(1, 2, 0))
            if count >= images_amout:
                break
        plt.axis('off')
        plt.savefig("/Users/vladimirsomers/Downloads/myplot_occlusions.jpg")
