import os
from time import sleep
import argparse
import torch
import torch.nn as nn
import torchreid
from tools.extract_part_based_features import extract_reid_features
from torchreid.data.data_augmentation import masks_preprocess_all
from torchreid.data.datasets import get_image_dataset
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity, Writer
)

from scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs, display_config_diff
)
from torchreid.utils.engine_state import EngineState


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler, writer, engine_state):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state
            )

        elif cfg.loss.name == 'triplet':
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state
            )

        elif cfg.loss.name == 'part_based':
            engine = torchreid.engine.ImagePartBasedEngine(
                datamanager,
                model,
                optimizer=optimizer,
                loss_name=cfg.loss.part_based.name,
                config=cfg,
                margin=cfg.loss.triplet.margin,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
                dist_combine_strat=cfg.test.part_based.dist_combine_strat,
                batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
                mask_filtering_training=cfg.model.bpbreid.mask_filtering_training,
                mask_filtering_testing=cfg.model.bpbreid.mask_filtering_testing
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.save_dir:
        cfg.data.save_dir = args.save_dir
    if args.inference_enabled:
        cfg.inference.enabled = args.inference_enabled
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms
    if args.job_id:
        cfg.project.job_id = args.job_id


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        '--save_dir', type=str, default='', help='path to output root dir'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    parser.add_argument(
        '--job-id',
        type=int,
        default=None,
        help='Slurm job id'
    )
    parser.add_argument(
        '--inference-enabled',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--epoch', type=int, default=5, help='number of epochs'
    )

    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    default_cfg_copy = cfg.clone()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        cfg.project.config_file = os.path.basename(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    ################################
    cfg.train.max_epoch = args.epoch
    ################################

    # set parts information (number of parts K and each part name),
    # depending on the original loaded masks size or the transformation applied:
    compute_parts_num_and_names(cfg)
    display_config_diff(cfg, default_cfg_copy)

    #print(os.path.join(cfg.data.save_dir, str(cfg.project.job_id)))


    cfg.data.save_dir = os.path.join(cfg.data.save_dir, str(cfg.project.job_id))
    os.makedirs(cfg.data.save_dir)

    if cfg.project.debug_mode:
        torch.autograd.set_detect_anomaly(True)

    logger = Logger(cfg)
    writer = Writer(cfg)

    set_random_seed(cfg.train.seed)

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)
    engine_state = EngineState(cfg.train.start_epoch, cfg.train.max_epoch)
    writer.init_engine_state(engine_state, cfg.data.parts_num)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        config=cfg
    )

    logger.add_model(model)

    num_params, flops = compute_model_complexity(
        model, cfg
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler, writer, engine_state)
    print('Starting experiment {} with job id {} and creation date {}'.format(cfg.project.experiment_id, cfg.project.job_id, cfg.project.start_time))
    engine.run(**engine_run_kwargs(cfg))
    print('End of experiment {} with job id {} and creation date {}'.format(cfg.project.experiment_id, cfg.project.job_id, cfg.project.start_time))

    if cfg.inference.enabled:
        print("Starting inference on external data")
        extract_reid_features(cfg, cfg.inference.input_folder, cfg.data.save_dir, model)


def compute_parts_num_and_names(cfg):
    mask_config = get_image_dataset(cfg.data.sources[0]).get_masks_config(cfg.data.masks_dir)
    if cfg.loss.name == 'part_based':
        if (mask_config is not None and mask_config[1]) or cfg.data.masks.preprocess == 'none':
            # ISP masks or no transform
            cfg.data.parts_num = mask_config[0]
            cfg.data.parts_names = mask_config[3]
        else:
            masks_transform = masks_preprocess_all[cfg.data.masks.preprocess]
            cfg.data.parts_num = masks_transform.parts_num
            cfg.data.parts_names = masks_transform.parts_names


if __name__ == '__main__':
    main()
