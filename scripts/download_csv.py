from datetime import datetime

import pandas as pd
import wandb
api = wandb.Api()
from pandas.io.json import json_normalize

# Project is specified by <entity/project-name>
runs = api.runs("vlsomers/Dev-Person-Re-Identification")

metrics_keys_list = [
    'Other/epoch',
    'r1_global',
    'mAP_global',
]

config_keys_list = [
    'project.job_id',
    'project.start_time',

    'model.bpbreid.backbone',
    'model.bpbreid.learnable_attention_enabled',
    'data.masks_dir',
    'model.bpbreid.test_use_target_segmentation',
    'data.sources',
    'data.transforms',
    'data.masks.mask_filtering_threshold',
    'model.bpbreid.mask_filtering_training',
    'model.bpbreid.mask_filtering_testing',
    'model.bpbreid.training_binary_visibility_score',
    'model.bpbreid.testing_binary_visibility_score',
    'data.masks.background_computation_strategy',
    'data.masks.preprocess',
    'data.parts_num',
    'model.bpbreid.dim_reduce',
    'model.bpbreid.dim_reduce_output',
    'model.bpbreid.normalization',
    'model.bpbreid.pooling',
    'test.normalize_feature',
    'test.part_based.dist_combine_strat',
    'train.batch_size',
    'loss.part_based.name',
    'data.masks.softmax_weight',

    'model.bpbreid.concat_parts_embeddings',
    'loss.part_based.weights.globl.ce',
    'loss.part_based.weights.globl.tr',
    'loss.part_based.weights.foreg.ce',
    'loss.part_based.weights.foreg.tr',
    'loss.part_based.weights.conct.ce',
    'loss.part_based.weights.conct.tr',
    'loss.part_based.weights.concat_parts.ce',
    'loss.part_based.weights.concat_parts.tr',
    'loss.part_based.weights.parts.ce',
    'loss.part_based.weights.parts.tr',
    'loss.part_based.weights.pixls.ce',
    'model.bpbreid.shared_parts_id_classifier',
    'model.bpbreid.test_embeddings',

    'data.height',
    'data.width',
    'test.dist_metric',
    'loss.triplet.margin',
]

metrics_keys = set(metrics_keys_list)
config_keys = set(config_keys_list)

keys_list = metrics_keys_list + config_keys_list

comparison_table = []


summary_list, config_list, name_list = [], [], []
print("LENGTH = {}".format(len(runs)))
for i, run in enumerate(runs):
    print(i)

    metrics = run.summary._json_dict
    metrics = json_normalize(metrics)
    metrics = dict(metrics)
    metrics = {k: v for k, v in metrics.items() if k in metrics_keys}

    line = []
    for k in metrics_keys_list:
        if k in metrics:
            line.append(metrics[k].values[-1])
        else:
            line.append('')


    config= run.config.items()
    config = json_normalize(dict(config))
    config = dict(config)
    config = {k: v for k, v in config.items() if k in config_keys}
    for k in config_keys_list:
        if k in config:
            line.append(config[k].values[-1])
        else:
            line.append('')


    comparison_table.append(line)

runs_df = pd.DataFrame(comparison_table)

date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%MS")
runs_df.to_csv("~/Downloads/deep_person_reid_{}.csv".format(date), header=keys_list, sep=";")







# config_keys_list = [
#     'project.job_id',
#     'project.start_time',
#
#     'model.pretrained',
#     'model.bpbreid.pooling',
#     'model.bpbreid.normalization',
#     'model.bpbreid.mask_filtering_training',
#     'model.bpbreid.mask_filtering_testing',
#     'model.bpbreid.last_stride',
#     'model.bpbreid.dim_reduce',
#     'model.bpbreid.dim_reduce_output',
#     'model.bpbreid.backbone',
#     'model.bpbreid.learnable_attention_enabled',
#     'model.bpbreid.test_embeddings',
#     'model.bpbreid.test_use_target_segmentation',
#     'model.bpbreid.training_binary_visibility_score',
#     'model.bpbreid.testing_binary_visibility_score',
#     'model.bpbreid.shared_parts_id_classifier',
#
#     'data.type',
#     'data.root',
#     'data.sources',
#     'data.targets',
#     'data.workers',
#     'data.split_id',
#     'data.height',
#     'data.width',
#     'data.combineall',
#     'data.transforms',
#     'data.norm_mean',
#     'data.norm_std',
#     'data.save_dir',
#     'data.load_train_targets',
#     'data.masks_dir',
#     'data.masks_suffix',
#     'data.parts_num',
#     'data.masks',
#     'data.masks.preprocess',
#     'data.masks.softmax_weight',
#     'data.masks.background_computation_strategy',
#     'data.masks.mask_filtering_threshold',
#
#     'train.batch_size',
#     'train.fixbase_epoch',
#     'train.staged_lr',
#     'train.new_layers',
#     'train.base_lr_mult',
#     'train.lr_scheduler',
#     'train.stepsize',
#     'train.gamma',
#     'train.print_freq',
#     'train.seed',
#     'train.eval_freq',
#     'train.batch_debug_freq',
#     'train.batch_log_freq',
#
#     'loss.part_based.name',
#     'loss.part_based.weights.global.ce',
#     'loss.part_based.weights.global.tr',
#     'loss.part_based.weights.foreground.ce',
#     'loss.part_based.weights.foreground.tr',
#     'loss.part_based.weights.concat_parts.ce',
#     'loss.part_based.weights.concat_parts.tr',
#     'loss.part_based.weights.parts.ce',
#     'loss.part_based.weights.parts.tr',
#     'loss.part_based.weights.pixels.ce',
#     'loss.softmax.label_smooth',
#     'loss.triplet.margin',
#     'loss.triplet.weight_t',
#     'loss.triplet.weight_x',
#
#     'test.batch_size',
#     'test.batch_size_pairwise_dist_matrix',
#     'test.dist_metric',
#     'test.normalize_feature',
#     'test.ranks',
#     'test.evaluate',
#     'test.start_eval',
#     'test.rerank',
#     'test.visrank',
#     'test.visrank_topk',
#     'test.visrank_count',
#     'test.visrank_q_idx_list',
#     'test.vis_feature_maps',
#     'test.visrank_per_body_part',
#     'test.vis_embedding_projection',
#     'test.save_features',
#     'test.part_based.dist_combine_strat',
# ]