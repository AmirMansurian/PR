import unittest

import numpy as np

from scripts.default_config import get_default_config
from torchreid.utils import Writer


class TestPlotting(unittest.TestCase):
    cfg = get_default_config()
    cfg.project.logger.use_clearml = False
    cfg.project.logger.use_tensorboard = False
    cfg.project.logger.matplotlib_show = True
    writer = Writer(cfg)

    def test_qg_body_part_distances_boxplot(self):
        body_part_pairwise_dist = np.random.rand(36, 80, 256) * 500
        self.writer.qg_body_part_distances_boxplot(body_part_pairwise_dist)

    def test_qg_body_part_distances_boxplot_invalid_distances(self):
        body_part_pairwise_dist = np.random.rand(36, 80, 256) * 500
        body_part_pairwise_dist[body_part_pairwise_dist > 400] = -1
        self.writer.qg_body_part_distances_boxplot(body_part_pairwise_dist)

    def test_qg_body_part_availability_barplot(self):
        gf_parts_visibility = np.random.rand(80, 36) > 0.2
        qf_parts_visibility = np.random.rand(256, 36) > 0.2
        self.writer.qg_body_part_availability_barplot(gf_parts_visibility, qf_parts_visibility)

    def test_qg_body_part_pairs_availability_barplot(self):
        body_part_pairwise_dist = np.random.rand(36, 8, 6) * 500
        body_part_pairwise_dist[body_part_pairwise_dist < 100] = -1
        self.writer.qg_body_part_pairs_availability_barplot(body_part_pairwise_dist)

    def test_qg_distribution_of_body_part_availability_histogram(self):
        gf_parts_visibility = np.random.rand(80, 36) > 0.9
        qf_parts_visibility = np.random.rand(256, 36) > 0.3
        self.writer.qg_distribution_of_body_part_availability_histogram(gf_parts_visibility, qf_parts_visibility)
