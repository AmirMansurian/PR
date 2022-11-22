import unittest
import numpy as np

from scripts.default_config import get_default_config
from torchreid.utils.distribution import plot_distributions, plot_body_parts_pairs_distance_distribution, \
    plot_pairs_distance_distribution

from torchreid.utils import Logger
import matplotlib.pyplot as plt

class TestDistributionPlot(unittest.TestCase):
    cfg = get_default_config()
    cfg.project.logger.use_clearml = False
    cfg.project.logger.use_tensorboard = False
    cfg.project.logger.matplotlib_show = True
    logger = Logger(cfg)

    def test_plot_distributions(self):
        mu, sigma = 2, 0.2  # mean and standard deviation
        neg_p = np.random.normal(mu, sigma, 100000)
        mu, sigma = 1, 0.3  # mean and standard deviation
        pos_p = np.random.normal(mu, sigma, 1000)

        fig, ax = plt.subplots()
        plot_distributions(ax, neg_p, pos_p, pos_p.mean(), 0, neg_p.mean(), 0)
        self.logger.add_figure("Pairs distance distribution", fig, 0)
        plt.close(fig)

    def test_plot_body_parts_pairs_distance_distribution(self):
        m = 31
        q_size = 80
        g_size = 256
        body_part_pairwise_dist = np.random.rand(m, q_size, g_size) * 500
        q_pids = np.random.randint(low=0, high=8, size=[q_size])
        g_pids = np.random.randint(low=0, high=32, size=[g_size])
        plot_body_parts_pairs_distance_distribution(body_part_pairwise_dist, q_pids, g_pids, "Test")

    def test_plot_pairs_distance_distribution(self):
        q_size = 80
        g_size = 256
        pairwise_dist = np.random.rand(q_size, g_size) * 500
        q_pids = np.random.randint(low=0, high=8, size=[q_size])
        g_pids = np.random.randint(low=0, high=32, size=[g_size])
        plot_pairs_distance_distribution(pairwise_dist, q_pids, g_pids, "Test")
