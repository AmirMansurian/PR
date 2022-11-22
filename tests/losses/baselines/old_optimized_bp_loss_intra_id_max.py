from __future__ import division, absolute_import

from tests.losses.baselines.old_optimized_bp_loss import OldOptimizedBodyPartsLoss


class OldOptimizedBPLossIntraIdMax(OldOptimizedBodyPartsLoss):
    """
    """

    def __init__(self, **kwargs):
        super(OldOptimizedBPLossIntraIdMax, self).__init__(**kwargs)

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist):
        return per_body_part_pairwise_dist.max(0, keepdim=True)[0]

    def _combine_body_parts_test_dist_matrices(self, per_body_part_pairwise_dist):
        return per_body_part_pairwise_dist.max(dim=0)[0]
