from tests.losses.baselines.old_bp_loss import OldBodyPartsLoss


class OldBPLossIntraIdMean(OldBodyPartsLoss):

    def __init__(self, **kwargs):
        super(OldBPLossIntraIdMean, self).__init__(**kwargs)

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist):
        pairwise_dist = per_body_part_pairwise_dist.mean(0)
        return pairwise_dist

    def compute_test_dist_matrix(self, qf, gf):
        body_parts_dist_matrices = self._compute_body_parts_test_dist_matrices(qf, gf)
        return body_parts_dist_matrices.mean(dim=0)
