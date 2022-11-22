from __future__ import division, absolute_import

from tests.losses.baselines import OldBodyPartsLoss


class OldBPLossIntraPartTriplets(OldBodyPartsLoss):

    def __init__(self, **kwargs):
        super(OldBPLossIntraPartTriplets, self).__init__(**kwargs)

    def forward(self, body_parts_features, targets):
        body_parts_dist_matrices = self._compute_body_parts_dist_matrices(body_parts_features, self.epsilon)  # body_parts_features.shape = [M, N, N]

        m = body_parts_dist_matrices.shape[0]
        loss = 0.
        for i in range(0, m):
            dist = body_parts_dist_matrices[i]
            loss += self._hard_mine_triplet_loss(dist, targets, self.ranking_loss)
        loss /= m

        return loss
