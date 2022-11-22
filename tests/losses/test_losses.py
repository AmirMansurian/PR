import unittest
import torch
import numpy as np
from tests.losses.baselines import OldBPLossIntraIdMean, OldBPLossIntraPartTriplets
from tests.losses.baselines.old_bp_loss_intra_id_max import OldBPLossIntraIdMax
from tests.losses.baselines.old_optimized_bp_loss import OldOptimizedBodyPartsLoss
from torchreid.losses import deep_supervision, TripletLoss, InterPartsTripletLoss, PartBasedTripletLossMean, PartBasedTripletLossMax
from torchreid.losses.intra_parts_triplet_loss import IntraPartsTripletLoss
import random


def batch_sample_1():
    dim = 1
    head = torch.from_numpy(np.array([[1, 1, 1], [1, 1, 8], [2, 2, 3], [10, 10, 10]])).type(torch.FloatTensor)
    torso = torch.from_numpy(np.array([[101, 101, 101], [101, 101, 108], [102, 102, 103], [110, 110, 110]])).type(
        torch.FloatTensor)
    legs = torch.from_numpy(np.array([[3, 3, 1], [3, 7, 1], [6, 1, 3], [12, 15, 6]])).type(torch.FloatTensor)
    body_parts_features = torch.cat([head.unsqueeze(dim), torso.unsqueeze(dim), legs.unsqueeze(dim)], dim)
    targets = torch.from_numpy(np.array([1, 1, 2, 2])).type(torch.LongTensor)
    parts_visibility = torch.ones([4, 3]).bool()
    return body_parts_features, targets, parts_visibility


def batch_sample_random(n, m, d, ids, masks=False):
    assert ids < n
    body_parts_features = torch.randn((n, m, d))
    labels = torch.randint(low=0, high=ids, size=[n])
    if masks:
        parts_visibility = torch.randint(low=0, high=2, size=[n, m]).bool()
        parts_visibility[0, :] = 0
    else:
        parts_visibility = torch.ones([n, m]).bool()
    return body_parts_features, labels, parts_visibility


def batch_samples():
    torch.manual_seed(0)
    return [batch_sample_random(100, 36, 2048, 10), batch_sample_random(64, 5, 512, 2), batch_sample_1()]


def test_sample_random(qn, gn, qids, gids, m, d, masks=False):
    q = batch_sample_random(qn, m, d, qids, masks)
    g = batch_sample_random(gn, m, d, gids, masks)
    return q, g


def test_samples(masks=False):
    torch.manual_seed(0)
    return [test_sample_random(4, 6, 2, 3, 2, 8, masks), test_sample_random(12, 24, 3, 4, 36, 128, masks)]


def pairwise_dist_matrix_sample_random(n, ids):
    assert (n/ids).is_integer()
    matrix = torch.randn((n, n))
    pairwise_dist = torch.abs((matrix - matrix.t()))
    labels = torch.arange(0, ids).unsqueeze(1).repeat(1, int(n/ids)).flatten().long()
    return pairwise_dist, labels


def pairwise_dist_matrix_sample_1():
    n = 4
    ids = 2
    labels = torch.arange(0, ids).unsqueeze(1).repeat(1, int(n/ids)).flatten().long()
    pairwise_dist = torch.from_numpy(np.array([
        [2, 5, -1, -1],
        [5, 3, 15, 10],
        [-1, 15, 1, -1],
        [-1, 10, -1, 3]
    ])).float()
    return pairwise_dist, labels


def pairwise_dist_matrix_samples():
    torch.manual_seed(0)
    return [pairwise_dist_matrix_sample_random(9, 3), pairwise_dist_matrix_sample_random(100, 10), pairwise_dist_matrix_sample_random(128, 8)]


def pairwise_dist_matrix_with_invalid_distances_samples():
    torch.manual_seed(0)
    return [pairwise_dist_matrix_sample_1()]


class TestLosses(unittest.TestCase):

    ###########################
    #          BPLoss         #
    ###########################
    def test_BPLoss_pairwise_distance_matrix(self):
        for body_parts_features, _, _ in batch_samples():
            baseline_loss = OldBPLossIntraIdMean()
            tested_loss = PartBasedTripletLossMean()
            baseline_result = baseline_loss._compute_body_parts_dist_matrices(body_parts_features, tested_loss.epsilon)
            tested_result = tested_loss._body_parts_pairwise_distance_matrix(body_parts_features, False, tested_loss.epsilon)
            assert torch.allclose(tested_result, baseline_result)

    def test_BPLoss_compute_body_parts_test_dist_matrices(self):
        random.seed(0)
        for _ in range(0, 5):
            qn = random.randint(8, 20)
            gn = random.randint(20, 40)
            qids = random.randint(2, int(qn/4))
            gids = random.randint(2, int(qn/4))

            m = 36
            d = 64

            qf, _, _ = batch_sample_random(qn, m, d, qids)
            gf, _, _ = batch_sample_random(gn, m, d, gids)
            baseline_loss = OldBPLossIntraIdMean()
            baseline_result = baseline_loss._compute_body_parts_test_dist_matrices(qf, gf)
            loss = PartBasedTripletLossMean()
            tested_result = loss._compute_body_parts_test_dist_matrices(qf, gf)
            assert torch.allclose(tested_result, baseline_result)

    def test_BPLoss_compute_test_dist_matrix(self):
        random.seed(0)
        for _ in range(0, 10):
            qn = random.randint(8, 20)
            gn = random.randint(20, 40)
            qids = random.randint(2, int(qn/4))
            gids = random.randint(2, int(qn/4))

            m = 36
            d = 64

            qf, _, _ = batch_sample_random(qn, m, d, qids)
            gf, _, _ = batch_sample_random(gn, m, d, gids)
            baseline = OldBPLossIntraIdMean()
            baseline_result = baseline._compute_test_pairwise_dist_matrix(qf, gf)
            loss = PartBasedTripletLossMean()
            tested_result = loss.compute_test_pairwise_dist_matrix(qf, gf)
            assert torch.allclose(tested_result, baseline_result)

    def test_BPLoss_compute_test_dist_matrix_with_mask_filtering(self):
        random.seed(0)
        for _ in range(0, 10):
            qn = random.randint(8, 20)
            gn = random.randint(20, 40)
            qids = random.randint(2, int(qn/4))
            gids = random.randint(2, int(qn/4))

            m = 36
            d = 64

            qf, _, _ = batch_sample_random(qn, m, d, qids)
            gf, _, _ = batch_sample_random(gn, m, d, gids)
            loss = PartBasedTripletLossMax()
            baseline = OldBPLossIntraIdMax()
            qf_parts_visibility = torch.randint(low=0, high=2, size=[qn, m]).bool()
            gf_parts_visibility = torch.randint(low=0, high=2, size=[gn, m]).bool()
            qf_parts_visibility[0, :] = 0
            gf_parts_visibility[0, :] = 0
            baseline_result = baseline._compute_test_pairwise_dist_matrix(qf, gf)
            tested_result = loss.compute_test_pairwise_dist_matrix(qf, gf, None, None)
            assert torch.allclose(baseline_result, tested_result)
            # tested_result = loss.compute_test_pairwise_dist_matrix(qf, gf, qf_parts_visibility, gf_parts_visibility)
            # assert torch.equal(tested_result[0, :], torch.ones(gn) * torch.finfo(tested_result.dtype).max)
            # assert torch.equal(tested_result[:, 0], torch.ones(qn) * torch.finfo(tested_result.dtype).max)

    def test_BPLoss_hard_mine_triplet_loss(self):
        baseline_loss = OldOptimizedBodyPartsLoss()
        tested_loss = PartBasedTripletLossMax()
        for pairwise_dist, labels in pairwise_dist_matrix_samples():
            baseline_result = baseline_loss._hard_mine_triplet_loss(pairwise_dist, labels, tested_loss.margin)
            tested_result = tested_loss._hard_mine_triplet_loss(pairwise_dist, labels, tested_loss.margin)
            assert torch.equal(baseline_result, tested_result)

    def test_BPLoss_hard_mine_triplet_loss_with_invalid_distances(self):
        loss = PartBasedTripletLossMax()
        for pairwise_dist, labels in pairwise_dist_matrix_with_invalid_distances_samples():
            tested_result = loss._hard_mine_triplet_loss(pairwise_dist, labels, loss.margin)

    ###########################
    #    BPLossIntraIdMax    #
    ###########################
    def test_BPLossIntraIdMax(self):
        for body_parts_features, targets, _ in batch_samples():
            # baseline loss
            baseline_loss = OldBPLossIntraIdMax()
            baseline_result = baseline_loss(body_parts_features, targets)
            # loss to be tested
            tested_result = PartBasedTripletLossMax()(body_parts_features, targets)
            assert torch.allclose(baseline_result, tested_result)

    def test_BPLossIntraIdMax_compute_test_pairwise_dist_matrix(self):
        for query, gallery in test_samples(False):
            qf, qlabels, qparts_visibility = query
            gf, glabels, gparts_visibility = gallery
            # baseline loss
            baseline_loss = OldBPLossIntraIdMax()
            baseline_result = baseline_loss.compute_test_dist_matrix(qf, gf)
            # loss to be tested
            tested_loss = PartBasedTripletLossMax()
            tested_result_1 = tested_loss.compute_test_pairwise_dist_matrix(qf, gf, None, None)
            tested_result_2 = tested_loss.compute_test_pairwise_dist_matrix(qf, gf, qparts_visibility, gparts_visibility)
            assert torch.allclose(baseline_result, tested_result_1)
            assert torch.allclose(baseline_result, tested_result_2)

    def test_BPLossIntraIdMax_compute_test_pairwise_dist_matrix_with_mask_filtering(self):
        for query, gallery in test_samples(True):
            qf, qlabels, qf_parts_visibility = query
            gf, glabels, gf_parts_visibility = gallery

            loss = PartBasedTripletLossMax()

            per_body_part_pairwise_dist = loss._compute_body_parts_test_dist_matrices(qf, gf)

            baseline_result = loss._combine_body_parts_test_dist_matrices(per_body_part_pairwise_dist, None, None)
            tested_result = loss._combine_body_parts_test_dist_matrices(per_body_part_pairwise_dist, qf_parts_visibility, gf_parts_visibility)

    ###########################
    #    BPLossIntraIdMean    #
    ###########################
    def test_BPLossIntraIdMean(self):
        for body_parts_features, targets, _ in batch_samples():
            # baseline loss
            baseline_loss = OldBPLossIntraIdMean()
            baseline_result = baseline_loss(body_parts_features, targets)
            # loss to be tested
            tested_loss = PartBasedTripletLossMean()
            tested_result = tested_loss(body_parts_features, targets)
            assert torch.allclose(baseline_result, tested_result)

    def test_BPLossIntraIdMean_compute_test_pairwise_dist_matrix(self):
        for query, gallery in test_samples(False):
            qf, qlabels, qparts_visibility = query
            gf, glabels, gparts_visibility = gallery
            # baseline loss
            baseline_loss = OldBPLossIntraIdMean()
            baseline_result = baseline_loss.compute_test_dist_matrix(qf, gf)
            # loss to be tested
            tested_loss = PartBasedTripletLossMean()
            tested_result_1, _ = tested_loss.compute_test_pairwise_dist_matrix(qf, gf, None, None)
            tested_result_2, _ = tested_loss.compute_test_pairwise_dist_matrix(qf, gf, qparts_visibility, gparts_visibility)
            assert torch.allclose(baseline_result, tested_result_1)
            assert torch.allclose(baseline_result, tested_result_2)

    # def test_BPLossIntraIdMean_compute_test_dist_matrix_with_mask_filtering(self):
    #     random.seed(0)
    #     for _ in range(0, 10):
    #         qn = random.randint(4, 6)
    #         gn = random.randint(6, 9)
    #         qids = random.randint(2, 3)
    #         gids = random.randint(2, 3)
    #
    #         m = 36
    #         d = 64
    #
    #         qf, _ = batch_sample_random(qn, m, d, qids)
    #         gf, _ = batch_sample_random(gn, m, d, gids)
    #         loss = BPLossIntraIdMean()
    #         baseline = OldBPLossIntraIdMean()
    #         qf_parts_visibility = torch.randint(low=0, high=2, size=[qn, m]).bool()
    #         gf_parts_visibility = torch.randint(low=0, high=2, size=[gn, m]).bool()
    #         qf_parts_visibility[0, :] = 0
    #         gf_parts_visibility[0, :] = 0
    #         baseline_result = baseline._compute_test_pairwise_dist_matrix(qf, gf)
    #         # tested_result = loss.compute_test_pairwise_dist_matrix(qf, gf, None, None)
    #         # assert torch.allclose(baseline_result, tested_result)
    #         tested_result = loss.compute_test_pairwise_dist_matrix(qf, gf, qf_parts_visibility, gf_parts_visibility)
    #         assert torch.equal(tested_result[0, :], torch.ones(gn) * torch.finfo(tested_result.dtype).max)
    #         assert torch.equal(tested_result[:, 0], torch.ones(qn) * torch.finfo(tested_result.dtype).max)

    ###########################
    # BPLossIntraPartTriplets #
    ###########################
    def test_OldBPLossIntraPartTriplets(self):
        for body_parts_features, targets, _ in batch_samples():
            # baseline loss
            baseline_result = deep_supervision(TripletLoss(margin=0.3), body_parts_features.unbind(dim=1), targets)
            # loss to be tested
            tested_loss = OldBPLossIntraPartTriplets(margin=0.3)
            tested_result = tested_loss(body_parts_features, targets)
            assert torch.allclose(baseline_result, tested_result)

    def test_BPLossIntraPartTriplets(self):
        for body_parts_features, targets, _ in batch_samples():
            # baseline loss
            baseline_loss = OldBPLossIntraPartTriplets(margin=0.3)
            baseline_result = baseline_loss(body_parts_features, targets)
            # loss to be tested
            tested_loss = IntraPartsTripletLoss(margin=0.3)
            tested_result = tested_loss(body_parts_features, targets)
            assert torch.allclose(baseline_result, tested_result)

    def test_BPLossIntraPartTriplets_compute_test_pairwise_dist_matrix(self):
        for query, gallery in test_samples(False):
            qf, qlabels, qparts_visibility = query
            gf, glabels, gparts_visibility = gallery
            # baseline loss
            baseline_loss = OldBPLossIntraPartTriplets()
            baseline_result = baseline_loss.compute_test_dist_matrix(qf, gf)
            # loss to be tested
            tested_loss = IntraPartsTripletLoss()
            tested_result_1 = tested_loss.compute_test_pairwise_dist_matrix(qf, gf, None, None)
            tested_result_2 = tested_loss.compute_test_pairwise_dist_matrix(qf, gf, qparts_visibility, gparts_visibility)
            assert torch.allclose(baseline_result, tested_result_1)
            assert torch.allclose(baseline_result, tested_result_2)

    ###########################
    # BPLossInterPartTriplets #
    ###########################
    def test_BPLossInterPartTriplets(self):
        def baseline(dist, targets):
            dist_an, dist_ap = [], []
            nm = dist.shape[0]
            for i in range(nm):
                loss_p = None
                loss_n = None
                for j in range(nm):
                    same_identity = torch.equal(targets[i % len(targets)], targets[j % len(targets)])
                    same_body_part = int(i / len(targets)) == int(j / len(targets))
                    if same_identity and same_body_part:
                        if loss_p is None:
                            loss_p = dist[i, j]
                        else:
                            loss_p = torch.max(loss_p, dist[i, j])
                    elif not same_identity:
                        if loss_n is None:
                            loss_n = dist[i, j]
                        else:
                            loss_n = torch.min(loss_n, dist[i, j])

                dist_ap.append(loss_p.unsqueeze(0))
                dist_an.append(loss_n.unsqueeze(0))

            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)

            y = torch.ones(nm)
            return torch.nn.MarginRankingLoss(margin=margin)(dist_an, dist_ap, y)

        margin = 0.3
        batch_samples = [batch_sample_random(8, 5, 16, 2), batch_sample_random(4, 3, 8, 2), batch_sample_1()]
        for body_parts_features, targets, _ in batch_samples:
            loss = InterPartsTripletLoss(margin=margin)
            dist = loss.compute_mixed_body_parts_dist_matrices(body_parts_features)
            # baseline
            base_loss = baseline(dist, targets)
            # method to be tested
            tested_loss = loss.hard_mine_triplet_loss(dist, targets)

            assert torch.allclose(base_loss, tested_loss)

    def test_BPLossInterPartTriplets_compute_mixed_body_parts_dist_matrices(self):
        for body_parts_features, targets, _ in batch_samples():
            loss = InterPartsTripletLoss(margin=0.3)
            dist = loss.compute_mixed_body_parts_dist_matrices(body_parts_features)

            nm = body_parts_features.shape[0] * body_parts_features.shape[1]

            assert dist.shape == torch.Size([nm, nm])

            def distance(a, b):
                return (a - b).pow(2).sum().sqrt()

            flattened_body_parts_features = body_parts_features.flatten(0, 1)
            random.seed(0)
            for i in range(0, 1000):
                i = random.randint(0, nm-1)
                j = random.randint(0, nm-1)
                assert torch.allclose(dist[i][j],
                                      distance(flattened_body_parts_features[i], flattened_body_parts_features[j]))


if __name__ == '__main__':
    unittest.main()
