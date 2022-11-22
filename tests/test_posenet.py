import unittest
from scripts.default_config import get_default_config
from scripts.main import build_datamanager
from torchreid.data.transforms import build_transforms
from torchreid.models.bpbreid import GlobalAveragePoolingHead, GlobalMaxPoolingHead, GlobalWeightedAveragePoolingHead, \
    GlobalWeightedAveragePoolingHead2, SoftmaxAveragePoolingHead, GlobalWeightedAveragePoolingHead21D, \
    GlobalWeightedAveragePoolingHead1D, GlobalWeightedAveragePoolingHead3D
from torchreid.utils import Writer, Logger, read_image, read_masks, torch
from torchreid.utils.engine_state import EngineState
from matplotlib import pyplot as plt


class GlobalWeightedAveragePoolingHead23D(object):
    pass


class TestBPBreID(unittest.TestCase):
    # TODO WHAT IS NUM FEATURES IN BATCH NORM AND HOW TO USE IT
    def test_heads(self):
        normalization = 'batch_norm_2d'
        depth = 2048
        gap_head = GlobalAveragePoolingHead(normalization, depth)
        gmp_head = GlobalMaxPoolingHead(normalization, depth)
        softmax_head = SoftmaxAveragePoolingHead(normalization, depth)
        gwap_head = GlobalWeightedAveragePoolingHead(normalization, depth)
        gwap_head_same = GlobalWeightedAveragePoolingHead(normalization, depth)
        gwap_head2 = GlobalWeightedAveragePoolingHead2(normalization, depth)

        N, M, D, H, W = 32, 36, 2048, 16, 8

        body_part_masks = torch.rand(size=[N, M, 1, H, W])  # [N, M, 1, Hf, Wf]
        features = torch.rand(size=[N, 1, D, H, W]) * 100  # [N, 1, D, Hf, Wf]

        gap_result = gap_head(features, body_part_masks)
        gmp_result = gmp_head(features, body_part_masks)
        softmax_result = softmax_head(features, body_part_masks)
        gwap_result = gwap_head(features, body_part_masks) # small
        gwap_same_result = gwap_head_same(features, body_part_masks)
        gwap2_result = gwap_head2(features, body_part_masks) # big

        assert(torch.abs(gwap_result - gwap2_result).max() < 0.0001)
        # assert torch.allclose(gwap_result, gwap_same_result)
        # assert torch.allclose(gwap_result, gwap2_result)

    def test_heads1D(self):
        normalization = 'batch_norm_1d'
        depth = 2048
        # gap_head = GlobalAveragePoolingHead(normalization, depth)
        # gmp_head = GlobalMaxPoolingHead(normalization, depth)
        # softmax_head = SoftmaxAveragePoolingHead(normalization, depth)
        gwap_head = GlobalWeightedAveragePoolingHead1D(normalization, depth)
        gwap_head_same = GlobalWeightedAveragePoolingHead1D(normalization, depth)
        gwap_head2 = GlobalWeightedAveragePoolingHead21D(normalization, depth)

        N, M, D, H, W = 32, 36, 2048, 16, 8

        body_part_masks = torch.rand(size=[N, M, 1, H, W])  # [N, M, 1, Hf, Wf]
        features = torch.rand(size=[N, 1, D, H, W]) * 100  # [N, 1, D, Hf, Wf]

        # gap_result = gap_head(features, body_part_masks)
        # gmp_result = gmp_head(features, body_part_masks)
        # softmax_result = softmax_head(features, body_part_masks)
        gwap_result = gwap_head(features, body_part_masks) # small
        gwap_same_result = gwap_head_same(features, body_part_masks)
        gwap2_result = gwap_head2(features, body_part_masks) # big

        assert (torch.abs(gwap_result - gwap2_result).max() < 0.0001)
        # assert torch.allclose(gwap_result, gwap_same_result)
        # assert torch.allclose(gwap_result, gwap2_result)

    def test_heads3D(self):
        normalization = 'batch_norm_1d'
        depth = 2048
        # gap_head = GlobalAveragePoolingHead(normalization, depth)
        # gmp_head = GlobalMaxPoolingHead(normalization, depth)
        # softmax_head = SoftmaxAveragePoolingHead(normalization, depth)
        gwap_head = GlobalWeightedAveragePoolingHead('batch_norm_2d', depth)
        gwap_head3d = GlobalWeightedAveragePoolingHead3D('batch_norm_3d', depth)

        N, M, D, H, W = 32, 36, 2048, 16, 8

        body_part_masks = torch.rand(size=[N, M, 1, H, W])  # [N, M, 1, Hf, Wf]
        features = torch.rand(size=[N, 1, D, H, W]) * 100  # [N, 1, D, Hf, Wf]

        # gap_result = gap_head(features, body_part_masks)
        # gmp_result = gmp_head(features, body_part_masks)
        # softmax_result = softmax_head(features, body_part_masks)
        gwap_result = gwap_head(features, body_part_masks) # small
        gwap2_result = gwap_head3d(features, body_part_masks) # big

        # assert (torch.abs(gwap_result - gwap2_result).max() < 0.0001)
        assert torch.allclose(gwap_result, gwap2_result)
