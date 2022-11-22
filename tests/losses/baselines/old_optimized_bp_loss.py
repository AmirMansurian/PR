from __future__ import division, absolute_import

import torch
import torch.nn as nn
from torchreid.utils import Writer
import torch.nn.functional as F


class OldOptimizedBodyPartsLoss(nn.Module):
    """An abstract class representing a Triplet Loss using body parts embeddings.
    """

    def __init__(self, margin=0.3, epsilon=1e-16):
        super(OldOptimizedBodyPartsLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.debug_ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.writer = Writer.current_writer()
        self.batch_debug = False
        self.imgs = None
        self.masks = None
        self.epsilon = epsilon

    def forward(self, body_parts_embeddings, labels):
        """
        Args:
            body_parts_embeddings (torch.Tensor): feature matrix with shape (batch_size, parts_num, feat_dim).
            labels (torch.LongTensor): ground truth labels with shape (num_classes).
        """

        # Compute pairwise distance matrix for each body part
        per_body_part_pairwise_dist = self._body_parts_pairwise_distance_matrix(body_parts_embeddings, False, self.epsilon)

        pairwise_dist = self._combine_body_parts_dist_matrices(per_body_part_pairwise_dist)

        return self._hard_mine_triplet_loss(pairwise_dist, labels, self.margin)

    @staticmethod
    def _body_parts_pairwise_distance_matrix(embeddings, squared, epsilon):
        """
        embeddings.shape = (M, N, C)
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        embeddings = embeddings.transpose(1, 0)
        dot_product = torch.matmul(embeddings, embeddings.transpose(2, 1))
        square_sum = dot_product.diagonal(dim1=1, dim2=2)
        distances = square_sum.unsqueeze(2) - 2 * dot_product + square_sum.unsqueeze(1)
        distances = F.relu(distances)

        if not squared:
            mask = torch.eq(distances, 0).float()
            distances = distances + mask * epsilon  # for numerical stability (infinite derivative of sqrt in 0)
            distances = torch.sqrt(distances)
            distances = distances * (1 - mask)

        return distances

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist):
        raise NotImplementedError

    def _hard_mine_triplet_loss(self, batch_pairwise_dist, labels, margin):
        """
        compute distance matrix; i.e. for each anchor a_i with i=range(0, batch_size) :
        - find the (a_i,p_i) pair with greatest distance s.t. a_i and p_i have the same label
        - find the (a_i,n_i) pair with smallest distance s.t. a_i and n_i have different label
        - compute triplet loss for each triplet (a_i, p_i, n_i), average them
        Source :
        - https://github.com/lyakaap/NetVLAD-pytorch/blob/master/hard_triplet_loss.py
        - https://github.com/Yuol96/pytorch-triplet-loss/blob/master/model/triplet_loss.py
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """

        # Get the hardest positive pairs
        mask_anchor_positive = self._get_anchor_positive_mask(labels).unsqueeze(0)
        valid_positive_dist = batch_pairwise_dist * mask_anchor_positive.float()
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=-1, keepdim=True)

        # Get the hardest negative pairs
        mask_anchor_negative = self._get_anchor_negative_mask(labels).unsqueeze(0)
        max_anchor_negative_dist, _ = torch.max(batch_pairwise_dist, dim=-1, keepdim=True)
        anchor_negative_dist = batch_pairwise_dist + max_anchor_negative_dist * (~mask_anchor_negative).float()
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=-1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss

    @staticmethod
    def _get_anchor_positive_mask(labels):
        """
        To be a valid positive pair (a,p) :
            - a and p are different embeddings
            - a and p have the same label
        """
        indices_equal_mask = torch.eye(labels.shape[0], device=(labels.get_device() if labels.is_cuda else None))
        indices_not_equal_mask = indices_equal_mask.byte() ^ 1

        # Check if labels[i] == labels[j]
        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

        mask_anchor_positive = indices_not_equal_mask * labels_equal_mask

        return mask_anchor_positive

    @staticmethod
    def _get_anchor_negative_mask(labels):
        """
        To be a valid negative pair (a,n) :
            - a and n have the different label (and therefore are different embeddings)
        """

        # Check if labels[i] != labels[k]
        labels_not_equal_mask = torch.ne(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

        return labels_not_equal_mask

    def compute_test_pairwise_dist_matrix(self, qf, gf):
        per_body_part_pairwise_dist = self._compute_body_parts_test_dist_matrices(qf, gf)
        return self._combine_body_parts_test_dist_matrices(per_body_part_pairwise_dist)

    def _combine_body_parts_test_dist_matrices(self, per_body_part_pairwise_dist):
        return per_body_part_pairwise_dist.mean(dim=0)

    @staticmethod
    def _compute_body_parts_test_dist_matrices(qf, gf):
        """
        gf, qf shapes = (N, M, C)
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        qf = qf.transpose(1, 0)
        gf = gf.transpose(1, 0)
        dot_product = torch.matmul(qf, gf.transpose(2, 1))
        qf_square_sum = qf.pow(2).sum(dim=-1)
        gf_square_sum = gf.pow(2).sum(dim=-1)

        distances = qf_square_sum.unsqueeze(2) - 2 * dot_product + gf_square_sum.unsqueeze(1)
        distances = F.relu(distances)
        distances = torch.sqrt(distances)

        return distances
