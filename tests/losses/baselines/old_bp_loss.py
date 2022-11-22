import torch
import torch.nn.functional as F


class OldBodyPartsLoss(torch.nn.Module):
    """An abstract class representing a Triplet Loss using body parts embeddings.
    """

    def __init__(self, margin=0.3, epsilon=1e-16):
        super(OldBodyPartsLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.epsilon = epsilon

    def forward(self, inputs, labels):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (body parts list size M, batch size N, feat_dim).
            labels (torch.LongTensor): ground truth labels with shape (num_classes).
        """

        # Compute pids distance matrix
        per_body_part_pairwise_dist = self._compute_body_parts_dist_matrices(inputs, self.epsilon)  # shape = [M, N, N]
        pairwise_dist = self._combine_body_parts_dist_matrices(per_body_part_pairwise_dist)  # shape = [N, N]

        # For each anchor, find the hardest positive and negative
        return self._hard_mine_triplet_loss(pairwise_dist, labels, self.ranking_loss)

    def _combine_body_parts_dist_matrices(self, per_body_part_pairwise_dist):
        raise NotImplementedError

    def compute_test_dist_matrix(self, gf, qf):
        body_parts_dist_matrices = self._compute_body_parts_test_dist_matrices(gf, qf)
        return body_parts_dist_matrices.mean(dim=0)

    def _compute_test_pairwise_dist_matrix(self, qf, gf):
        body_parts_dist_matrices = self._compute_body_parts_test_dist_matrices(qf, gf)
        return body_parts_dist_matrices.mean(dim=0)

    # FIXME Doesn't work, numerical instability,
    # def _compute_body_parts_dist_matrices(inputs, epsilon):
        ##inputs.shape = [N, M, D]
        # inputs = inputs.transpose(1, 0)
        # n = inputs.shape[1]
        # m = inputs.shape[0]
        # dist = inputs.pow(2).sum(dim=2, keepdim=True).expand(m, n, n)
        # dist = dist + dist.transpose(1, 2)
        # dist = dist - 2 * torch.bmm(inputs, inputs.transpose(1, 2))
        # dist = (~torch.lt(dist, 1e-8)).float * dist
        # mask = torch.eq(dist, 0.).float()
        # dist = dist + mask * epsilon  # for numerical stability (infinite derivative of sqrt in 0)
        # dist = torch.sqrt(dist)
        # dist = dist * (1 - mask)
        #
        # return dist


    def _compute_body_parts_dist_matrices(self, embeddings, epsilon):
        """
        embeddings.shape = (M, N, C)
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        embeddings = embeddings.transpose(1, 0)
        dot_product = torch.matmul(embeddings, embeddings.transpose(2, 1))
        square_sum = dot_product.diagonal(dim1=1, dim2=2)
        distances = square_sum.unsqueeze(2) - 2 * dot_product + square_sum.unsqueeze(1)
        distances = F.relu(distances)

        mask = torch.eq(distances, 0).float()
        distances = distances + mask * epsilon  # for numerical stability (infinite derivative of sqrt in 0)
        distances = torch.sqrt(distances)
        distances = distances * (1 - mask)

        return distances

    def _hard_mine_triplet_loss(self, dist, targets, ranking_loss):
        n = dist.shape[0]
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        return ranking_loss(dist_an, dist_ap, y)


    def _compute_body_parts_test_dist_matrices(self, qf, gf):
        # gf.shape, qf.shape = [N, M, D]
        qf_body_part_features_list = qf.permute(1, 0, 2).split(1, dim=0)  # list of Tensor of size [1, N, D]
        gf_body_part_features_list = gf.permute(1, 0, 2).split(1, dim=0)  # list of Tensor of size [1, N, D]
        dist_matrices = []
        for i in range(0, len(gf_body_part_features_list)):
            distmat = self._euclidean_squared_distance(qf_body_part_features_list[i].squeeze(),
                                                  gf_body_part_features_list[i].squeeze())
            dist_matrices.append(distmat)
        body_parts_dist_matrices = torch.stack(dist_matrices, dim=0)
        return body_parts_dist_matrices


    def _euclidean_squared_distance(self, input1, input2):
        # dist(a, b) = sum((a_i - b_i)^2) = sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i)
        m, n = input1.size(0), input2.size(0)
        mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) # sum(a_i^2)
        mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t() # sum(b_i^2)
        distmat = mat1 + mat2 # sum(a_i^2) + sum(b_i^2)
        distmat.addmm_(input1, input2.t(), beta=1, alpha=-2) # sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i)
        distmat = torch.sqrt(distmat)
        return distmat
