import torch
import numpy as np


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat


def compute_dist_matrix_old(inputs):
    n = inputs.size(1)
    m = inputs.size(0)
    print(inputs)
    dist = inputs.pow(2)
    print(dist)
    dist = dist.sum(dim=2, keepdim=True)
    print(dist)
    dist = dist.expand(m, n, n)
    print(dist)
    dist_transpose = dist.transpose(1, 2)
    print(dist_transpose)
    dist = dist + dist_transpose
    print(dist)
    inputs_transpose = inputs.transpose(1, 2)
    print('inputs and inputs transpose')
    print(inputs)
    print(inputs_transpose)
    matmul = torch.bmm(inputs, inputs_transpose)
    print(matmul)
    dist = dist - 2 * matmul
    # dist.addmm_(1, -2, inputs, inputs.transpose(0, 2))
    print(dist)
    dist = dist.max(0)[0]
    # dist = dist.mean(0)
    return dist

def compute_dist_matrix(inputs):
    # inputs.shape = [N, M, D]
    dist = compute_body_parts_dist_matrices(inputs)  # dist.shape = [M, N, N]
    dist = combine_body_parts_dist_matrices(dist)  # dist.shape = [N, N]
    return dist

def combine_body_parts_dist_matrices(dist):
    dist = dist.max(0)[0] # .squeeze()
    return dist

def compute_body_parts_dist_matrices(inputs):
    n = inputs.shape[1]
    m = inputs.shape[0]
    dist = inputs.pow(2).sum(dim=2, keepdim=True).expand(m, n, n)
    dist = dist + dist.transpose(1, 2)
    dist = dist - 2 * torch.bmm(inputs, inputs.transpose(1, 2))
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist

dim = 0

head = torch.from_numpy(np.array([[1, 1, 1], [1, 1, 8], [2, 2, 3], [10, 10, 10]]))
torso = torch.from_numpy(np.array([[101, 101, 101], [101, 101, 108], [102, 102, 103], [110, 110, 110]]))
legs = torch.from_numpy(np.array([[3, 3, 1], [3, 7, 1], [6, 1, 3], [12, 15, 6]]))
full = torch.cat([head.unsqueeze(dim), torso.unsqueeze(dim), legs.unsqueeze(dim)], dim)

# head = torch.from_numpy(np.array([[1, 1, 1], [1, 1, 8], [10, 10, 10]]))
# torso = torch.from_numpy(np.array([[101, 101, 101], [101, 101, 108], [110, 110, 110]]))
# full = torch.cat([head.unsqueeze(dim), torso.unsqueeze(dim)], dim)
dis_old = compute_dist_matrix_old(full)

test1 = euclidean_squared_distance(head, head)
print("TEST1")
print(test1)

test2 = euclidean_squared_distance(torso, torso)
print("TEST2")
print(test2)

test3 = euclidean_squared_distance(legs, legs)
print("TEST3")
print(test3)

print('FUUUUUULLL')
print(dis_old)

dist_final = compute_dist_matrix(full)
print(dist_final)

print(torch.all(torch.eq(dis_old, dist_final)))