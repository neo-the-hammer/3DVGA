import torch
import torch.nn as nn
import numpy as np


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1)
    dist1, idx1 = torch.min(pc_dist, dim=2)
    dist2, idx2 = torch.min(pc_dist, dim=1)
    return dist1, idx1, dist2, idx2


def knn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False, k=1):
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)
    if N < k:
        k = N
    dist, idx = pc_dist.topk(k, dim=1, largest=False)

    return dist, idx

def demo_nn_distance():
    np.random.seed(0)
    pc1arr = np.random.random((1,5,3))
    pc2arr = np.random.random((1,6,3))
    pc1 = torch.from_numpy(pc1arr.astype(np.float32))
    pc2 = torch.from_numpy(pc2arr.astype(np.float32))
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            dist[i,j] = np.sum((pc1arr[0,i,:] - pc2arr[0,j,:]) ** 2)
    print(dist)
    print('-'*30)
    print('L1smooth dists:')
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2, True)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            error = np.abs(pc1arr[0,i,:] - pc2arr[0,j,:])
            quad = np.minimum(error, 1.0)
            linear = error - quad
            loss = 0.5*quad**2 + 1.0*linear
            dist[i,j] = np.sum(loss)
    print(dist)


if __name__ == '__main__':
    demo_nn_distance()
