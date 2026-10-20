# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''

import torch
import torch.nn as nn
import sys
import os
from typing import *

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def square_distance(src, dst):

    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2 = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist   


def furthest_point_sample(xyz, npoint):
    # type: (torch.Tensor, int) -> torch.Tensor
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance

    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of features in the sampled set

    Returns
    -------
    torch.Tensor
        (B, npoint) tensor containing the set
    """

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def gather_operation(points, idx):

    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Parameters
    ----------
    points : torch.Tensor
        (B, N, C) tensor

    idx : torch.Tensor
        (B, npoint) tensor of the points to gather

    Returns
    -------
    torch.Tensor
        (B, npoint, C) tensor
    """

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points   


def three_nn(unknown, known):
    # type: (torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    """
        Find the three nearest neighbors of unknown in known
    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of unknown features
    known : torch.Tensor
        (B, m, 3) tensor of known features

    Returns
    -------
    dist : torch.Tensor
        (B, n, 3) l2 distance to the three nearest neighbors
    idx : torch.Tensor
        (B, n, 3) index of 3 nearest neighbors
    """

    dist = square_distance(unknown, known)
    dist, idx = dist.sort(dim=-1)
    dist, idx = dist[:, :, :3], idx[:, :, :3]  # [B, N, 3]
    return dist,idx


def three_interpolate(features, idx, weight):

    # type(torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
    """
        Performs weight linear interpolation on 3 features
    Parameters
    ----------
    features : torch.Tensor
        (B, C, M) Features descriptors to be interpolated from
    idx : torch.Tensor
        (B, N, 3) three nearest neighbors of the target features in features
    weight : torch.Tensor
        (B, N, 3) weights

    Returns
    -------
    torch.Tensor
        (B, C, N) tensor of the interpolated features
    """

    features = features.permute(0, 2, 1) # (B, M, C)
    B, _, _ = features.shape
    _, N, _ = idx.shape
    indexed_points = gather_operation(features, idx) # (B, N, 3, C)
    interpolated_points = torch.sum(indexed_points * weight.view(B, N, 3, 1), dim=2) # (B, N, C)
    interpolated_points = interpolated_points.permute(0,2,1) # (B, C, N)
    return interpolated_points


def ball_query(radius, nsample, xyz, new_xyz):
    # type: (float, int, torch.Tensor, torch.Tensor) -> torch.Tensor

    """
    Parameters
    ----------
    radius : float
        radius of the balls
    nsample : int
        maximum number of features in the balls
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the ball query

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """

    device = xyz.device
    B, N, _ = xyz.shape
    _, npoint, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, npoint, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, npoint, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool, bool, bool, bool, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        device = xyz.device

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz) # (B, npoint, nsample)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]),device=device)
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long, device=device)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        grouped_xyz = gather_operation(xyz, idx)                                    # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(-2)                                        # (B, npoint, nsample, 3) - (B, npoint, 1, 3) 
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        grouped_xyz = grouped_xyz.permute(0,3,1,2)                                  # (B, 3, npoint, nsample)
        
        if features is not None:

            features = features.transpose(1, 2).contiguous()                        # (B, N, C)
            grouped_features = gather_operation(features, idx)                      # (B, npoint, nsample, C)
            grouped_features = grouped_features.permute(0,3,1,2)                    # (B, C, npoint, nsample)

            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)    # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
            
        else:
            assert (self.use_xyz), "It's not possible to have no features and no xyz coordinates to use as a feature!"
            new_features = grouped_xyz 

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz 

    def forward(self, xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)

        if features is not None:

            grouped_features = features.unsqueeze(2)
            
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
