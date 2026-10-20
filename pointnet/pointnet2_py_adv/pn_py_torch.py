# Fast pure-PyTorch PointNet++ ops -- optimized drop-in for pointnet2_python.
#
# Same API, same outputs, several times faster. No CUDA extension, no new
# dependencies: every op is plain torch and runs on CPU or GPU identically.
#
# Why the original is slow (and what this file does about it)
# -----------------------------------------------------------
# 1. ball_query sorted ALL N=40,000 candidate columns per centroid just to keep
#    the first `nsample` in-radius indices, after materializing an
#    arange(N).repeat(B, npoint, 1) int64 tensor (~655 MB at SA1 scale).
#    Here: torch.topk(k=nsample, largest=False) over a broadcast int32 index
#    grid -- the k smallest indices are EXACTLY what sort()[:nsample] returned,
#    at O(N log k) instead of O(N log N) and ~1/4 the memory. The centroid axis
#    is chunked so the distance matrix never exceeds ~16M elements (~64 MB)
#    regardless of scene size. Index math runs under torch.no_grad().
# 2. three_nn full-sorted the distance matrix for the 3 nearest neighbours.
#    Here: topk(3).
# 3. Grouping gathered in (B, N, C) layout, which forced a full
#    transpose+contiguous copy of the feature tensor per SA layer, then a
#    permute to (B, C, npoint, nsample) that the following conv had to
#    re-materialize. Here: torch.gather directly on the (B, C, N) tensor along
#    dim 2 with expanded (view-only) indices -- output is born contiguous in
#    exactly the layout the shared MLP consumes. Zero large copies.
# 4. furthest_point_sample keeps the original sequential algorithm (FPS is
#    inherently sequential and must stay bit-identical), but the loop body uses
#    gather + torch.minimum(out=) instead of fancy indexing + boolean-mask
#    assignment: fewer allocations per iteration, same arithmetic, same RNG
#    consumption -> identical output for an identical torch seed.
#
# Equivalence guarantees (verified against pointnet2_python on random data)
# -------------------------------------------------------------------------
# * ball_query, gather_operation, furthest_point_sample (same seed):
#   torch.equal -- bitwise identical.
# * three_nn / three_interpolate / module forwards: allclose; the only
#   intentional non-guarantee is neighbour ordering when two points are at
#   bitwise-equal distance (the original torch.sort is unstable there too).
# * PointnetSAModuleVotes / PointnetFPModule reuse pointnet2_python.pt_utils
#   SharedMLP, so state-dict keys match the existing checkpoints exactly.
#
# Wiring: models/backbone_module.py and models/proposal_module.py resolve their
# implementation through a three-tier chain, each tier tried only when the one
# above it cannot be imported:
#
#   1. pointnet2_ops.pointnet2_modules            CUDA extension (fastest)
#   2. pointnet.pointnet2_py_adv.pn_py_torch      this file (optimized pure torch)
#   3. pointnet.pointnet2_python.pointnet2_modules   the original pure-Python fallback
#
# ABLATION.POINTNET_IMPL (experiments/ablation/ablation_config.py) pins a specific
# tier; when it is unset the chain is walked in the order above.

import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# SharedMLP is imported from pointnet2_python so module attribute names -- and
# therefore checkpoint state-dict keys -- are identical by construction.
# pointnet2_python has no __init__.py; mirror pointnet2_modules.py's own import
# mechanism (its directory on sys.path, top-level import).
# Both packages live side by side under pointnet/, so the sibling directory is
# one level up from this file.
_P2P_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pointnet2_python"
)
if _P2P_DIR not in sys.path:
    sys.path.append(_P2P_DIR)
import pt_utils  # noqa: E402  (pointnet2_python/pt_utils.py)

#: ball_query works on the distance matrix in centroid-axis chunks so peak
#: extra memory stays ~O(_CHUNK_ELEMS) however large the scene is.
#: 16M fp32 elements = 64 MB (plus the int32 index grid of the same shape).
_CHUNK_ELEMS = 16_777_216


# --------------------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------------------

def square_distance(src, dst):
    """Squared euclidean distance between every src/dst pair.

    Identical formula and float-op order to pointnet2_python.square_distance
    (matmul identity), so downstream radius comparisons see the same values.

    src: (B, N, C), dst: (B, M, C) -> (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------

def furthest_point_sample(xyz, npoint):
    """Iterative furthest point sampling. (B, N, 3) -> (B, npoint) int64.

    Bit-identical to pointnet2_python.furthest_point_sample for the same torch
    RNG state: one torch.randint call, then the same (xyz - centroid)**2 update
    per step. The loop body avoids the original's boolean-mask fancy assignment
    (two indexing kernels + temporaries) in favour of torch.minimum(out=).
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    with torch.no_grad():
        distance = torch.full((B, N), 1e10, dtype=xyz.dtype, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz.gather(1, farthest.view(B, 1, 1).expand(B, 1, 3))
            dist = torch.sum((xyz - centroid) ** 2, -1)
            torch.minimum(distance, dist, out=distance)
            # torch.max (not argmax): the original resolves ties to the first
            # maximal index this way; keep the exact same tie-breaking.
            farthest = torch.max(distance, -1)[1]
    return centroids


# --------------------------------------------------------------------------------------
# gathering
# --------------------------------------------------------------------------------------

def gather_operation(points, idx):
    """Row-gather without materialized batch-index grids.

    points: (B, N, C); idx: (B, S) or (B, S, K) int64
    -> (B, S, C) or (B, S, K, C), same values as the original advanced-indexing
    version, but the index tensor is expand()ed (a view), never repeat()ed.
    """
    B, N, C = points.shape
    flat = idx.reshape(B, -1)                                   # (B, S*K)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(B, flat.shape[1], C))
    return out.reshape(*idx.shape, C)


def _gather_channels(features, idx):
    """Gather along the point axis of a channel-first tensor.

    features: (B, C, N); idx: (B, S, K) int64 -> (B, C, S, K), contiguous.
    This is the layout the shared MLPs consume, so no transpose/permute of the
    big feature tensor ever happens.
    """
    B, C, N = features.shape
    _, S, K = idx.shape
    flat = idx.reshape(B, 1, S * K).expand(B, C, S * K)         # view, no copy
    return torch.gather(features, 2, flat).reshape(B, C, S, K)


# --------------------------------------------------------------------------------------
# neighbourhood queries
# --------------------------------------------------------------------------------------

def ball_query(radius, nsample, xyz, new_xyz):
    """Indices of up to `nsample` in-radius points per centroid, smallest-index
    first, padded with the first hit -- exactly the original's semantics.

    xyz: (B, N, 3); new_xyz: (B, S, 3) -> (B, S, nsample) int64
    """
    with torch.no_grad():
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        device = xyz.device
        r2 = radius ** 2

        # int32 grid: the topk input; broadcast row (1, 1, N), expanded per chunk.
        arange = torch.arange(N, dtype=torch.int32, device=device).view(1, 1, N)
        sentinel = torch.tensor(N, dtype=torch.int32, device=device)

        chunk = max(1, min(S, _CHUNK_ELEMS // max(B * N, 1)))
        out = torch.empty(B, S, nsample, dtype=torch.int64, device=device)

        for s0 in range(0, S, chunk):
            s1 = min(s0 + chunk, S)
            sqrdists = square_distance(new_xyz[:, s0:s1], xyz)  # (B, s, N)
            # keep the index where in radius, else the sentinel N ...
            cand = torch.where(sqrdists <= r2, arange, sentinel)
            # ... so the k smallest values ARE the first k in-radius indices:
            # identical to the original sort()[:, :, :nsample], at O(N log k).
            idx = torch.topk(cand, nsample, dim=-1, largest=False, sorted=True)[0]
            # pad empty tail slots with the first (guaranteed) hit
            first = idx[:, :, 0:1].expand_as(idx)
            idx = torch.where(idx == sentinel, first, idx)
            out[:, s0:s1] = idx.long()
    return out


class QueryAndGroup(nn.Module):
    """Ball-query + channel-first grouping. Same constructor and outputs as the
    pointnet2_python version; the big feature tensor is never transposed."""

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False,
                 normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        """xyz: (B, N, 3); new_xyz: (B, S, 3); features: (B, C, N)
        -> new_features (B, 3 + C, S, nsample) [+ grouped_xyz, + unique_cnt]"""
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, S, K)

        if self.sample_uniformly:
            # Kept verbatim from the original (off in this repo): re-draw
            # duplicate slots uniformly from the unique in-radius points.
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]), device=xyz.device)
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique,
                                               (self.nsample - num_unique,),
                                               dtype=torch.long, device=xyz.device)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        # (B, 3, N) is a tiny copy (N x 3 floats); it buys contiguous
        # channel-first gathers for both xyz and features.
        xyz_t = xyz.transpose(1, 2).contiguous()
        grouped_xyz = _gather_channels(xyz_t, idx)                  # (B, 3, S, K)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        if features is not None:
            grouped_features = _gather_channels(features, idx)      # (B, C, S, K)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, \
                "It's not possible to have no features and no xyz coordinates to use as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        return ret[0] if len(ret) == 1 else tuple(ret)


class GroupAll(nn.Module):
    """Group every point into one region. Verbatim semantics."""

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        super().__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, features=None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)              # (B, 3, 1, N)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        return new_features


# --------------------------------------------------------------------------------------
# interpolation (feature propagation)
# --------------------------------------------------------------------------------------

def three_nn(unknown, known):
    """3 nearest neighbours of `unknown` in `known` via topk (no full sort).

    unknown: (B, n, 3); known: (B, m, 3) -> dist (B, n, 3), idx (B, n, 3)
    """
    with torch.no_grad():
        sqrdists = square_distance(unknown, known)
        dist, idx = torch.topk(sqrdists, 3, dim=-1, largest=False, sorted=True)
    return dist, idx


def three_interpolate(features, idx, weight):
    """Weighted interpolation of 3-NN features, fused in channel-first layout.

    features: (B, C, M); idx: (B, n, 3); weight: (B, n, 3) -> (B, C, n)
    """
    B, C, M = features.shape
    _, n, _ = idx.shape
    gathered = _gather_channels(features, idx)                      # (B, C, n, 3)
    return torch.sum(gathered * weight.view(B, 1, n, 3), dim=-1)   # (B, C, n)


# --------------------------------------------------------------------------------------
# modules (same signatures / attribute names / returns as pointnet2_python)
# --------------------------------------------------------------------------------------

class PointnetSAModuleVotes(nn.Module):
    """Set-abstraction layer with vote-index passthrough (VoteNet variant)."""

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,
            normalize_xyz: bool = False,
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = QueryAndGroup(
                radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True,
                normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly,
                ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None,
                inds: torch.Tensor = None):
        """xyz (B, N, 3), features (B, C, N), optional inds (B, npoint)
        -> new_xyz (B, npoint, 3), new_features (B, mlp[-1], npoint), inds"""
        if inds is None:
            inds = furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = gather_operation(xyz, inds) if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)

        new_features = self.mlp_module(grouped_features)            # (B, mlp[-1], S, K)

        if self.pooling == 'max':
            new_features = F.max_pool2d(new_features,
                                        kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(new_features,
                                        kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'rbf':
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1, keepdim=False)
                            / (self.sigma ** 2) / 2)                # (B, S, K)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1,
                                     keepdim=True) / float(self.nsample)

        new_features = new_features.squeeze(-1)                     # (B, mlp[-1], S)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        return new_xyz, new_features, inds, unique_cnt


class PointnetFPModule(nn.Module):
    """Feature propagation: 3-NN inverse-distance interpolation + shared MLP."""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,
                unknown_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        """unknown (B, n, 3), known (B, m, 3), unknown_feats (B, C1, n),
        known_feats (B, C2, m) -> (B, mlp[-1], n)"""
        if known is not None:
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2],
                                                    unknown.size(1))

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)
