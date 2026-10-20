"""
AudioMatchModule: the fusion head of the 3D-VGA model.

Keeps that module's overall shape -- project detector features, reason over a spatial
graph, copy-paste augmentation, a Conv1d scoring stack producing ``cluster_ref`` -- and
swaps its four parallel text cross-attention streams (lang / target / adjectives /
neighbors) for the acoustic fusion in models/fusion/spot_fusion.py.

One deliberate departure: the spatial graph is implemented here with a dense masked kNN
instead of torch_geometric's ``knn_graph`` + ``GCNConv``. The operation is the same
symmetric-normalised propagation over a k-nearest-neighbour graph of proposal centres, but
it drops the pyg_lib dependency, which the README lists as the first troubleshooting item
("model can't build at all") because its wheel is keyed to an exact torch build. At M = 256
the dense form is also faster than scatter-gather on GPU. The text branch is untouched --
models/match_module.py still uses pyg -- so nothing existing changes behaviour.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.audio_config import AUDIO
from models.fusion.geometry import default_distance_bias
from models.fusion.spot_fusion import build_fusion
from models.transformer.attention import MultiHeadAttention


class SpatialGraphNet(nn.Module):
    """Dense kNN graph over proposal centres + symmetric-normalised GCN propagation.

    Equivalent to the ``knn_graph`` + six ``GCNConv`` stack in models/match_module.py,
    without torch_geometric. Invalid (non-object) proposals are excluded from every
    neighbourhood, so a real object never aggregates from an empty slot -- the same
    guarantee the masked-then-scattered pyg path gives.
    """

    def __init__(self, hidden_size, num_layers=6, k=5, dropout=0.1):
        super().__init__()
        self.k = k
        self.layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def build_adjacency(self, centers, mask):
        """(B, M, M) row-normalised adjacency of the masked kNN graph, self-loops included."""
        batch, num_proposals = centers.shape[:2]
        distance = torch.cdist(centers, centers)                           # (B, M, M)

        # Never select an invalid neighbour, and never select self as a *neighbour* (the
        # self-loop is added explicitly below so it is not one of the k slots).
        invalid = ~mask.unsqueeze(1).expand(-1, num_proposals, -1)
        eye = torch.eye(num_proposals, device=centers.device, dtype=torch.bool)
        distance = distance.masked_fill(invalid | eye.unsqueeze(0), float("inf"))

        # k is bounded by the emptiest scene in the batch, matching match_module.py's
        # `k_neighbors = num_ngh if obj_lens.min() >= num_ngh else obj_lens.min()`.
        k = int(min(self.k, max(1, int(mask.sum(dim=1).min().item()))))
        _, index = distance.topk(k, dim=-1, largest=False)                 # (B, M, k)

        adjacency = torch.zeros(batch, num_proposals, num_proposals, device=centers.device)
        adjacency.scatter_(2, index, 1.0)
        # Drop edges pointing at padding (topk still returns indices when a row is all-inf).
        adjacency = adjacency * mask.unsqueeze(1).float()
        adjacency = adjacency + torch.diag_embed(mask.float())             # self-loops
        # Undirected, then row-normalised. Row normalisation rather than the symmetric
        # D^-1/2 A D^-1/2 form: proposals here have wildly different degrees and row
        # normalisation keeps every node's message at unit scale.
        adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
        adjacency = adjacency * mask.unsqueeze(2).float()
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, features, centers, mask):
        adjacency = self.build_adjacency(centers, mask)
        out = features
        for layer, norm in zip(self.layers, self.norms):
            message = layer(torch.bmm(adjacency, out))
            out = norm(out + self.dropout(F.relu(message)))                # residual: 6 plain
        return out * mask.unsqueeze(-1).float()                            # GCN layers wash out


def copy_paste_proposals(features, objectness_mask):
    """models/match_module.py's copy-paste augmentation, factored out of the forward pass.

    Overwrites empty proposal slots with real object features from elsewhere in the batch,
    so the scoring head sees a full set of plausible distractors and cannot learn "few
    non-empty slots => the answer is among them". Behaviour matches the original loop; it
    is written with tensor ops here because it ran per batch item there.
    """
    batch_size, num_proposals, hidden = features.shape
    flat_mask = objectness_mask.reshape(-1)
    donors = features.reshape(-1, hidden)[flat_mask]
    if donors.shape[0] == 0:
        return features

    donors = donors.repeat(2, 1)
    out = features.clone()
    obj_lens = objectness_mask.sum(dim=1)
    total = int(flat_mask.sum().item())

    cursor = 0
    for i in range(batch_size):
        empty = torch.where(~objectness_mask[i])[0]
        cursor += int(obj_lens[i].item())
        available = total - int(obj_lens[i].item())
        take = min(int(empty.shape[0]), max(0, available))
        if take > 0:
            out[i, empty[:take], :] = donors[cursor:cursor + take, :]
    return out


class AudioMatchModule(nn.Module):
    """Detector proposals + acoustic slots -> per-proposal grounding scores."""

    def __init__(self, args, num_proposals=256, hidden_size=None, det_channel=288,
                 head=None, depth=None):
        super().__init__()
        self.args = args
        self.num_proposals = num_proposals
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        self.num_heads = head or AUDIO.NUM_HEADS
        self.depth = (depth or AUDIO.FUSION_DEPTH) - 1

        self.features_concat = nn.Sequential(
            nn.Conv1d(det_channel, self.hidden_size, 1),
            nn.BatchNorm1d(self.hidden_size),
            nn.PReLU(self.hidden_size),
            nn.Conv1d(self.hidden_size, self.hidden_size, 1),
        )

        head_dim = self.hidden_size // self.num_heads
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=self.hidden_size, d_k=head_dim, d_v=head_dim,
                               h=self.num_heads)
            for _ in range(max(1, self.depth)))

        # AUDIO.GRAPH_K = 0 removes the spatial graph entirely (ablation).
        self.graph = (SpatialGraphNet(self.hidden_size, num_layers=AUDIO.GRAPH_LAYERS,
                                      k=AUDIO.GRAPH_K)
                      if AUDIO.GRAPH_K > 0 else None)

        self.fusion = build_fusion(hidden_size=self.hidden_size, num_heads=self.num_heads,
                                   depth=AUDIO.FUSION_DEPTH)

        # Same scoring stack as models/match_module.py, over [fused ; graph] -- the two
        # streams that module concatenates before scoring.
        self.match = nn.Sequential(
            nn.Conv1d(2 * self.hidden_size, 2 * self.hidden_size, 1),
            nn.BatchNorm1d(2 * self.hidden_size),
            nn.PReLU(),
            nn.Conv1d(2 * self.hidden_size, 2 * self.hidden_size, 1),
            nn.BatchNorm1d(2 * self.hidden_size),
            nn.PReLU(),
            nn.Conv1d(2 * self.hidden_size, self.hidden_size, 1),
            nn.BatchNorm1d(self.hidden_size),
            nn.PReLU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, 1),
            nn.BatchNorm1d(self.hidden_size),
            nn.PReLU(),
            nn.Conv1d(self.hidden_size, 1, 1),
        )

        # Mixes the transport plan's target row into the score. Learned, initialised at the
        # configured weight, so the model can walk it back if the MLP head is enough.
        self.ot_weight = nn.Parameter(torch.tensor(float(AUDIO.OT_LOGIT_WEIGHT)))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _objectness_mask(data_dict):
        scores = data_dict["objectness_scores"]
        if scores.shape[2] == 2:
            mask = scores.max(2)[1].bool()                                 # (B, M)
        else:
            mask = (scores > 0).squeeze(-1).bool()

        # Early in training the detector can call every proposal in a scene empty. An
        # all-False row makes softmax over an all-masked axis produce NaN -- in the slot
        # cross-attention, in Sinkhorn, and then in every gradient after them -- so keep
        # one proposal alive. It scores like any other; it just cannot poison the batch.
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    def _detector_features(self, data_dict):
        if self.args.detector == "VN":
            return data_dict["detr_features"].permute(0, 2, 1)
        return data_dict["features"]

    # ------------------------------------------------------------------ forward

    def forward(self, data_dict):
        features = self.features_concat(self._detector_features(data_dict))
        features = features.permute(0, 2, 1)                               # (B, M, H)
        batch_size, num_proposals = features.shape[:2]

        centers = data_dict["center"]                                      # (B, M, 3)
        obj_mask = self._objectness_mask(data_dict)                        # (B, M)
        len_num_max = data_dict["audio_feat_list"].shape[1]

        # Scene-level proposal self-attention under the fixed geometric prior. This stage
        # is audio-independent -- it is the detector's own spatial context -- so it stays
        # as models/match_module.py has it and runs once per scene, not per annotation.
        geo_bias = default_distance_bias(centers, self.num_heads)
        for layer in self.self_attn:
            features = layer(features, features, features,
                             attention_weights=geo_bias, way="add")

        # With GRAPH_K = 0 the proposals go into the fusion head without relational
        # message passing; the scoring head still concatenates both tensors, so the
        # arm removes the graph and keeps every shape.
        graph_features = (self.graph(features, centers, obj_mask)
                          if self.graph is not None else features)

        is_train = bool(data_dict["istrain"][0] == 1)
        roll = random.random()
        data_dict["random"] = roll                                         # read by the ref loss
        if is_train and roll < 0.5 and not AUDIO.DISABLE_COPY_PASTE:
            features = copy_paste_proposals(features, obj_mask)
            graph_features = copy_paste_proposals(graph_features, obj_mask)

        # Broadcast the scene to its L annotations, as the text branch does.
        def expand(tensor):
            return (tensor.unsqueeze(1).expand(-1, len_num_max, -1, -1)
                    .reshape(batch_size * len_num_max, num_proposals, -1))

        prop_flat = expand(features)
        graph_flat = expand(graph_features)
        mask_flat = (obj_mask.unsqueeze(1).expand(-1, len_num_max, -1)
                     .reshape(batch_size * len_num_max, num_proposals))

        fused, plan = self.fusion(
            proposals=graph_flat,
            slots=data_dict["audio_slots"],
            slot_mass=data_dict["audio_slot_mass"],
            proposal_mask=mask_flat,
            centers=centers,
            len_num_max=len_num_max,
            frames=data_dict.get("audio_fea"),
            frame_mask=data_dict.get("audio_attention_mask"),
            relation=data_dict.get("audio_relation"),
        )

        aggregated = torch.cat([fused, prop_flat], dim=-1).permute(0, 2, 1).contiguous()
        confidence = self.match(aggregated).squeeze(1)                     # (B*L, M)

        if plan is not None:
            # The referent's transport row is already a grounding score, and one that
            # carries the competition constraint the MLP head has no way to express. Log
            # turns the plan's mass back into a logit on the same scale as the head.
            #
            # Which row that is comes from the role election, not a slot index: the
            # referent weights (B*L, K) contract the plan's K rows into one. Hardcoding a
            # row would assert something about an unsupervised partition that no signal in
            # this model could make true.
            referent_weights = data_dict["audio_role_attn"][:, AUDIO.ROLE_REFERENT]
            target_row = torch.einsum("bk,bkm->bm", referent_weights, plan)  # (B*L, M)
            target_row = target_row / target_row.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            confidence = confidence + self.ot_weight * torch.log(target_row.clamp_min(1e-8))
            data_dict["ot_plan"] = plan
            data_dict["ot_target_logit"] = torch.log(target_row.clamp_min(1e-8))

        # Invalid proposals can never be the answer; -1e4 rather than -inf keeps the
        # SoftmaxRankingLoss finite when a scene has very few objects.
        confidence = confidence.masked_fill(~mask_flat, -1e4)

        data_dict["cluster_ref"] = confidence
        data_dict["fused_proposals"] = fused
        return data_dict
