"""
SPOT -- Slot-Proposal Optimal Transport fusion.

The proposed fusion operator for 3D visual grounding by audio. Full write-up and the
reasoning behind each piece: docs/3DVGA.md.

--------------------------------------------------------------------------------------
What every existing method does, and why it caps out
--------------------------------------------------------------------------------------

AP-Refer (Neurocomputing 611, 2025) concatenates *mean-pooled* audio with seed-point
features, multiplies by a mask and runs an MLP. Mean pooling over ~200 wav2vec frames
throws away the word order that a referring expression is made of, so "the chair left of
the table" and "the table left of the chair" arrive identical. Overall 23.09 / 13.77.

Attentive fusion -- proposals cross-attend over audio frames -- fixes the pooling and is a
clear improvement. But it keeps a deeper flaw: **softmax over the audio axis makes every
proposal an independent bidder**. Each of nine chairs in a room attends to the "chair"
region of the utterance, each gets a high score, and nothing in the operator expresses that
*exactly one* of them is the referent. The ranking that follows is decided by whatever
noise survives, which is exactly the Unique/Multiple gap every 3DVG-Audio number shows.

Grounding is an assignment problem. So use an assignment operator.

--------------------------------------------------------------------------------------
The operator
--------------------------------------------------------------------------------------

Per layer, over K acoustic role slots (from models/audio_module.py) and M proposals:

  (1) Proposals read slots.   Standard cross-attention, cheap because K ~ 6, not ~200
      frames: the slot bank has already summarised the utterance, so this attends over
      roles rather than over syllables.

  (2) Slots read proposals.   The reverse direction, which no prior 3DVG-Audio work has:
      the "anchor" slot should mean something different in a room with one table than in a
      room with six. Slots become scene-aware before they are used to score anything.

  (3) Optimal-transport routing.   Cost from cosine similarity between slots and
      proposals; a few Sinkhorn iterations under marginals (slot masses from the audio
      branch; uniform over valid proposals). The column constraint is the whole point --
      a slot has a fixed mass budget, so proposals *compete*. Nine chairs can no longer all
      win the chair slot.

  (4) Routed read-out.   Each proposal is enriched with the slots that claimed it, weighted
      by the transport plan, through a learned gate.

  (5) Geometric self-attention with the RSK bias (models/fusion/geometry.py), so the
      spatial prior applied between proposals is the one the *relation slot* asked for.

The final score mixes the usual MLP head with the target slot's transport row, which is a
grounding score in its own right and one that already carries the competition constraint.

--------------------------------------------------------------------------------------

`XAttnFusion` and `ConcatFusion` in this file are the two baselines, sharing the scoring
head and the training-time copy-paste augmentation so a fusion ablation changes the fusion
and nothing else.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.audio_config import AUDIO
from models.fusion.geometry import RelationSlotKernel, default_distance_bias
from models.fusion.sinkhorn import masked_log_softmax, plan_to_attention, sinkhorn_log
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward


class SPOTLayer(nn.Module):
    """One bidirectional slot/proposal layer: (1)-(4) above."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        head_dim = hidden_size // num_heads

        self.prop_from_slots = MultiHeadAttention(
            d_model=hidden_size, d_k=head_dim, d_v=head_dim, h=num_heads, dropout=dropout)
        self.slots_from_prop = MultiHeadAttention(
            d_model=hidden_size, d_k=head_dim, d_v=head_dim, h=num_heads, dropout=dropout)

        self.slot_proj = nn.Linear(hidden_size, hidden_size)
        self.prop_proj = nn.Linear(hidden_size, hidden_size)

        # Learned scale on the OT cost, which is a cosine similarity -- so this alone fixes
        # the cost range, and with it Sinkhorn's convergence. Bounded to [0.25, 2]: a
        # contrastive-style scale of ~14 would read as a cost range of +-14, where the
        # transport plan's marginals are still ~50% off after any affordable number of
        # iterations and "the slot has a mass budget" stops describing what the plan does.
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Gate on the routed read-out: the model decides per channel how much of the slot
        # message to accept, rather than being forced to take it.
        self.route_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.route_out = nn.Linear(hidden_size * 2, hidden_size)
        self.route_norm = nn.LayerNorm(hidden_size)
        self.ffn = PositionWiseFeedForward(d_model=hidden_size, d_ff=hidden_size * 2,
                                           dropout=dropout)

    def forward(self, proposals, slots, slot_mass, proposal_mask):
        """
        Args:
            proposals:     (B', M, H)
            slots:         (B', K, H)
            slot_mass:     (B', K) how much of the utterance each slot claimed
            proposal_mask: (B', M) bool, True on real objects

        Returns:
            proposals (B', M, H), slots (B', K, H), plan (B', K, M)
        """
        num_slots = slots.shape[1]

        # (1) proposals read slots. No mask: every slot is always available, and a slot
        #     that was not spoken is already near-empty via its mass.
        proposals = self.prop_from_slots(proposals, slots, slots)

        # (2) slots read proposals, masked to real objects. AUDIO.BIDIRECTIONAL_SLOTS
        #     removes this direction for the ablation; the module is still constructed, so
        #     the two arms differ in the computation rather than in parameter count.
        if AUDIO.BIDIRECTIONAL_SLOTS:
            key_mask = (~proposal_mask).unsqueeze(1).unsqueeze(1)             # (B',1,1,M)
            slots = self.slots_from_prop(slots, proposals, proposals, key_mask)

        # (3) optimal-transport routing.
        slot_emb = F.normalize(self.slot_proj(slots), dim=-1)                 # (B', K, H)
        prop_emb = F.normalize(self.prop_proj(proposals), dim=-1)             # (B', M, H)
        cost = torch.bmm(slot_emb, prop_emb.transpose(1, 2))                  # (B', K, M)
        cost = cost * self.logit_scale.clamp(0.25, 2.0)

        pair_mask = proposal_mask.unsqueeze(1).expand(-1, num_slots, -1)

        # Row marginal is the slot budget, renormalised: a slot that claimed little audio
        # also gets little say over proposals, which is how "no relation was spoken"
        # propagates all the way to the score.
        row_marginal = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if AUDIO.OT_ROUTING:
            plan = sinkhorn_log(cost, pair_mask, row_marginal=row_marginal,
                                eps=AUDIO.OT_EPS, n_iters=AUDIO.OT_ITERS)     # (B', K, M)
        else:
            # The ablation for the central claim. Same cost, same temperature, same row
            # masses, same read-out below -- but each row is normalised independently, so
            # nothing couples the proposals and they no longer compete for a slot's budget.
            # The result is still a non-negative matrix summing to 1 overall, so every
            # downstream consumer (plan_to_attention, the grounding logit, plan_entropy)
            # sees the same object and only the competition is gone.
            log_attn = masked_log_softmax(cost / AUDIO.OT_EPS, pair_mask, dim=-1)
            plan = log_attn.exp() * row_marginal.unsqueeze(-1)                # (B', K, M)

        # (4) routed read-out: per proposal, the mixture of slots that claimed it.
        routed = torch.bmm(plan_to_attention(plan, dim=1).transpose(1, 2), slots)
        pair = torch.cat([proposals, routed], dim=-1)
        message = self.route_out(pair)
        if AUDIO.ROUTE_GATE:
            message = self.route_gate(pair) * message
        proposals = self.route_norm(proposals + message)
        proposals = self.ffn(proposals)

        return proposals, slots, plan


class SPOTFusion(nn.Module):
    """The proposed fusion head: stacked SPOT layers + RSK geometric self-attention."""

    def __init__(self, hidden_size=None, num_heads=None, depth=None, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        self.num_heads = num_heads or AUDIO.NUM_HEADS
        self.depth = depth or AUDIO.FUSION_DEPTH
        head_dim = self.hidden_size // self.num_heads

        self.layers = nn.ModuleList(
            SPOTLayer(self.hidden_size, self.num_heads, dropout) for _ in range(self.depth))

        # Geometric self-attention between proposals, biased by RSK.
        self.geo_attn = nn.ModuleList(
            MultiHeadAttention(d_model=self.hidden_size, d_k=head_dim, d_v=head_dim,
                               h=self.num_heads, dropout=dropout)
            for _ in range(self.depth))

        self.rsk = (RelationSlotKernel(self.hidden_size, self.num_heads)
                     if AUDIO.USE_RSK else None)

        # A residual pass over raw frames, kept for the fine phonetic detail the K slots
        # necessarily compress away (a mumbled modifier, a number). Cheap and it only has
        # to add on top of the slot path, so it is a single layer rather than a stack.
        self.frame_attn = MultiHeadAttention(
            d_model=self.hidden_size, d_k=head_dim, d_v=head_dim, h=self.num_heads,
            dropout=dropout)

    def forward(self, proposals, slots, slot_mass, proposal_mask, centers, len_num_max,
                frames=None, frame_mask=None, relation=None):
        """
        Args:
            proposals:     (B*L, M, H)
            slots:         (B*L, K, H)
            slot_mass:     (B*L, K)
            proposal_mask: (B*L, M) bool
            centers:       (B, M, 3) scene-level proposal centres
            len_num_max:   L
            frames:        (B*L, T, H) optional raw frame features
            frame_mask:    (B*L, 1, 1, T) optional attention mask over frames
            relation:      (B*L, H) the elected relation vector from AudioModule's role
                           attention. Required when RSK is on, and deliberately NOT a slot
                           index: with no text supervision nothing anchors a numbered slot
                           to a linguistic role (models/audio_module.py).

        Returns:
            proposals (B*L, M, H), plan (B*L, K, M)
        """
        if self.rsk is not None:
            if relation is None:
                raise ValueError(
                    "RSK needs the elected relation vector. Pass relation=<(B*L, H)>; "
                    "no slot index reliably carries the relation role in an audio-only "
                    "model.")
            scene_mask = proposal_mask.view(centers.shape[0], len_num_max, -1)[:, 0]
            geo_bias = self.rsk(relation, centers, len_num_max,
                                 objectness_mask=scene_mask)
        else:
            geo_bias = default_distance_bias(centers, self.num_heads)
            geo_bias = geo_bias.unsqueeze(1).expand(-1, len_num_max, -1, -1, -1)
            geo_bias = geo_bias.reshape(proposals.shape[0], self.num_heads,
                                        proposals.shape[1], proposals.shape[1])

        plan = None
        for i, layer in enumerate(self.layers):
            proposals, slots, plan = layer(proposals, slots, slot_mass, proposal_mask)
            proposals = self.geo_attn[i](proposals, proposals, proposals,
                                         attention_weights=geo_bias, way="add")

        if frames is not None and AUDIO.FRAME_RESIDUAL:
            proposals = self.frame_attn(proposals, frames, frames, frame_mask)

        return proposals, plan


class XAttnFusion(nn.Module):
    """Baseline: proposals cross-attend over audio frames and slots, no transport.

    This is "attentive fusion of audio and point-cloud features" -- the strong baseline
    that already beats AP-Refer. It shares the geometric stack with SPOT, so an
    ablation between them isolates the OT routing and the bidirectional slot update.
    """

    def __init__(self, hidden_size=None, num_heads=None, depth=None, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        self.num_heads = num_heads or AUDIO.NUM_HEADS
        self.depth = depth or AUDIO.FUSION_DEPTH
        head_dim = self.hidden_size // self.num_heads

        def attn():
            return MultiHeadAttention(d_model=self.hidden_size, d_k=head_dim, d_v=head_dim,
                                      h=self.num_heads, dropout=dropout)

        self.frame_attn = nn.ModuleList(attn() for _ in range(self.depth))
        self.slot_attn = nn.ModuleList(attn() for _ in range(self.depth))
        self.geo_attn = nn.ModuleList(attn() for _ in range(self.depth))

    def forward(self, proposals, slots, slot_mass, proposal_mask, centers, len_num_max,
                frames=None, frame_mask=None, relation=None):
        geo_bias = default_distance_bias(centers, self.num_heads)
        geo_bias = geo_bias.unsqueeze(1).expand(-1, len_num_max, -1, -1, -1)
        geo_bias = geo_bias.reshape(proposals.shape[0], self.num_heads,
                                    proposals.shape[1], proposals.shape[1])

        for i in range(self.depth):
            if frames is not None:
                proposals = self.frame_attn[i](proposals, frames, frames, frame_mask)
            proposals = self.slot_attn[i](proposals, slots, slots)
            proposals = self.geo_attn[i](proposals, proposals, proposals,
                                         attention_weights=geo_bias, way="add")
        return proposals, None


class ConcatFusion(nn.Module):
    """Baseline: AP-Refer's fusion. Mean-pooled audio broadcast onto proposals, masked,
    MLP. Reproduced so the comparison in docs/3DVGA.md has a like-for-like arm rather than
    a number copied out of the paper under a different detector."""

    def __init__(self, hidden_size=None, **_):
        super().__init__()
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(self, proposals, slots, slot_mass, proposal_mask, centers, len_num_max,
                frames=None, frame_mask=None, relation=None):
        if frames is not None:
            valid = (~frame_mask.squeeze(1).squeeze(1)).float().unsqueeze(-1)
            pooled = (frames * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            pooled = slots.mean(dim=1)
        pooled = pooled.unsqueeze(1).expand(-1, proposals.shape[1], -1)
        fused = self.mlp(torch.cat([proposals, pooled], dim=-1))
        fused = fused * proposal_mask.unsqueeze(-1).float()
        return fused, None


def build_fusion(variant=None, **kwargs):
    variant = variant or AUDIO.FUSION
    return {"spot": SPOTFusion, "xattn": XAttnFusion, "concat": ConcatFusion}[variant](**kwargs)
