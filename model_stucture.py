"""
model_stucture.py -- the whole 3D-VGA model in one file.

Every module the model is built from, copied verbatim out of the repository and laid out in
the order the data flows through it, so that the architecture can be read end to end without
opening nine files. Nothing here is a reimplementation: each section is the exact source of
the file named in its banner, and each section keeps that file's own documentation.

    waveform  ->  frozen wav2vec2 frames + 6-channel prosody
              ->  AudioFrameEncoder     (FiLM conditioning, normalised-time encoding, saliency)
              ->  SlotInduction         (Sinkhorn: frames -> K role slots)
              ->  AudioModule           (role election: referent and relation)

    point cloud -> PointNet++ / Group-Free -> M proposals
              ->  SpatialGraphNet       (masked kNN graph over proposal centres)
              ->  SPOTFusion            (bidirectional slots <-> proposals, OT routing, RSK)
              ->  AudioMatchModule      (scoring head + the plan's referent row)
              ->  one grounding score per proposal

This file contains model code only. Nothing about datasets, training loops, optimisers,
losses, evaluation or checkpointing appears here; those live in lib/ and scripts/.

The detector half (PointNet++ backbone, voting, proposal decoder, Group-Free) is imported
rather than copied. It is inherited byte-for-byte from the text-grounding model -- that is
what lets a detector checkpoint transfer between the two tasks -- and it depends on compiled
pointnet2 operators, so a copy of it here could not be self-contained anyway.

Read it top to bottom: primitives, transport, acoustic branch, geometric kernel, fusion,
head, and finally AudioRefNet, which wires all of it together, with CachedAudioRefNet --
the same model minus the detector, fed from the scene cache -- beside it.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- detector half
# Unchanged upstream code; see the module docstring above for why it is imported and not
# copied. The shim keeps this file importable for reading on a machine without the compiled
# pointnet2 operators, and fails with a readable message only if a detector is constructed.
try:
    from models.backbone_module import Pointnet2Backbone
    from models.group_free.GF_detector import GroupFreeDetector
    from models.proposal_module import ProposalModule
    from models.voting_module import VotingModule
except ImportError as _exc:                                     # pragma: no cover
    def _unavailable(name, exc):
        def _raise(*_args, **_kwargs):
            raise ImportError(
                f"{name} needs the repository's detector modules and the pointnet2 "
                f"operators, which could not be imported here: {exc}")
        return _raise

    Pointnet2Backbone = _unavailable("Pointnet2Backbone", _exc)
    GroupFreeDetector = _unavailable("GroupFreeDetector", _exc)
    ProposalModule = _unavailable("ProposalModule", _exc)
    VotingModule = _unavailable("VotingModule", _exc)



# =============================================================================
#  Configuration the model reads
#  mirrors lib/audio_config.py
# =============================================================================
#
#  Every switch the modules below consult, with the values the proposed model uses. The
#  ablation arms in experiments/audio/fusion_arms.py are exactly assignments to these names.
#
#  When the repository is importable the live configuration wins, so this file can never
#  silently describe a different model from the one a run builds; the block is a snapshot so
#  that the file also stands alone.


class _AudioConfig:
    """Snapshot of the model-facing half of lib/audio_config.py."""

    ENCODER_DIM = 768            # width of the frozen speech encoder
    HIDDEN_SIZE = 128            # H, the model width
    NUM_HEADS = 4
    NUM_SLOTS = 6                # K acoustic role slots
    NUM_ROLES = 2
    ROLE_REFERENT = 0
    ROLE_RELATION = 1

    USE_PROSODY = True
    PROSODY_DIM = 6              # log-energy, d-energy, log-F0, voicing, flux, boundary
    PROSODY_CHANNELS = "all"     # "all" / "energy" / "pitch" / "spectral" / "none"

    SLOT_TIME_PRIOR = True       # learnable Gaussian prior over normalised time
    SLOT_SINKHORN_EPS = 0.3
    SLOT_SINKHORN_ITERS = 8
    SLOT_DROPOUT = 0.1

    ROLE_ELECTION = True         # False -> read fixed slot indices instead
    FUSION = "spot"              # "spot" / "xattn" / "concat"
    FUSION_DEPTH = 3             # D SPOT layers
    OT_ROUTING = True            # False -> row softmax instead of Sinkhorn
    OT_EPS = 0.3
    OT_ITERS = 8
    OT_LOGIT_WEIGHT = 0.5        # initial weight of the plan's referent row in the logit
    BIDIRECTIONAL_SLOTS = True   # False -> drop SPOT step (ii)
    ROUTE_GATE = True
    FRAME_RESIDUAL = True

    USE_RSK = True
    RSK_MIN_SIGMA = 0.1          # metres
    RSK_MAX_MU = 4.0             # metres

    GRAPH_K = 5                  # 0 removes the spatial graph
    GRAPH_LAYERS = 6

    LOSS_SLOT_ENTROPY = 0.02     # read here only by plan_entropy's docstring
    DISABLE_COPY_PASTE = False
    POINTNET_IMPL = None


try:                                                            # prefer the live config
    from lib.audio_config import AUDIO, PROSODY_GROUPS
except ImportError:                                             # pragma: no cover
    AUDIO = _AudioConfig()
    PROSODY_GROUPS = {"all": (0, 1, 2, 3, 4, 5), "energy": (0, 1), "pitch": (2, 3),
                      "spectral": (4, 5), "none": ()}



# =============================================================================
#  Attention primitives
#  copied verbatim from models/transformer/attention.py
# =============================================================================


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        '''
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries)
        q = q.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            if way == 'mul':
                att = att * attention_weights
            elif way == 'add':
                att = att + attention_weights
            else:
                raise NotImplementedError(way)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way='mul'):
        '''
        :param attention_pos: Mask over attention values (b_s, nq, pk). True indicates masking indices. pk << nk_real
        '''
        b_s, nq, d_model = queries.shape
        nk = keys.shape[1]
        pk = attention_pos.shape[2]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)[: ,:, :, None, :]  # (b_s, h, nq, 1, d_k)

        attention_pos = attention_pos.view(b_s, nq*pk)

        k = self.fc_k(keys)
        v = self.fc_v(values)
        i_ind = torch.arange(b_s)[:, None].type_as(attention_pos) * nq
        attention_pos = attention_pos + i_ind.long()  # index into the flattened (b_s*nq) key/value table
        k = k.view(b_s*nq, -1)[attention_pos].view(b_s, nq, pk, self.h, self.d_k)  # (b_s, nq, pk, h, dk)
        v = v.view(b_s*nq, -1)[attention_pos].view(b_s, nq, pk, self.h, self.d_v)  # (b_s, nq, pk, h, dv)

        k = k.permute(0, 3, 1, 4, 2)  # (b_s, h, nq, d_k, pk)
        v = v.permute(0, 3, 1, 2, 4)  # (b_s, h, nq, pk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, 1, p_k)
        if attention_weights is not None:
            attention_weights = attention_weights[:, :, :, None, :]
            if way == 'mul':
                att = att * attention_weights
            elif way == 'add':
                att = att + attention_weights
            else:
                raise NotImplementedError(way)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v)  # (b_s, h, nq, 1, d_v)
        out = out.permute(0, 2, 1, 4, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m = 20)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention.forward_faster(q_norm, k_norm, v_norm, attention_pos, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention.forward_faster(queries, keys, values, attention_pos, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out



# =============================================================================
#  Position-wise feed-forward
#  copied verbatim from models/transformer/utils.py
# =============================================================================


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out



# =============================================================================
#  Entropic optimal transport (used twice)
#  copied verbatim from models/fusion/sinkhorn.py
# =============================================================================
#
#  Log-domain Sinkhorn, used twice in the 3D-VGA model for two different jobs.
#
#  Both are assignment problems that the usual softmax attention answers badly:
#
#    1. frames -> slots (models/audio_module.py). Plain cross-attention normalises over
#       frames per slot, so nothing stops every slot from landing on the same loud syllable.
#       Sinkhorn adds the *column* constraint -- each frame's mass is also spread over slots --
#       which forces the slots apart and makes them partition the utterance.
#
#    2. slots -> proposals (models/fusion/spot_fusion.py). Softmax attention lets every
#       proposal independently decide how much it likes the "chair" slot, so in a room with
#       nine chairs all nine score high and the ranking that follows is nearly arbitrary.
#       Sinkhorn makes proposals *compete* for a slot's fixed mass budget: one chair can only
#       win the slot at the others' expense. Grounding is an assignment, so this is the
#       operator that matches the task.
#
#  Everything runs in log space (no exp of a large negative), is masked, and is differentiable
#  in a handful of iterations -- the standard entropic-OT setup as used in SuperGlue and
#  SwAV, with unequal marginals so a slot can declare itself unused.
#


def masked_log_softmax(logits, mask, dim):
    """log_softmax that treats masked entries as absent, not as -inf competitors.

    torch.log_softmax over an all-masked row yields NaN. Rows like that do occur here --
    a padded annotation has no valid frames -- so masked entries are pushed to a large
    finite negative and the result is re-masked afterwards.
    """
    logits = logits.masked_fill(~mask, -1e4)
    out = torch.log_softmax(logits, dim=dim)
    return out.masked_fill(~mask, -1e4)

def sinkhorn_log(scores, mask, row_marginal=None, col_marginal=None, eps=0.05, n_iters=5):
    """Entropic optimal transport between the two axes of ``scores``.

    Args:
        scores: (B, R, C) similarity, higher = cheaper to transport.
        mask:   (B, R, C) bool, True where a pairing is allowed.
        row_marginal: (B, R) non-negative mass per row. Uniform if None. Marginals are
            used as given -- nothing here renormalises them -- so a caller supplying both
            must give them the same total, or the iteration converges to neither. What the
            callers do supply is an *uneven* row marginal summing to one: a row holding
            little mass is how a slot says "this role is not mentioned", which the text
            branch writes with that literal token and which audio has no way to utter.
        col_marginal: (B, C) non-negative mass per column. Uniform over valid columns if
            None.
        eps: entropic temperature. What governs convergence is the ratio between the
            *range* of ``scores`` and eps, not eps alone, so both callers feed cosine
            similarities times a bounded learned scale -- a fixed range -- and leave eps
            fixed. Measured on this problem's shapes (6 slots x 128 frames, 6 slots x 256
            proposals), with scores in [-1, 1]: eps 0.3 and 8 iterations leave under 2%
            relative error on the row marginals, while eps 0.05 and 5 iterations leave
            30-50%. Sharpness is the learned scale's job; eps only has to keep the
            iteration converging.
        n_iters: Sinkhorn iterations. This runs inside the training loop twice per layer,
            so the count is the thing being bought with the eps above.

    Returns:
        (B, R, C) transport plan, non-negative, zero outside the mask.
    """
    batch, n_rows, n_cols = scores.shape
    device, dtype = scores.device, scores.dtype

    if row_marginal is None:
        row_marginal = scores.new_full((batch, n_rows), 1.0 / n_rows)
    if col_marginal is None:
        valid = mask.any(dim=1).to(dtype)                       # (B, C)
        col_marginal = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    log_mu = torch.log(row_marginal.clamp_min(1e-8))
    log_nu = torch.log(col_marginal.clamp_min(1e-8))

    # Kernel in log space. Masked pairs are held at a large finite negative rather than
    # -inf, so a fully masked row cannot turn a logsumexp into -inf and the dual into NaN.
    neg = -1e4
    log_k = (scores / eps).masked_fill(~mask, neg)

    # A column no row may use (a padding frame, a non-object proposal) gets no mass, so its
    # dual is pinned rather than left to the iteration. Its v would otherwise grow to
    # ~ +1e4 on the first update -- the only way to give a -1e4 kernel the mass its
    # marginal asks for -- and log_k + v would be back near zero on the next row update.
    # The clamp on log_nu caps that leak at ~1e-8 of the total mass, so measured against a
    # peak plan entry of ~1e-2 it moves nothing (checked: a relative change below 1e-6);
    # this makes the guarantee exact instead of merely small, for one comparison per pass.
    row_valid = mask.any(dim=2)
    col_valid = mask.any(dim=1)

    u = torch.zeros(batch, n_rows, device=device, dtype=dtype)
    v = torch.zeros(batch, n_cols, device=device, dtype=dtype)

    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(log_k + v.unsqueeze(1), dim=2)
        u = u.masked_fill(~row_valid, neg)
        v = log_nu - torch.logsumexp(log_k + u.unsqueeze(2), dim=1)
        v = v.masked_fill(~col_valid, neg)

    # After the final column update the plan is exactly column-normalised, so every
    # exponent is <= 0 and this cannot overflow.
    plan = torch.exp(log_k + u.unsqueeze(2) + v.unsqueeze(1))
    return plan.masked_fill(~mask, 0.0)

def plan_to_attention(plan, dim, eps=1e-8):
    """Row- or column-normalise a transport plan so it can be used as attention weights.

    The plan's rows sum to the slot masses, not to one, which is the point of it -- but the
    moment it multiplies a value tensor it has to be a convex combination, or an unused
    slot's output silently shrinks toward zero and the residual path sees a scale change
    rather than an information change.
    """
    return plan / plan.sum(dim=dim, keepdim=True).clamp_min(eps)

def plan_entropy(plan, eps=1e-8):
    """Mean entropy of the plan's rows, as a scalar regulariser.

    Maximising it early keeps the slots from collapsing onto one frame before the encoder
    has learned anything; the loss weight (AUDIO.LOSS_SLOT_ENTROPY) is small, because the
    endpoint wanted is a sharp assignment, not a smooth one.
    """
    normalised = plan_to_attention(plan, dim=-1, eps=eps)
    return -(normalised * torch.log(normalised.clamp_min(eps))).sum(dim=-1).mean()



# =============================================================================
#  Prosody-augmented acoustic frame encoder
#  copied verbatim from models/audio_encoder.py
# =============================================================================
#
#  Prosody-Augmented Audio Frame Encoder (PA-AFE).
#
#  Takes frozen wav2vec2/WavLM/HuBERT frames from the cache and turns them into the model's
#  working width, with three things the frozen encoder does not provide:
#
#    * **Prosodic conditioning.** Self-supervised speech encoders are trained to be invariant
#      to pitch and loudness -- that is what makes them good phone recognisers and exactly
#      what loses the cue this task needs. In "the *chair* next to the table", the pitch accent
#      on the head noun is the speaker telling you which word is the referent, and the pause
#      before "next to" is the speaker telling you where the relational clause starts. The
#      prosody stream is injected by FiLM rather than concatenated, so six channels can
#      modulate 768 without being drowned by them.
#
#    * **Position.** Cross-attention is permutation-invariant, but a referring expression is
#      strongly ordered: the target noun comes early, the anchor late. Frames carry a
#      normalised-time encoding so the slot module downstream can use that order.
#
#    * **Saliency.** A scalar per frame, learned, marking content-bearing frames. Roughly half
#      of any utterance is function words and silence; this is what lets the slot module spend
#      its attention budget on the rest.
#


class FiLM(nn.Module):
    """Feature-wise linear modulation: h <- gamma(p) * h + beta(p).

    gamma is centred at 1 and beta at 0 (the last layer is zero-initialised), so the module
    starts as the identity. Without that, a randomly initialised FiLM multiplies the
    encoder features by noise for the first few hundred steps and the run never recovers.
    """

    def __init__(self, cond_dim, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return features * (1.0 + gamma) + beta

def time_encoding(lengths, max_len, dim, device):
    """Sinusoidal encoding of *normalised* position t/T, not absolute frame index.

    Absolute position would make the model read a 2 s utterance and a 5 s one differently
    at the same word; normalised position keeps "one third of the way in" meaning the same
    thing at any speaking rate. Speaking rate itself is not lost -- it survives in the
    prosody stream.
    """
    steps = torch.arange(max_len, device=device, dtype=torch.float32)          # (T,)
    denom = lengths.clamp_min(1).to(torch.float32).unsqueeze(1)                # (B, 1)
    position = (steps.unsqueeze(0) / denom).clamp(0.0, 1.0)                    # (B, T)

    half = dim // 2
    freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                     * (-torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1)))
    angle = position.unsqueeze(-1) * freq.view(1, 1, -1) * 10.0                # (B, T, half)

    encoding = torch.zeros(position.shape[0], max_len, dim, device=device)
    encoding[..., 0:2 * half:2] = torch.sin(angle)
    encoding[..., 1:2 * half:2] = torch.cos(angle)
    return encoding

def length_mask(lengths, max_len):
    """(B, T) bool, True on real frames."""
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)

class AudioFrameEncoder(nn.Module):
    """(B, T, C_enc) frozen frames + (B, T, P) prosody -> (B, T, H) contextual frames."""

    def __init__(self, in_dim=None, hidden_size=None, prosody_dim=None, num_heads=None,
                 depth=2, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim or AUDIO.ENCODER_DIM
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        # PROSODY_CHANNELS selects a cue group (energy / pitch / spectral / all), so the
        # ablation can say WHICH prosodic cue carries the signal rather than only that
        # prosody as a whole helps. The selection is on the channel axis, so the FiLM
        # generator shrinks with it and the arms differ in more than a zeroed input.
        self.prosody_channels = list(PROSODY_GROUPS[AUDIO.PROSODY_CHANNELS])
        self.prosody_dim = len(self.prosody_channels) or (prosody_dim or AUDIO.PROSODY_DIM)
        self.use_prosody = bool(AUDIO.USE_PROSODY) and bool(self.prosody_channels)
        heads = num_heads or AUDIO.NUM_HEADS

        self.project = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_size),
        )

        if self.use_prosody:
            self.prosody_norm = nn.LayerNorm(self.prosody_dim)
            self.film = FiLM(self.prosody_dim, self.hidden_size)

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=self.hidden_size,
                               d_k=self.hidden_size // heads,
                               d_v=self.hidden_size // heads,
                               h=heads, dropout=dropout)
            for _ in range(depth)
        )

        # Saliency is read off the frame *and* its prosody, so a loud stressed frame can be
        # marked even when its phonetic content is unremarkable.
        saliency_in = self.hidden_size + (self.prosody_dim if self.use_prosody else 0)
        self.saliency = nn.Sequential(
            nn.Linear(saliency_in, self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.out_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, frames, prosody, lengths):
        """
        Args:
            frames:  (B, T, C_enc) float
            prosody: (B, T, P) float
            lengths: (B,) long, real frame count

        Returns:
            features:  (B, T, H)   contextual frame features
            saliency:  (B, T)      per-frame content score (log-scale, unbounded)
            mask:      (B, T)      bool, True on real frames
            attn_mask: (B, 1, 1, T) bool, True where MultiHeadAttention should mask
        """
        batch, max_len = frames.shape[:2]
        mask = length_mask(lengths, max_len)

        features = self.project(frames)
        features = features + time_encoding(lengths, max_len, self.hidden_size, frames.device)

        if self.use_prosody:
            if len(self.prosody_channels) != prosody.shape[-1]:
                prosody = prosody[..., self.prosody_channels]
            prosody = self.prosody_norm(prosody)
            features = self.film(features, prosody)

        # MultiHeadAttention's convention: True means "mask this out".
        attn_mask = (~mask).view(batch, 1, 1, max_len)
        for layer in self.self_attn:
            features = layer(features, features, features, attn_mask)

        features = self.out_norm(features)
        # Zero the padded tail so anything pooling over frames without re-masking still
        # gets the right answer.
        features = features * mask.unsqueeze(-1).float()

        saliency_in = torch.cat([features, prosody], dim=-1) if self.use_prosody else features
        saliency = self.saliency(saliency_in).squeeze(-1)
        saliency = saliency.masked_fill(~mask, -1e4)

        return features, saliency, mask, attn_mask



# =============================================================================
#  Acoustic role-slot induction and role election
#  copied verbatim from models/audio_module.py
# =============================================================================
#
#  AudioModule: the acoustic counterpart of models/lang_module.py.
#
#  LangModule's real contribution in this repo is not the GRU -- it is that an LLM has already
#  split the sentence into ``target`` / ``adjectives`` / ``neighbors``, and the fusion head
#  then runs one cross-attention stream per field. That decomposition is where the accuracy
#  comes from, and it is the first thing 3DVG-Audio loses: you cannot run GPT-4o-mini on a
#  waveform. Transcribing first would work, but it puts an ASR system back on the critical
#  path and hands the task's error budget to word-error rate.
#
#  So this module induces the decomposition acoustically. K learnable slot queries compete for
#  the utterance's frames under a Sinkhorn (doubly-stochastic) constraint, which forces them
#  to *partition* the audio instead of all landing on the same loud syllable. Three biases
#  steer the partition toward linguistic roles without any labels:
#
#    * a per-slot learnable Gaussian over normalised time -- English referring expressions are
#      ordered, so the slots start life spread across the utterance;
#    * frame saliency from the prosody-aware encoder -- accented frames are content words;
#    * an orthogonality penalty between slot vectors, so they cannot converge on one role.
#
#  Nothing here reads text, at training time or any other time. There is no transcript, no
#  ASR pass and no distillation: the decomposition is induced from the acoustic signal alone,
#  and the only direct supervision any role receives comes from GEOMETRY -- the in-batch
#  contrastive term that ties the elected referent to the ground-truth proposal.
#
#  That is a harder problem than the supervised alternative (Audio-3DVG's Object Mention
#  Detection classifies mentioned object classes against a text-derived label set), and it is
#  the point: a class vocabulary cannot represent "brown" or "next to", and it discards word
#  order. It is also more fragile, which is why analysis.py slot-roles exists.
#
#  Interface mirrors LangModule so the fusion head can stay symmetric:
#
#      data_dict["audio_fea"]            (B*L, T, H)   contextual frames
#      data_dict["audio_slots"]          (B*L, K, H)   role slots
#      data_dict["audio_slot_plan"]      (B*L, K, T)   frame -> slot transport plan
#      data_dict["audio_slot_mass"]      (B*L, K)      how much of the utterance each slot won
#      data_dict["audio_role_attn"]      (B*L, R, K)   which slot won each role
#      data_dict["audio_referent"]       (B*L, H)      elected referent vector
#      data_dict["audio_relation"]       (B*L, H)      elected relation vector -> RSK
#      data_dict["audio_emb"]            (B*L, H)      global utterance embedding
#      data_dict["audio_attention_mask"] (B*L, 1, 1, T)
#      data_dict["audio_scores"]         (B*L, n_class) AudioCls head -> L_AudioCls
#


class SlotInduction(nn.Module):
    """Frames -> K role slots, by entropic optimal transport."""

    def __init__(self, hidden_size, num_slots, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots

        self.queries = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)

        # Slot logits are cosine similarities times this, so the cost range is bounded and
        # Sinkhorn's convergence does not drift as the encoder's feature norms grow. An
        # unbounded dot product is the failure here: at a cost range of +-10 the marginals
        # are 40% off after 8 iterations, whatever eps is set to. Sharpness is bought with
        # this scale (up to 2.0), not by lowering eps.
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Learnable Gaussian prior over normalised time, one per slot. Means start evenly
        # spread over the utterance; sigma starts wide (~0.3) so the prior is a nudge and
        # not a cage, and both are free to move.
        self.time_mu = nn.Parameter(torch.linspace(0.15, 0.85, num_slots))
        self.time_log_sigma = nn.Parameter(torch.full((num_slots,), -1.2))
        self.prior_gate = nn.Parameter(torch.tensor(1.0))
        self.saliency_gate = nn.Parameter(torch.tensor(0.5))

        # Per-utterance slot budget: how much of this utterance each role should claim.
        # This is the module's way of saying "no relation was spoken" -- that slot simply
        # wins very little mass, and everything downstream scales with it.
        self.mass_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_slots),
        )

        self.refine = PositionWiseFeedForward(d_model=hidden_size, d_ff=hidden_size * 2,
                                              dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features, saliency, mask):
        """
        Args:
            features: (B, T, H)
            saliency: (B, T)   -1e4 on padding
            mask:     (B, T)   bool

        Returns:
            slots (B, K, H), plan (B, K, T), mass (B, K), attention (B, K, T)
        """
        batch, max_len = features.shape[:2]
        device = features.device

        queries = F.normalize(self.to_q(self.queries), dim=-1)                 # (K, H)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)                   # (B, K, H)
        keys = F.normalize(self.to_k(features), dim=-1)                       # (B, T, H)
        values = self.to_v(features)

        scale = self.logit_scale.clamp(0.25, 2.0)
        scores = torch.bmm(queries, keys.transpose(1, 2)) * scale             # (B, K, T)

        # Every additive bias below is bounded too, for the same reason: an unbounded term
        # would put the total cost range back outside what Sinkhorn closes in 8 iterations.
        if AUDIO.SLOT_TIME_PRIOR:
            lengths = mask.sum(dim=1).clamp_min(1).float()                    # (B,)
            steps = torch.arange(max_len, device=device, dtype=torch.float32)
            position = steps.unsqueeze(0) / lengths.unsqueeze(1)              # (B, T)
            mu = self.time_mu.view(1, -1, 1)
            sigma = self.time_log_sigma.exp().clamp_min(0.05).view(1, -1, 1)
            prior = -((position.unsqueeze(1) - mu) ** 2) / (2.0 * sigma ** 2)
            scores = scores + self.prior_gate.clamp(0.0, 2.0) * prior.clamp_min(-4.0)

        # Saliency is shared across slots: it says which frames are worth claiming at all,
        # not which slot should claim them. tanh, because the encoder's saliency head is a
        # plain linear layer and nothing else bounds it.
        scores = scores + self.saliency_gate.clamp(0.0, 2.0) * torch.tanh(saliency).unsqueeze(1)

        pair_mask = mask.unsqueeze(1).expand(-1, self.num_slots, -1)          # (B, K, T)

        # Row marginal: a softmax over slots, so the budget is a *relative* allocation and
        # the row and column totals agree (both 1). Pooling is masked -- padded frames are
        # zeroed by the encoder, but the divisor still has to be the real length.
        pooled = (features * mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        row_marginal = torch.softmax(self.mass_head(pooled), dim=-1)          # (B, K)

        plan = sinkhorn_log(scores, pair_mask, row_marginal=row_marginal,
                            eps=AUDIO.SLOT_SINKHORN_EPS,
                            n_iters=AUDIO.SLOT_SINKHORN_ITERS)                # (B, K, T)

        attention = plan_to_attention(plan, dim=-1)                           # rows sum to 1
        slots = torch.bmm(attention, values)                                  # (B, K, H)
        slots = self.norm(slots + self.queries.unsqueeze(0))                  # residual on the query
        slots = self.refine(slots)

        mass = plan.sum(dim=-1)                                               # (B, K)
        return slots, plan, mass, attention

class AudioModule(nn.Module):
    """Frozen-encoder frames in, role slots and an utterance embedding out."""

    def __init__(self, num_text_classes, in_dim=None, hidden_size=None, num_slots=None,
                 dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        self.num_slots = num_slots or AUDIO.NUM_SLOTS
        self.num_text_classes = num_text_classes

        self.encoder = AudioFrameEncoder(in_dim=in_dim, hidden_size=self.hidden_size,
                                         dropout=dropout)
        self.slots = SlotInduction(self.hidden_size, self.num_slots, dropout=dropout)

        # ROLE ELECTION.  Two learned queries attend over the slot bank and pool it into a
        # referent vector and a relation vector.
        #
        # This is not decoration over a simpler `slots[:, 0]` and `slots[:, 3]`. Those
        # indices were only meaningful while a distillation loss pinned them there from the
        # text parse. The model is audio-only now, so nothing anchors slot number 3 to the
        # concept "relation" -- reading it as one would be an unenforced assumption sitting
        # directly underneath the geometric kernel, which is the component most sensitive
        # to getting the wrong vector. Which slot plays which role has to be learned, so it
        # is learned, and analysis.py reports the election so the assumption is auditable.
        self.role_queries = nn.Parameter(torch.randn(AUDIO.NUM_ROLES, self.hidden_size) * 0.02)
        self.role_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.role_scale = nn.Parameter(torch.tensor(1.0))

        # AudioCls, i.e. Eq. (3) of the 3DVG-Audio paper: MLP(Mean(F_audio)) over the 18
        # ScanNet classes. Kept identical so the L_AudioCls ablation is comparable to
        # theirs -- except that the pooled vector here is the *elected referent*,
        # concatenated with the mean, since "which object class was named" is a property of
        # the referring phrase and not of the whole utterance.
        self.audio_cls = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_text_classes),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(self, data_dict):
        frames = data_dict["audio_feat_list"]          # (B, L, T, C)
        prosody = data_dict["prosody_list"]            # (B, L, T, P)
        lengths = data_dict["audio_len_list"]          # (B, L)

        batch_size, len_num_max, max_len = frames.shape[:3]
        frames = frames.reshape(batch_size * len_num_max, max_len, -1)
        prosody = prosody.reshape(batch_size * len_num_max, max_len, -1)
        lengths = lengths.reshape(batch_size * len_num_max)

        features, saliency, mask, attn_mask = self.encoder(frames, prosody, lengths)
        slots, plan, mass, slot_attn = self.slots(features, saliency, mask)

        # Slot dropout: zero whole slots at random so no single role becomes load-bearing.
        # The text branch gets the same treatment by word masking in lang_module.py; the
        # acoustic analogue has to act at slot granularity, because masking frames would
        # just be SpecAugment again (already applied in the dataset).
        if self.training and AUDIO.SLOT_DROPOUT > 0:
            keep = (torch.rand(slots.shape[0], self.num_slots, 1, device=slots.device)
                    >= AUDIO.SLOT_DROPOUT).float()
            slots = slots * keep
            mass = mass * keep.squeeze(-1)

        pooled = (features * mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).float()

        # Elect the roles. Cosine logits times a bounded scale, for the same reason the
        # slot costs are bounded: an unbounded dot product would drive this softmax to a
        # hard one-hot early, before any slot means anything.
        queries = F.normalize(self.role_queries, dim=-1)                  # (R, H)
        keys = F.normalize(self.role_proj(slots), dim=-1)                 # (B', K, H)
        role_logits = torch.einsum("rh,bkh->brk", queries, keys)
        role_attn = torch.softmax(role_logits * self.role_scale.clamp(0.5, 4.0), dim=-1)
        if not AUDIO.ROLE_ELECTION:
            # Ablation: go back to reading fixed slot indices, which is what the earlier
            # distilled drafts did. One-hot rows keep every downstream consumer --
            # audio_role_attn contracts the transport plan, and the role-separation loss
            # reads the same tensor -- working on the same shape, so this arm differs from
            # the proposed one in exactly one thing: whether the roles are chosen.
            fixed = torch.zeros_like(role_attn)
            for r in range(AUDIO.NUM_ROLES):
                fixed[:, r, min(r, fixed.shape[-1] - 1)] = 1.0
            role_attn = fixed
        role_vecs = torch.bmm(role_attn, slots)                           # (B', R, H)
        referent = role_vecs[:, AUDIO.ROLE_REFERENT]
        relation = role_vecs[:, AUDIO.ROLE_RELATION]

        data_dict["audio_fea"] = features
        data_dict["audio_slots"] = slots
        data_dict["audio_slot_plan"] = plan
        data_dict["audio_slot_mass"] = mass
        data_dict["audio_slot_attn"] = slot_attn
        data_dict["audio_frame_mask"] = mask
        data_dict["audio_attention_mask"] = attn_mask
        data_dict["audio_saliency"] = saliency
        data_dict["audio_pooled"] = pooled
        data_dict["audio_role_attn"] = role_attn                          # (B', R, K)
        data_dict["audio_referent"] = referent                            # (B', H)
        data_dict["audio_relation"] = relation                            # (B', H)
        data_dict["audio_emb"] = self.global_proj(torch.cat([pooled, referent], dim=-1))
        data_dict["audio_scores"] = self.audio_cls(torch.cat([pooled, referent], dim=-1))

        return data_dict



# =============================================================================
#  RSK -- relation-slot geometric kernel
#  copied verbatim from models/fusion/geometry.py
# =============================================================================
#
#  RSK -- Relation-Slot Geometric Kernel.
#
#  models/match_module.py biases proposal self-attention with a fixed geometric prior:
#
#      dist_weights = cat([-d, -d, 0, 0])          # d = pairwise centre distance
#
#  Two heads are told "nearer is more relevant", two are left alone. It is a good prior and it
#  is the same prior for every utterance -- which is the problem. "The chair *next to* the
#  table" and "the chair *on the far side of* the room" want opposite distance profiles, and
#  "the lamp *above* the desk" wants a direction, not a distance at all. A fixed -d kernel
#  answers all three the same way, and the cost lands almost entirely on the Multiple subset,
#  where relations are the only thing separating same-class objects (AP-Refer: 16.94 / 9.96,
#  against 48.62 / 29.59 on Unique).
#
#  So make the prior a function of what was said. A hypernetwork reads the *relation slot*
#  from the acoustic branch and emits, per attention head, a spatial kernel:
#
#      radial      = -(d_ij - mu)^2 / (2 sigma^2)        a distance *band*, not a monotone decay
#      directional = kappa * (u . r_ij)                  a preferred direction on the unit sphere
#      bias_ij     = lambda * (radial + directional) + (1 - lambda) * (-d_ij)
#
#  mu is where the relation wants its anchor ("next to" -> small, "across from" -> large);
#  u picks out "above" (+z), "under" (-z), "left of" (in-plane); lambda blends against the
#  original prior. lambda is initialised at ~0, so at step zero the module *is* the repo's
#  existing -d prior and training only ever moves away from it if that helps. A new geometric
#  prior that starts out worse than the old one is a slow way to lose, and this avoids it.
#
#  Shapes are chosen to drop straight into ScaledDotProductAttention(..., way='add'), so the
#  existing attention stack does not change at all.
#


class RelationSlotKernel(nn.Module):
    """Relation slot -> per-head spatial attention bias over proposal pairs."""

    def __init__(self, hidden_size, num_heads=4, min_sigma=None, max_mu=None):
        super().__init__()
        self.num_heads = num_heads
        self.min_sigma = min_sigma if min_sigma is not None else AUDIO.RSK_MIN_SIGMA
        self.max_mu = max_mu if max_mu is not None else AUDIO.RSK_MAX_MU

        # 7 parameters per head: direction (3), log-distance-band centre, log-band width,
        # directional sharpness, blend gate.
        self.params_per_head = 7
        self.hyper = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_heads * self.params_per_head),
        )
        # Zero-init the last layer and bias the gate strongly negative: sigmoid(-4) ~ 0.018,
        # so the module opens as the original -d prior and earns its way out of it.
        nn.init.zeros_(self.hyper[-1].weight)
        bias = torch.zeros(num_heads * self.params_per_head)
        bias.view(num_heads, self.params_per_head)[:, 6] = -4.0
        with torch.no_grad():
            self.hyper[-1].bias.copy_(bias)

    def forward(self, relation_slot, centers, len_num_max, objectness_mask=None):
        """
        Args:
            relation_slot:  (B * L, H) the acoustic relation slot.
            centers:        (B, M, 3) proposal centres (scene-level, shared across the L
                            annotations of that scene).
            len_num_max:    L, annotations per scene in this batch.
            objectness_mask:(B, M) bool, True on real objects. Optional.

        Returns:
            (B * L, num_heads, M, M) additive attention bias.
        """
        batch, num_proposals = centers.shape[:2]
        heads = self.num_heads

        params = self.hyper(relation_slot)                                  # (B*L, h*7)
        params = params.view(batch, len_num_max, heads, self.params_per_head)

        direction = params[..., 0:3]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mu = torch.sigmoid(params[..., 3]) * self.max_mu                    # (B, L, h)
        sigma = torch.nn.functional.softplus(params[..., 4]) + self.min_sigma
        kappa = torch.nn.functional.softplus(params[..., 5])
        blend = torch.sigmoid(params[..., 6])

        # Pairwise geometry, computed once per scene rather than per annotation: this is
        # the tensor that would otherwise dominate memory at L > 8.
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)                 # (B, M, M, 3) = c_j - c_i
        distance = delta.norm(dim=-1)                                       # (B, M, M)
        unit = delta / distance.unsqueeze(-1).clamp_min(1e-6)

        radial = -((distance[:, None, None] - mu[..., None, None]) ** 2)
        radial = radial / (2.0 * (sigma[..., None, None] ** 2))             # (B, L, h, M, M)

        directional = torch.einsum("bmnk,blhk->blhmn", unit, direction)
        directional = kappa[..., None, None] * directional

        learned = radial + directional
        baseline = -distance[:, None, None]                                 # the repo's prior
        bias = blend[..., None, None] * learned + (1.0 - blend[..., None, None]) * baseline

        if objectness_mask is not None:
            # Keys that are not objects should not attract mass, whatever the relation says.
            key_mask = objectness_mask[:, None, None, None, :]              # (B,1,1,1,M)
            bias = bias.masked_fill(~key_mask, -1e4)

        return bias.reshape(batch * len_num_max, heads, num_proposals, num_proposals)

def default_distance_bias(centers, num_heads=4):
    """models/match_module.py's fixed prior, as a function, for the --no_rsk ablation.

    Reproduces cat([-d, -d, 0, 0]) exactly for num_heads=4, and generalises it by giving
    the first half of the heads the distance prior and leaving the rest free.
    """
    delta = centers.unsqueeze(1) - centers.unsqueeze(2)
    distance = delta.norm(dim=-1).unsqueeze(1)                              # (B, 1, M, M)
    half = max(1, num_heads // 2)
    return torch.cat([(-distance).expand(-1, half, -1, -1),
                      torch.zeros_like(distance).expand(-1, num_heads - half, -1, -1)],
                     dim=1).detach()



# =============================================================================
#  SPOT fusion, and the two baseline fusions
#  copied verbatim from models/fusion/spot_fusion.py
# =============================================================================
#
#  SPOT -- Slot-Proposal Optimal Transport fusion.
#
#  The proposed fusion operator for 3D visual grounding by audio. Full write-up and the
#  reasoning behind each piece: docs/3DVGA.md.
#
#  --------------------------------------------------------------------------------------
#  What every existing method does, and why it caps out
#  --------------------------------------------------------------------------------------
#
#  AP-Refer (Neurocomputing 611, 2025) concatenates *mean-pooled* audio with seed-point
#  features, multiplies by a mask and runs an MLP. Mean pooling over ~200 wav2vec frames
#  throws away the word order that a referring expression is made of, so "the chair left of
#  the table" and "the table left of the chair" arrive identical. Overall 23.09 / 13.77.
#
#  Attentive fusion -- proposals cross-attend over audio frames -- fixes the pooling and is a
#  clear improvement. But it keeps a deeper flaw: **softmax over the audio axis makes every
#  proposal an independent bidder**. Each of nine chairs in a room attends to the "chair"
#  region of the utterance, each gets a high score, and nothing in the operator expresses that
#  *exactly one* of them is the referent. The ranking that follows is decided by whatever
#  noise survives, which is exactly the Unique/Multiple gap every 3DVG-Audio number shows.
#
#  Grounding is an assignment problem. So use an assignment operator.
#
#  --------------------------------------------------------------------------------------
#  The operator
#  --------------------------------------------------------------------------------------
#
#  Per layer, over K acoustic role slots (from models/audio_module.py) and M proposals:
#
#    (1) Proposals read slots.   Standard cross-attention, cheap because K ~ 6, not ~200
#        frames: the slot bank has already summarised the utterance, so this attends over
#        roles rather than over syllables.
#
#    (2) Slots read proposals.   The reverse direction, which no prior 3DVG-Audio work has:
#        the "anchor" slot should mean something different in a room with one table than in a
#        room with six. Slots become scene-aware before they are used to score anything.
#
#    (3) Optimal-transport routing.   Cost from cosine similarity between slots and
#        proposals; a few Sinkhorn iterations under marginals (slot masses from the audio
#        branch; uniform over valid proposals). The column constraint is the whole point --
#        a slot has a fixed mass budget, so proposals *compete*. Nine chairs can no longer all
#        win the chair slot.
#
#    (4) Routed read-out.   Each proposal is enriched with the slots that claimed it, weighted
#        by the transport plan, through a learned gate.
#
#    (5) Geometric self-attention with the RSK bias (models/fusion/geometry.py), so the
#        spatial prior applied between proposals is the one the *relation slot* asked for.
#
#  The final score mixes the usual MLP head with the target slot's transport row, which is a
#  grounding score in its own right and one that already carries the competition constraint.
#
#  --------------------------------------------------------------------------------------
#
#  `XAttnFusion` and `ConcatFusion` in this file are the two baselines, sharing the scoring
#  head and the training-time copy-paste augmentation so a fusion ablation changes the fusion
#  and nothing else.
#


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



# =============================================================================
#  Fusion head: spatial graph + SPOT + scoring
#  copied verbatim from models/audio_match_module.py
# =============================================================================
#
#  AudioMatchModule: the fusion head of the 3D-VGA model.
#
#  Keeps that module's overall shape -- project detector features, reason over a spatial
#  graph, copy-paste augmentation, a Conv1d scoring stack producing ``cluster_ref`` -- and
#  swaps its four parallel text cross-attention streams (lang / target / adjectives /
#  neighbors) for the acoustic fusion in models/fusion/spot_fusion.py.
#
#  One deliberate departure: the spatial graph is implemented here with a dense masked kNN
#  instead of torch_geometric's ``knn_graph`` + ``GCNConv``. The operation is the same
#  symmetric-normalised propagation over a k-nearest-neighbour graph of proposal centres, but
#  it drops the pyg_lib dependency, which the README lists as the first troubleshooting item
#  ("model can't build at all") because its wheel is keyed to an exact torch build. At M = 256
#  the dense form is also faster than scatter-gather on GPU. The text branch is untouched --
#  models/match_module.py still uses pyg -- so nothing existing changes behaviour.
#


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



# =============================================================================
#  Top level: AudioRefNet
#  copied verbatim from models/refnet_audio.py
# =============================================================================
#
#  AudioRefNet: models/refnet.py with the language branch replaced by the acoustic one.
#
#      point cloud (N x 10)                     audio (16 kHz)
#            |                                        |
#            v                                        v
#      PointNet++ -> voting -> DETR            frozen wav2vec2/WavLM [offline cache]
#        or GroupFree                                 |
#            |                                  AudioModule: prosody FiLM
#            |                                  -> Sinkhorn slot induction -> K role slots
#            +------------> proposals ----------------+
#                                |
#                                v
#                      AudioMatchModule: SPOT fusion
#                      (bidirectional slots <-> proposals,
#                       OT routing, RSK geometry)
#                                |
#                                v
#                       referred-object scores
#
#  The detector half is byte-for-byte the construction in RefNet -- same GroupFree/VoteNet
#  choice, same config_transformer, same pretrained-weight plumbing -- so a detector
#  checkpoint trained for the text model loads straight into the audio one and the two are
#  comparable under a frozen detector.
#
#  The model is audio-only, and audio-only in the strong sense: no transcript, no ASR pass and
#  no text supervision, at training time or inference time. models/lang_module.py is never
#  constructed here. The text branch of this repository is a baseline to compare against.
#
#  That is a deliberately harder setting than the published alternatives. 3DVG-AudioSet's
#  audio is synthesised *from* ScanRefer text, so the transcript is sitting right there and
#  distilling from it is free -- which is exactly why it is worth refusing. A result obtained
#  with a text teacher does not transfer to speech that was never text, and the task exists
#  for speech that was never text.
#


class AudioRefNet(nn.Module):
    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None, audio_dim=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert mean_size_arr.shape[0] == self.num_size_cluster
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.dataset_config = dataset_config
        self.args = args

        # ---------------------------------------------------------- detector (unchanged)
        if args.detector == "GF":
            self.detector = GroupFreeDetector(
                num_class=num_class,
                num_heading_bin=num_heading_bin,
                num_size_cluster=num_size_cluster,
                mean_size_arr=mean_size_arr,
                input_feature_dim=input_feature_dim,
                width=args.width,
                bn_momentum=args.bn_momentum,
                sync_bn=True if args.syncbn else False,
                num_proposal=args.num_target,
                sampling=args.sampling,
                dropout=args.transformer_dropout,
                activation=args.transformer_activation,
                nhead=args.nhead,
                num_decoder_layers=args.num_decoder_layers,
                dim_feedforward=args.dim_feedforward,
                self_position_embedding=args.self_position_embedding,
                cross_position_embedding=args.cross_position_embedding,
                size_cls_agnostic=True if args.size_cls_agnostic else False)
        else:
            self.backbone_net = Pointnet2Backbone(args=args,
                                                  input_feature_dim=self.input_feature_dim)
            self.vgen = VotingModule(self.vote_factor, 256)
            config_transformer = {
                'mask': 'no_mask',
                'weighted_input': True,
                'transformer_type': 'myAdd_20;deformable',
                'deformable_type': 'myAdd',
                'position_embedding': 'none',
                'input_dim': 0,
                'enc_layers': 0,
                'dec_layers': 2,
                'dim_feedforward': 2048,
                'hidden_dim': 288,
                'dropout': 0.1,
                'nheads': 8,
                'pre_norm': False,
            }
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                           mean_size_arr, num_proposal, sampling,
                                           config_transformer=config_transformer,
                                           dataset_config=dataset_config)

        # ---------------------------------------------------------- acoustic branch
        if not no_reference:
            self.audio = AudioModule(num_text_classes=num_class,
                                     in_dim=audio_dim or AUDIO.ENCODER_DIM)
            self.match = AudioMatchModule(args, num_proposals=num_proposal, det_channel=288)


    # ------------------------------------------------------------------ forward

    def forward(self, data_dict):
        if self.args.detector == "GF":
            data_dict = self.detector(data_dict)
        else:
            data_dict = self.backbone_net(data_dict)

            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features

            xyz, features = self.vgen(xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
            data_dict["vote_features"] = features

            data_dict = self.proposal(xyz, features, data_dict)

        if self.no_reference:
            return data_dict

        data_dict = self.audio(data_dict)
        data_dict = self.match(data_dict)
        return data_dict


class CachedAudioRefNet(nn.Module):
    """Acoustic branch + fusion only. Signature matches AudioRefNet so callers can swap them.

    The detector's output arrives in ``data_dict`` from the scene cache
    (lib/scene_cache.py), so no detector is built and none runs. This is the model every
    fusion experiment trains: both of its inputs -- proposals and encoder frames -- are
    read from disk.
    """

    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None, audio_dim=None):
        super().__init__()

        if no_reference:
            raise ValueError(
                "CachedAudioRefNet trains the acoustic and fusion branches; "
                "no_reference=True leaves it with nothing to train. Use AudioRefNet for "
                "detection-only runs.")

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.no_reference = no_reference
        self.dataset_config = dataset_config
        self.args = args

        self.audio = AudioModule(num_text_classes=num_class,
                                 in_dim=audio_dim or AUDIO.ENCODER_DIM)
        # det_channel=288 is the width of the cached detr_features.
        self.match = AudioMatchModule(args, num_proposals=num_proposal, det_channel=288)

    def forward(self, data_dict):
        """AudioRefNet.forward without the detection branch; data_dict carries the cache."""
        data_dict = self.audio(data_dict)
        data_dict = self.match(data_dict)
        return data_dict


def build_audio_model(**kwargs):
    """AudioRefNet, or the cached-feature variant under AUDIO.USE_CACHED_SCENES."""
    model = (CachedAudioRefNet if AUDIO.USE_CACHED_SCENES else AudioRefNet)(**kwargs)
    if not model.no_reference:
        n_audio = sum(p.numel() for p in model.audio.parameters() if p.requires_grad)
        n_match = sum(p.numel() for p in model.match.parameters() if p.requires_grad)
        print(f"[3D-VGA] audio branch {n_audio/1e6:.2f}M params, "
              f"fusion head {n_match/1e6:.2f}M params, fusion='{AUDIO.FUSION}'")
    return model
