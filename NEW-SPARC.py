"""
NEW-SPARC.py -- SPARC (Speech-Prosodic Anchor-Referent Coupling) as ONE self-contained file.

SPARC extends SPOT with two things, both OFF by default:

  1. Prosodic clock + ordered slots.  The slot time prior is evaluated on a clock that runs
     faster across prosodic boundaries (pauses, phrase breaks), and the slot centres become an
     increasing sequence, so the K slots are consecutive segments of the utterance.
  2. Latent-anchor pair grounding.  A third elected role (the anchor) joins referent and
     relation, and every (target i, anchor j) pair is scored; the anchor is marginalised
     exactly, with an optional null anchor for utterances that name none.

This file holds every SPARC component that the repository's own files do not have, and it
changes no other file. It subclasses the existing modules and patches three names at run time.

--------------------------------------------------------------------------------------------
Use
--------------------------------------------------------------------------------------------
Load the file (the name has a hyphen, so it cannot be `import`ed) and call install() BEFORE
the model is built, e.g. at the top of scripts/audio_train.py / scripts/audio_eval.py or in a
notebook cell:

    import runpy
    runpy.run_path("NEW-SPARC.py")["install"]()

Switches are environment variables, read when install() runs (all default to the SPOT model):

    VGA_PROSODIC_CLOCK=1      prosodic clock on the slot time prior
    VGA_ORDERED_SLOTS=0       free slot centres (default 1; only matters with the clock)
    VGA_LATENT_ANCHOR=1       anchor role + pair grounder
    VGA_ANCHOR_NULL=0         every utterance must have an anchor (default 1)
    VGA_ANCHOR_HARD=1         max over anchors instead of logsumexp (ablation)
    VGA_PAIR_GEOMETRY=0       anchor matched by features only (ablation)
    VGA_PAIR_BASIS=8          number of basis potentials the relation vector mixes

    python NEW-SPARC.py arms      # the seven arms as environment settings
    python NEW-SPARC.py check     # pure-Python arithmetic checks, no torch needed

Status: written against the module layout of this repository (models/audio_module.py,
models/audio_match_module.py, lib/audio_loss_helper.py). It has NOT been executed: the
authoring environment had no PyTorch. Run the repository's selftest with install() applied
before trusting it. Checkpoints trained without SPARC load only with strict=False, because
the new modules add parameters (and the ordered-slot arm replaces `time_mu` by `time_gap`).
"""

import math
import os
import random
import sys

# ============================================================================================
#  Configuration
# ============================================================================================

DEFAULTS = {
    "PROSODIC_CLOCK": False,   # slot time prior on a prosody-warped clock
    "ORDERED_SLOTS": True,     # with the clock: slot centres are an increasing sequence
    "LATENT_ANCHOR": False,    # third elected role + exact marginal over the anchor
    "ANCHOR_NULL": True,       # allow "no anchor was spoken"
    "ANCHOR_HARD": False,      # True: max over anchors (ablation)
    "PAIR_GEOMETRY": True,     # False: anchor by features only (ablation)
    "PAIR_BASIS": 8,           # basis potentials mixed by the relation vector
    "ROLE_ANCHOR": 2,
}

#: The ablation arms, as environment settings. Each differs from "sparc" by one thing.
SPARC_ARMS = {
    "sparc":            {"PROSODIC_CLOCK": 1, "LATENT_ANCHOR": 1},
    "sparc-noclock":    {"PROSODIC_CLOCK": 0, "LATENT_ANCHOR": 1},
    "sparc-noanchor":   {"PROSODIC_CLOCK": 1, "LATENT_ANCHOR": 0},
    "sparc-unordered":  {"PROSODIC_CLOCK": 1, "LATENT_ANCHOR": 1, "ORDERED_SLOTS": 0},
    "sparc-nonull":     {"PROSODIC_CLOCK": 1, "LATENT_ANCHOR": 1, "ANCHOR_NULL": 0},
    "sparc-hardanchor": {"PROSODIC_CLOCK": 1, "LATENT_ANCHOR": 1, "ANCHOR_HARD": 1},
    "sparc-nopairgeom": {"PROSODIC_CLOCK": 1, "LATENT_ANCHOR": 1, "PAIR_GEOMETRY": 0},
}

#: Index of the boundary-strength channel in the 6-channel prosody tensor
#: (lib/audio_config.PROSODY_GROUPS["spectral"] = (flux, boundary)).
BOUNDARY_CHANNEL = 5
#: Dimension of the pair geometry: distance, unit vector (3), vertical offset,
#: horizontal distance, log distance.
GEOM_DIM = 7

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lib.audio_config import AUDIO
    _HAVE_REPO = True
except ImportError:                                            # pragma: no cover
    _HAVE_REPO = False
    AUDIO = None


def _typed_env(raw, current):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def ensure_config():
    """Add the SPARC keys to AUDIO (if absent), fold in VGA_<KEY> overrides, set NUM_ROLES.

    Idempotent. Called again by every SPARC module's constructor, so the role count is right
    even if a module is built before install() ran in this process.
    """
    for key, default in DEFAULTS.items():
        if key not in AUDIO:
            AUDIO[key] = default
        raw = os.environ.get(f"VGA_{key}")
        if raw is not None:
            AUDIO[key] = _typed_env(raw, DEFAULTS[key])
    AUDIO.NUM_ROLES = 3 if AUDIO.LATENT_ANCHOR else 2
    return AUDIO


# ============================================================================================
#  Torch components
# ============================================================================================

if _HAVE_REPO:
    from models.audio_match_module import AudioMatchModule
    from models.audio_module import AudioModule, SlotInduction
    from models.fusion.sinkhorn import plan_to_attention, sinkhorn_log

    # ----------------------------------------------------------------------------------------
    #  Part 1: prosodic clock and ordered slots
    # ----------------------------------------------------------------------------------------

    class ClockSlotInduction(SlotInduction):
        """SlotInduction whose time prior runs on a prosodic clock, with ordered centres.

        The clock advances by w_t = 0.25 + sigmoid(g * z(boundary_t) + c) per frame, so it runs
        faster across a pause. position = (1 - m) * t/T_real + m * clock, with m = sigmoid(
        clock_mix) initialised at sigmoid(-4) ~ 0.018: the module starts at SPOT's linear time
        and training moves away from it only if the warped clock helps. With constant w the
        clock equals t/T exactly.

        Slot centres (when ORDERED_SLOTS) are mu_k = sum_{i<=k} softplus(gap_i), initialised at
        the same linspace(0.15, 0.85) as SPOT's free parameters, so slots cannot swap places.
        """

        def __init__(self, hidden_size, num_slots, dropout=0.1):
            super().__init__(hidden_size, num_slots, dropout=dropout)
            ensure_config()
            if AUDIO.PROSODIC_CLOCK:
                self.clock_gain = nn.Parameter(torch.tensor(1.0))
                self.clock_bias = nn.Parameter(torch.tensor(0.0))
                self.clock_mix = nn.Parameter(torch.tensor(-4.0))
                if AUDIO.ORDERED_SLOTS:
                    del self.time_mu
                    step = (0.85 - 0.15) / max(num_slots - 1, 1)
                    gaps = torch.full((num_slots,), step)
                    gaps[0] = 0.15
                    self.time_gap = nn.Parameter(torch.log(torch.expm1(gaps)))

        def forward(self, features, saliency, mask, boundary=None):
            """Same contract as SlotInduction.forward, plus ``boundary`` (B, T) or None."""
            batch, max_len = features.shape[:2]
            device = features.device

            queries = F.normalize(self.to_q(self.queries), dim=-1)
            queries = queries.unsqueeze(0).expand(batch, -1, -1)
            keys = F.normalize(self.to_k(features), dim=-1)
            values = self.to_v(features)

            scale = self.logit_scale.clamp(0.25, 2.0)
            scores = torch.bmm(queries, keys.transpose(1, 2)) * scale          # (B, K, T)

            if AUDIO.SLOT_TIME_PRIOR:
                lengths = mask.sum(dim=1).clamp_min(1).float()
                steps = torch.arange(max_len, device=device, dtype=torch.float32)
                position = steps.unsqueeze(0) / lengths.unsqueeze(1)           # (B, T)
                if AUDIO.PROSODIC_CLOCK and boundary is not None:
                    valid = mask.float()
                    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean = (boundary * valid).sum(dim=1, keepdim=True) / count
                    var = (((boundary - mean) * valid) ** 2).sum(dim=1, keepdim=True) / count
                    zb = (boundary - mean) / torch.sqrt(var + 1e-6)
                    rate = (0.25 + torch.sigmoid(self.clock_gain * zb + self.clock_bias)) * valid
                    clock = ((rate.cumsum(dim=1) - rate)
                             / rate.sum(dim=1, keepdim=True).clamp_min(1e-6))
                    mix = torch.sigmoid(self.clock_mix)
                    position = (1.0 - mix) * position + mix * clock
                if AUDIO.PROSODIC_CLOCK and AUDIO.ORDERED_SLOTS:
                    mu = F.softplus(self.time_gap).cumsum(dim=0).view(1, -1, 1)
                else:
                    mu = self.time_mu.view(1, -1, 1)
                sigma = self.time_log_sigma.exp().clamp_min(0.05).view(1, -1, 1)
                prior = -((position.unsqueeze(1) - mu) ** 2) / (2.0 * sigma ** 2)
                scores = scores + self.prior_gate.clamp(0.0, 2.0) * prior.clamp_min(-4.0)

            scores = scores + self.saliency_gate.clamp(0.0, 2.0) * torch.tanh(saliency).unsqueeze(1)

            pair_mask = mask.unsqueeze(1).expand(-1, self.num_slots, -1)       # (B, K, T)

            pooled = (features * mask.unsqueeze(-1).float()).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            row_marginal = torch.softmax(self.mass_head(pooled), dim=-1)       # (B, K)

            plan = sinkhorn_log(scores, pair_mask, row_marginal=row_marginal,
                                eps=AUDIO.SLOT_SINKHORN_EPS, n_iters=AUDIO.SLOT_SINKHORN_ITERS)

            attention = plan_to_attention(plan, dim=-1)
            slots = torch.bmm(attention, values)
            slots = self.norm(slots + self.queries.unsqueeze(0))
            slots = self.refine(slots)

            mass = plan.sum(dim=-1)
            return slots, plan, mass, attention

    # ----------------------------------------------------------------------------------------
    #  Part 2a: a third elected role
    # ----------------------------------------------------------------------------------------

    class SparcAudioModule(AudioModule):
        """AudioModule with prosody-clock slots and, optionally, an elected anchor role.

        Differences from AudioModule.forward: the boundary channel is handed to the slot
        module, and with LATENT_ANCHOR a third role vector ``audio_anchor`` and a scalar
        ``audio_anchor_presence`` (~1 when the anchor slot won an average share of the
        utterance, ~0 when it won nothing) are written to the batch.
        """

        def __init__(self, *args, **kwargs):
            ensure_config()                     # NUM_ROLES must be right before super().__init__
            super().__init__(*args, **kwargs)
            self.slots = ClockSlotInduction(self.hidden_size, self.num_slots,
                                            dropout=kwargs.get("dropout", 0.1))

        def forward(self, data_dict):
            frames = data_dict["audio_feat_list"]
            prosody = data_dict["prosody_list"]
            lengths = data_dict["audio_len_list"]

            batch_size, len_num_max, max_len = frames.shape[:3]
            frames = frames.reshape(batch_size * len_num_max, max_len, -1)
            prosody = prosody.reshape(batch_size * len_num_max, max_len, -1)
            lengths = lengths.reshape(batch_size * len_num_max)

            boundary = (prosody[..., BOUNDARY_CHANNEL]
                        if (AUDIO.PROSODIC_CLOCK and AUDIO.USE_PROSODY
                            and prosody.shape[-1] > BOUNDARY_CHANNEL) else None)
            features, saliency, mask, attn_mask = self.encoder(frames, prosody, lengths)
            slots, plan, mass, slot_attn = self.slots(features, saliency, mask, boundary)

            if self.training and AUDIO.SLOT_DROPOUT > 0:
                keep = (torch.rand(slots.shape[0], self.num_slots, 1, device=slots.device)
                        >= AUDIO.SLOT_DROPOUT).float()
                slots = slots * keep
                mass = mass * keep.squeeze(-1)

            pooled = (features * mask.unsqueeze(-1).float()).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).float()

            queries = F.normalize(self.role_queries, dim=-1)
            keys = F.normalize(self.role_proj(slots), dim=-1)
            role_logits = torch.einsum("rh,bkh->brk", queries, keys)
            role_attn = torch.softmax(role_logits * self.role_scale.clamp(0.5, 4.0), dim=-1)
            if not AUDIO.ROLE_ELECTION:
                fixed = torch.zeros_like(role_attn)
                for r in range(AUDIO.NUM_ROLES):
                    fixed[:, r, min(r, fixed.shape[-1] - 1)] = 1.0
                role_attn = fixed
            role_vecs = torch.bmm(role_attn, slots)
            referent = role_vecs[:, AUDIO.ROLE_REFERENT]
            relation = role_vecs[:, AUDIO.ROLE_RELATION]

            if AUDIO.LATENT_ANCHOR:
                anchor_attn = role_attn[:, AUDIO.ROLE_ANCHOR]
                data_dict["audio_anchor"] = role_vecs[:, AUDIO.ROLE_ANCHOR]
                data_dict["audio_anchor_presence"] = ((anchor_attn * mass).sum(dim=-1)
                                                      * self.num_slots)

            data_dict["audio_fea"] = features
            data_dict["audio_slots"] = slots
            data_dict["audio_slot_plan"] = plan
            data_dict["audio_slot_mass"] = mass
            data_dict["audio_slot_attn"] = slot_attn
            data_dict["audio_frame_mask"] = mask
            data_dict["audio_attention_mask"] = attn_mask
            data_dict["audio_saliency"] = saliency
            data_dict["audio_pooled"] = pooled
            data_dict["audio_role_attn"] = role_attn
            data_dict["audio_referent"] = referent
            data_dict["audio_relation"] = relation
            data_dict["audio_emb"] = self.global_proj(torch.cat([pooled, referent], dim=-1))
            data_dict["audio_scores"] = self.audio_cls(torch.cat([pooled, referent], dim=-1))
            return data_dict

    # ----------------------------------------------------------------------------------------
    #  Part 2b: latent-anchor pair grounding
    # ----------------------------------------------------------------------------------------

    def pair_geometry(centers):
        """(B, M, M, 7) geometry of every ordered pair; entry [b, i, j] describes j relative to i."""
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)                    # c_j - c_i
        distance = delta.norm(dim=-1)
        unit = delta / distance.unsqueeze(-1).clamp_min(1e-6)
        horizontal = delta[..., :2].norm(dim=-1)
        return torch.cat([
            distance.unsqueeze(-1),
            unit,
            delta[..., 2:3],
            horizontal.unsqueeze(-1),
            torch.log(distance + 0.1).unsqueeze(-1),
        ], dim=-1)

    class LatentAnchorGrounder(nn.Module):
        """Proposal features + elected roles -> a pair-consistent log-score per proposal.

            s_ij    = u_i + v_j + g_ij
            u_i     = scale * cos(referent, proposal_i)
            v_j     = scale * cos(anchor,   proposal_j)
            g_ij    = 5 tanh( sum_k w_k(relation) phi_k(geometry_ij) / 5 )
            score_i = u_i + logsumexp_j( v_j + g_ij )        j != i, j real, plus a null anchor

        ``gain`` is 0 at init, so the module begins as a no-op and training decides how much of
        it to use. That also blocks gradients to the rest of the module until ``gain`` moves;
        if it stays near 0 for the first epoch, start it at 0.1.
        """

        def __init__(self, hidden_size, n_basis=None):
            super().__init__()
            ensure_config()
            self.n_basis = n_basis or AUDIO.PAIR_BASIS

            self.ref_query = nn.Linear(hidden_size, hidden_size)
            self.ref_key = nn.Linear(hidden_size, hidden_size)
            self.anchor_query = nn.Linear(hidden_size, hidden_size)
            self.anchor_key = nn.Linear(hidden_size, hidden_size)
            self.unary_scale = nn.Parameter(torch.tensor(2.0))

            if AUDIO.PAIR_GEOMETRY:
                self.basis = nn.Sequential(
                    nn.Linear(GEOM_DIM, 32), nn.ReLU(inplace=True), nn.Linear(32, self.n_basis))
                self.mix = nn.Linear(hidden_size, self.n_basis)

            if AUDIO.ANCHOR_NULL:
                self.null_bias = nn.Parameter(torch.tensor(0.0))
                self.null_gain = nn.Parameter(torch.tensor(2.0))

            self.gain = nn.Parameter(torch.tensor(0.0))

        def forward(self, proposals, centers, len_num_max, proposal_mask, referent, anchor,
                    relation, anchor_presence):
            """
            Args:
                proposals:       (B', M, H) fused proposal features
                centers:         (B, M, 3)  scene-level proposal centres
                len_num_max:     L
                proposal_mask:   (B', M) bool, True on real objects
                referent/anchor/relation: (B', H) elected role vectors
                anchor_presence: (B',)

            Returns:
                pair_logit (B', M), anchor_post (B', M), null_mass (B',)
            """
            batch_flat, num_proposals, _ = proposals.shape
            scale = self.unary_scale.clamp(0.5, 8.0)

            ref_q = F.normalize(self.ref_query(referent), dim=-1).unsqueeze(1)
            ref_k = F.normalize(self.ref_key(proposals), dim=-1)
            unary = scale * (ref_k * ref_q).sum(dim=-1)

            anc_q = F.normalize(self.anchor_query(anchor), dim=-1).unsqueeze(1)
            anc_k = F.normalize(self.anchor_key(proposals), dim=-1)
            anchor_fit = scale * (anc_k * anc_q).sum(dim=-1)

            pair = anchor_fit.unsqueeze(1)                                      # (B', 1, M)
            if AUDIO.PAIR_GEOMETRY:
                basis = self.basis(pair_geometry(centers))                      # (B, M, M, nb)
                weights = self.mix(relation).view(centers.shape[0], len_num_max, self.n_basis)
                potential = torch.einsum("bijk,blk->blij", basis, weights)
                potential = potential.reshape(batch_flat, num_proposals, num_proposals)
                potential = 5.0 * torch.tanh(potential / 5.0)
                pair = pair + potential
            else:
                pair = pair.expand(-1, num_proposals, -1)

            eye = torch.eye(num_proposals, dtype=torch.bool, device=proposals.device)
            allowed = proposal_mask.unsqueeze(1) & ~eye.unsqueeze(0)
            pair = pair.masked_fill(~allowed, -1e4)

            if AUDIO.ANCHOR_NULL:
                presence = anchor_presence.clamp(0.0, 2.0)
                null = self.null_bias + self.null_gain * (1.0 - presence)
                null = null.view(-1, 1, 1).expand(-1, num_proposals, 1)
                pair = torch.cat([pair, null], dim=-1)

            if AUDIO.ANCHOR_HARD:
                marginal = pair.max(dim=-1)[0]
            else:
                marginal = torch.logsumexp(pair, dim=-1)

            total = (unary + marginal).masked_fill(~proposal_mask, -1e4)
            pair_logit = torch.log_softmax(total, dim=-1).masked_fill(~proposal_mask, -1e4)

            with torch.no_grad():                       # diagnostics only
                target_p = torch.softmax(total, dim=-1)
                anchor_p = torch.softmax(pair, dim=-1)
                joint = torch.einsum("bi,bij->bj", target_p, anchor_p)
                anchor_post = joint[:, :num_proposals]
                null_mass = (joint[:, num_proposals] if AUDIO.ANCHOR_NULL
                             else torch.zeros_like(joint[:, 0]))
            return pair_logit, anchor_post, null_mass

    class SparcAudioMatchModule(AudioMatchModule):
        """AudioMatchModule plus the pair score, added to ``cluster_ref`` after the base forward.

        The base forward ends with confidence.masked_fill(~mask, -1e4). Adding
        gain * pair_logit afterwards and re-masking is the same arithmetic as adding it before
        the final mask, so the result matches an in-place edit of the base module.
        """

        def __init__(self, *args, **kwargs):
            ensure_config()
            super().__init__(*args, **kwargs)
            self.anchor = (LatentAnchorGrounder(self.hidden_size) if AUDIO.LATENT_ANCHOR
                           else None)

        def forward(self, data_dict):
            data_dict = super().forward(data_dict)
            if self.anchor is None:
                return data_dict

            len_num_max = data_dict["audio_feat_list"].shape[1]
            obj_mask = self._objectness_mask(data_dict)                        # (B, M)
            batch_size, num_proposals = obj_mask.shape
            mask_flat = (obj_mask.unsqueeze(1).expand(-1, len_num_max, -1)
                         .reshape(batch_size * len_num_max, num_proposals))

            pair_logit, anchor_post, null_mass = self.anchor(
                data_dict["fused_proposals"], data_dict["center"], len_num_max, mask_flat,
                referent=data_dict["audio_referent"], anchor=data_dict["audio_anchor"],
                relation=data_dict["audio_relation"],
                anchor_presence=data_dict["audio_anchor_presence"])

            confidence = data_dict["cluster_ref"] + self.anchor.gain * pair_logit
            data_dict["cluster_ref"] = confidence.masked_fill(~mask_flat, -1e4)
            data_dict["anchor_posterior"] = anchor_post
            data_dict["anchor_null_mass"] = null_mass
            return data_dict

    # ----------------------------------------------------------------------------------------
    #  Loss: role separation over every pair of roles
    # ----------------------------------------------------------------------------------------

    def compute_role_separation_loss(data_dict):
        """Penalise overlap between every pair of role elections.

        With two roles this is exactly the repository's referent/relation term; with the
        anchor role it also keeps the anchor off the other two roles' slots.
        """
        from lib.audio_loss_helper import _valid_utterance_mask
        attn = data_dict.get("audio_role_attn")
        if attn is None or attn.shape[1] < 2:
            return data_dict["audio_slots"].sum() * 0.0
        valid = _valid_utterance_mask(data_dict)
        if valid.sum() == 0:
            return attn.sum() * 0.0
        attn = attn[valid]
        total = attn.sum() * 0.0
        for first in range(attn.shape[1]):
            for second in range(first + 1, attn.shape[1]):
                total = total + (attn[:, first] * attn[:, second]).sum(-1).mean()
        return total

    # ----------------------------------------------------------------------------------------
    #  Installation
    # ----------------------------------------------------------------------------------------

    def install():
        """Swap the repository's audio modules for the SPARC ones. Call BEFORE building the model.

        Patches three names: models.refnet_audio.AudioModule, models.refnet_audio.AudioMatchModule
        and lib.audio_loss_helper.compute_role_separation_loss. With every switch off the
        resulting model is SPOT (the extra modules are not created).
        """
        import lib.audio_loss_helper as loss_helper
        import models.refnet_audio as refnet
        ensure_config()
        refnet.AudioModule = SparcAudioModule
        refnet.AudioMatchModule = SparcAudioMatchModule
        loss_helper.compute_role_separation_loss = compute_role_separation_loss
        on = [k for k in ("PROSODIC_CLOCK", "ORDERED_SLOTS", "LATENT_ANCHOR", "ANCHOR_NULL",
                          "ANCHOR_HARD", "PAIR_GEOMETRY") if AUDIO[k]]
        print(f"[SPARC] installed; NUM_ROLES={AUDIO.NUM_ROLES}; switches on: {', '.join(on)}")


# ============================================================================================
#  Pure-Python reference checks (no torch needed)
# ============================================================================================

def reference_checks():
    """Arithmetic the torch code relies on, checked with the standard library only."""
    sp = lambda x: math.log1p(math.exp(x))
    sig = lambda x: 1 / (1 + math.exp(-x))
    lse = lambda xs: max(xs) + math.log(sum(math.exp(x - max(xs)) for x in xs))
    random.seed(0)

    k = 6
    step = (0.85 - 0.15) / (k - 1)
    gaps = [0.15] + [step] * (k - 1)
    mu, run = [], 0.0
    for g in gaps:
        run += sp(math.log(math.expm1(g)))
        mu.append(run)
    assert all(abs(m - (0.15 + step * i)) < 1e-9 for i, m in enumerate(mu)), mu
    print("ordered centres start at linspace(0.15, 0.85):", [round(m, 3) for m in mu])

    def clock(rate):
        total, out, run = sum(rate), [], 0.0
        for r in rate:
            out.append(run / total)
            run += r
        return out

    t = 20
    assert max(abs(a - b / t) for b, a in enumerate(clock([1.0] * t))) < 1e-12
    warped = clock([0.25 + sig(random.gauss(0, 1)) for _ in range(t)])
    assert all(b >= a for a, b in zip(warped, warped[1:])) and warped[-1] < 1
    print("constant-rate clock equals linear time; a warped clock is monotone and below 1")

    m = 5
    u = [random.gauss(0, 1) for _ in range(m)]
    v = [random.gauss(0, 1) for _ in range(m)]
    g = [[random.gauss(0, 1) for _ in range(m)] for _ in range(m)]
    null = 0.3
    ell = [u[i] + lse([v[j] + g[i][j] for j in range(m) if j != i] + [null]) for i in range(m)]
    z = lse(ell)
    target = [math.exp(e - z) for e in ell]
    joint = [0.0] * (m + 1)
    for i in range(m):
        terms = [v[j] + g[i][j] if j != i else -1e4 for j in range(m)] + [null]
        zz = lse(terms)
        for j in range(m + 1):
            joint[j] += target[i] * math.exp(terms[j] - zz)
    assert abs(sum(joint) - 1) < 1e-9 and abs(sum(target) - 1) < 1e-9
    print("anchor posterior plus null mass sums to 1:", round(sum(joint), 9))
    print("all reference checks passed")


def print_arms():
    for name, settings in SPARC_ARMS.items():
        env = " ".join(f"VGA_{k}={v}" for k, v in settings.items())
        print(f"{name:18s} {env}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "arms":
        print_arms()
    elif command == "check":
        reference_checks()
    else:
        raise SystemExit("usage: python NEW-SPARC.py [check|arms]")
