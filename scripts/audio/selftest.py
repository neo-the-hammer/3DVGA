"""
Shape and gradient self-test for the 3D-VGA branch. Needs torch; needs no data.

    python scripts/audio/selftest.py
    python scripts/audio/selftest.py --fusion xattn --slots 4 --device cuda

Builds synthetic proposals, audio frames and labels, runs AudioModule + AudioMatchModule +
the loss, and checks that every tensor has the shape the next stage expects and that
gradients reach the parts that are supposed to learn.

This exists because the expensive failure mode on this pipeline is not a crash -- it is a
silent one. A slot bank that collapses, a transport plan that is uniform, a fusion head
whose gradient never reaches the audio encoder: all of those train to completion and report
a number. The checks below are the cheap versions of those questions, run in two seconds
instead of two days.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

# lib/config.py lists the ScanNet scene directory at import time. An empty directory is
# enough for it, and this keeps the self-test runnable on a machine with no dataset.
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "scannet", "scans"), exist_ok=True)

import torch

from lib.audio_config import AUDIO, PROSODY_GROUPS


class Checks:
    def __init__(self):
        self.failures = []
        self.passed = 0

    def __call__(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  ok    {name}")
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"  FAIL  {name}  {detail}")

    def shape(self, name, tensor, expected):
        self(name, tuple(tensor.shape) == tuple(expected),
             f"got {tuple(tensor.shape)}, expected {tuple(expected)}")

    def finite(self, name, tensor):
        self(f"{name} is finite", bool(torch.isfinite(tensor).all()),
             "contains NaN or Inf")


def fake_batch(batch, lang, frames, feat_dim, proposals, device, num_class=18):
    """A data_dict with everything the audio branch reads, and nothing it does not."""
    lengths = torch.randint(frames // 2, frames + 1, (batch, lang), device=device)
    centers = torch.randn(batch, proposals, 3, device=device) * 2.0

    objectness = torch.zeros(batch, proposals, 2, device=device)
    keep = torch.rand(batch, proposals, device=device) > 0.3
    objectness[..., 1] = keep.float()
    objectness[..., 0] = 1.0 - keep.float()

    return {
        "audio_feat_list": torch.randn(batch, lang, frames, feat_dim, device=device),
        "prosody_list": torch.randn(batch, lang, frames, AUDIO.PROSODY_DIM, device=device),
        "audio_len_list": lengths,
        "audio_cat_list": torch.randint(0, num_class, (batch, lang), device=device),
        "features": torch.randn(batch, 288, proposals, device=device),
        "center": centers,
        "objectness_scores": objectness,
        "istrain": torch.ones(batch, device=device),
        "lang_num": torch.full((batch,), lang, dtype=torch.long, device=device),
    }


def run(args):
    device = torch.device(args.device)
    torch.manual_seed(0)

    AUDIO.FUSION = args.fusion
    AUDIO.NUM_SLOTS = args.slots
    AUDIO.USE_RSK = not args.no_rsk
    AUDIO.USE_PROSODY = not args.no_prosody
    AUDIO.OT_ROUTING = not args.no_ot
    AUDIO.BIDIRECTIONAL_SLOTS = not args.no_bidirectional
    AUDIO.ROUTE_GATE = not args.no_route_gate
    AUDIO.FRAME_RESIDUAL = not args.no_frame_residual
    AUDIO.ROLE_ELECTION = not args.no_role_election
    AUDIO.GRAPH_K = args.graph_k
    AUDIO.PROSODY_CHANNELS = args.prosody_channels
    AUDIO.SLOT_TIME_PRIOR = not args.no_time_prior
    AUDIO.DISABLE_COPY_PASTE = args.disable_copy_paste

    from lib.audio_loss_helper import compute_audio_classification_loss, compute_slot_regularisers
    from models.audio_match_module import AudioMatchModule
    from models.audio_module import AudioModule

    batch, lang, frames = args.batch, args.lang, args.frames
    feat_dim, proposals = args.feat_dim, args.proposals
    flat = batch * lang
    hidden, slots = AUDIO.HIDDEN_SIZE, AUDIO.NUM_SLOTS

    print(f"\n3D-VGA self-test  fusion={AUDIO.FUSION} slots={slots} rsk={AUDIO.USE_RSK} "
          f"prosody={AUDIO.USE_PROSODY}/{AUDIO.PROSODY_CHANNELS} device={device}")
    print(f"ablation switches  ot={AUDIO.OT_ROUTING} bidir={AUDIO.BIDIRECTIONAL_SLOTS} "
          f"gate={AUDIO.ROUTE_GATE} frames={AUDIO.FRAME_RESIDUAL} "
          f"roles={AUDIO.ROLE_ELECTION} graph_k={AUDIO.GRAPH_K}")
    print(f"batch={batch} lang={lang} frames={frames} feat_dim={feat_dim} "
          f"proposals={proposals}\n")

    check = Checks()
    data = fake_batch(batch, lang, frames, feat_dim, proposals, device)

    class Args:
        detector = "GF"

    audio = AudioModule(num_text_classes=18, in_dim=feat_dim).to(device).train()
    match = AudioMatchModule(Args(), num_proposals=proposals, det_channel=288).to(device).train()

    print("AudioModule")
    data = audio(data)
    check.shape("audio_fea", data["audio_fea"], (flat, frames, hidden))
    check.shape("audio_slots", data["audio_slots"], (flat, slots, hidden))
    check.shape("audio_slot_plan", data["audio_slot_plan"], (flat, slots, frames))
    check.shape("audio_slot_mass", data["audio_slot_mass"], (flat, slots))
    check.shape("audio_emb", data["audio_emb"], (flat, hidden))
    check.shape("audio_role_attn", data["audio_role_attn"], (flat, AUDIO.NUM_ROLES, slots))
    check.shape("audio_referent", data["audio_referent"], (flat, hidden))
    check.shape("audio_relation", data["audio_relation"], (flat, hidden))
    role = data["audio_role_attn"]
    check("role election normalises over slots",
          bool((role.sum(-1) - 1.0).abs().max() < 1e-4),
          f"max error {(role.sum(-1) - 1.0).abs().max().item():.2e}")
    if not AUDIO.ROLE_ELECTION:
        # The fixed-index arm must still hand downstream code a valid distribution, or the
        # arm differs from the proposed one in more than the thing it claims to ablate.
        check("fixed-role fallback is one-hot",
              bool(((role == 0) | (role == 1)).all() and role.sum(-1).eq(1).all()),
              "fixed role attention is not one-hot")
    check.shape("audio_scores", data["audio_scores"], (flat, 18))
    check.shape("audio_attention_mask", data["audio_attention_mask"], (flat, 1, 1, frames))
    check.finite("audio_slots", data["audio_slots"])
    check.finite("audio_slot_plan", data["audio_slot_plan"])

    plan = data["audio_slot_plan"]
    mask = data["audio_frame_mask"]
    check("frame->slot plan carries no mass on padding",
          bool((plan * (~mask).unsqueeze(1).float()).abs().max() < 1e-6))

    # Sinkhorn is only an assignment if it converged. The column marginal is exact by
    # construction (the last update is the column one); the row marginal is the one that
    # can silently drift, and the "slot mass is a budget" reading of the plan depends on
    # it. AUDIO.SLOT_SINKHORN_EPS/ITERS were chosen against this threshold.
    total_err = (plan.sum(dim=(1, 2)) - 1.0).abs().max()
    check("frame->slot plan sums to 1 per utterance", bool(total_err < 1e-3),
          f"max total-mass error {total_err.item():.2e}")

    col_mass = plan.sum(dim=1)[mask]
    expected = (1.0 / mask.sum(dim=1, keepdim=True).float()).expand_as(mask)[mask]
    col_err = ((col_mass - expected).abs() / expected).max()
    check("frame->slot column marginals converged", bool(col_err < 0.05),
          f"max relative column error {col_err.item():.3f}")
    # Sinkhorn's whole purpose: slots must not all land on the same frames. Compare the
    # plan's per-slot temporal centroids -- identical centroids means collapse.
    rows = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    position = torch.arange(frames, device=device).float()
    centroid = (rows * position).sum(dim=-1)                              # (flat, slots)
    spread = (centroid.max(dim=-1)[0] - centroid.min(dim=-1)[0]).mean()
    if AUDIO.SLOT_TIME_PRIOR:
        check("slots occupy distinct time spans", bool(spread > 1.0),
              f"mean centroid spread {spread.item():.2f} frames (want > 1)")
    else:
        # At initialisation the time prior is the only thing that separates the slots;
        # without it separation has to be learned, so an untrained model cannot show it.
        print(f"  --    slot spread at init, no time prior: {spread.item():.2f} frames")

    print("\nAudioMatchModule")
    data = match(data)
    check.shape("cluster_ref", data["cluster_ref"], (flat, proposals))
    check.shape("fused_proposals", data["fused_proposals"], (flat, proposals, hidden))
    check.finite("cluster_ref", data["cluster_ref"])

    obj_mask = data["objectness_scores"].max(2)[1].bool()
    flat_mask = obj_mask.unsqueeze(1).expand(-1, lang, -1).reshape(flat, proposals)
    check("non-objects cannot be selected",
          bool((data["cluster_ref"][~flat_mask] <= -1e3).all()))

    if AUDIO.FUSION == "spot":
        check.shape("ot_plan", data["ot_plan"], (flat, slots, proposals))
        check.finite("ot_plan", data["ot_plan"])
        plan_fuse = data["ot_plan"]
        check("slot->proposal plan carries no mass on non-objects",
              bool((plan_fuse * (~flat_mask).unsqueeze(1).float()).abs().max() < 1e-6))
        # Both routing modes must produce a plan whose rows are the slot budget and whose
        # total mass is 1. That is what lets spot-softmax be a one-variable ablation: every
        # downstream consumer sees the same kind of object either way.
        total = (plan_fuse.sum(dim=(1, 2)) - 1.0).abs().max()
        check(f"plan total mass is 1 (ot={AUDIO.OT_ROUTING})", bool(total < 1e-2),
              f"max total-mass error {total.item():.2e}")
        rows_err = (plan_fuse.sum(dim=-1)
                    - data["audio_slot_mass"]
                    / data["audio_slot_mass"].sum(-1, keepdim=True)).abs().max()
        check("plan row marginals equal the slot budget", bool(rows_err < 5e-2),
              f"max row error {rows_err.item():.2e}")
        weights = data["audio_role_attn"][:, AUDIO.ROLE_REFERENT]
        target_row = torch.einsum("bk,bkm->bm", weights, data["ot_plan"])
        target_row = target_row / target_row.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        peak = target_row.max(dim=-1)[0].mean()
        uniform = 1.0 / float(flat_mask.sum(dim=-1).float().mean())
        check("OT concentrates mass above uniform", bool(peak > uniform),
              f"peak {peak.item():.4f} vs uniform {uniform:.4f}")

    print("\nlosses and gradients")
    audio_cls = compute_audio_classification_loss(data)
    entropy, ortho = compute_slot_regularisers(data)
    loss = data["cluster_ref"].logsumexp(dim=-1).mean() + audio_cls + ortho - entropy
    check.finite("loss", loss)
    loss.backward()

    def grad_reaches(name, module):
        total = sum(p.grad.abs().sum().item()
                    for p in module.parameters() if p.grad is not None)
        check(f"gradient reaches {name}", total > 0, "all gradients are zero or None")

    grad_reaches("audio encoder", audio.encoder)
    grad_reaches("slot induction", audio.slots)
    # The role election replaced hardcoded slot indices once the text teacher was removed.
    # If no gradient reaches it, the referent and relation vectors are a fixed random
    # mixture of slots and every claim resting on them is empty.
    if AUDIO.ROLE_ELECTION:
        check("gradient reaches the role election",
              audio.role_queries.grad is not None
              and float(audio.role_queries.grad.abs().sum()) > 0,
              "role_queries has no gradient")
    else:
        # The fixed-index arm reads the roles off slots 0 and 1, so the queries are unused
        # by construction and a gradient on them would mean the arm is not what it says.
        check("role election is bypassed",
              audio.role_queries.grad is None
              or float(audio.role_queries.grad.abs().sum()) == 0,
              "role_queries received a gradient in the fixed-role arm")
    grad_reaches("fusion head", match.fusion)
    grad_reaches("scoring stack", match.match)
    if AUDIO.USE_RSK and AUDIO.FUSION == "spot":
        grad_reaches("RSK hypernetwork", match.fusion.rsk)
    if AUDIO.GRAPH_K > 0:
        grad_reaches("spatial graph", match.graph)
    else:
        check("graph is absent at graph_k=0", match.graph is None,
              "SpatialGraphNet was built despite GRAPH_K=0")

    print("\neval mode (no slot dropout, no copy-paste)")
    audio.eval()
    match.eval()
    with torch.no_grad():
        clean = fake_batch(batch, lang, frames, feat_dim, proposals, device)
        clean["istrain"] = torch.zeros(batch, device=device)
        first = match(audio(dict(clean)))["cluster_ref"]
        second = match(audio(dict(clean)))["cluster_ref"]
    check.finite("cluster_ref (eval)", first)
    # Dropout, slot dropout and copy-paste must all be off in eval, or two passes over the
    # same scene disagree and the reported accuracy depends on the seed.
    check("eval is deterministic", bool(torch.allclose(first, second, atol=1e-5)),
          f"max difference {(first - second).abs().max().item():.2e}")

    params = sum(p.numel() for p in audio.parameters()) + \
        sum(p.numel() for p in match.parameters())
    print(f"\naudio branch + fusion head: {params / 1e6:.2f}M parameters")

    print(f"\n{check.passed} check(s) passed, {len(check.failures)} failed")
    for failure in check.failures:
        print(f"  - {failure}")
    return 1 if check.failures else 0


ARM_FLAGS = {
    "spot":            {},
    "spot-softmax":    {"no_ot": True},
    "spot-unidir":     {"no_bidirectional": True},
    "spot-fixedroles": {"no_role_election": True},
    "spot-norsk":      {"no_rsk": True},
    "spot-nograph":    {"graph_k": 0},
    "spot-nogate":     {"no_route_gate": True},
    "spot-noframes":   {"no_frame_residual": True},
    "spot-noprosody":  {"no_prosody": True},
    "spot-nocopypaste": {"disable_copy_paste": True},
    "spot-notimeprior": {"no_time_prior": True},
    # SpecAugment lives in the dataset, so this arm is the proposed model here.
    "spot-nospecaug":  {},
    "spot-k2":         {"slots": 2},
    "xattn":           {"fusion": "xattn"},
    "concat":          {"fusion": "concat"},
}


def run_all_arms(args):
    """Build and backprop every ablation arm once. Nothing else checks that they run."""
    import copy
    failed = []
    for arm, overrides in ARM_FLAGS.items():
        local = copy.deepcopy(args)
        local.all_arms = False
        for key, value in overrides.items():
            setattr(local, key, value)
        print("\n" + "#" * 78)
        print(f"# arm: {arm}")
        print("#" * 78)
        # Each arm gets a fresh AUDIO state; run() writes into it.
        for key, value in _DEFAULTS.items():
            AUDIO[key] = value
        if run(local) != 0:
            failed.append(arm)
    print("\n" + "=" * 78)
    print(f"{len(ARM_FLAGS) - len(failed)}/{len(ARM_FLAGS)} arm(s) passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    print("=" * 78)
    return 1 if failed else 0


#: AUDIO keys run() overwrites, captured before any arm mutates them.
_DEFAULTS = {k: AUDIO[k] for k in
             ("FUSION", "NUM_SLOTS", "USE_RSK", "USE_PROSODY", "OT_ROUTING",
              "BIDIRECTIONAL_SLOTS", "ROUTE_GATE", "FRAME_RESIDUAL", "ROLE_ELECTION",
              "GRAPH_K", "PROSODY_CHANNELS", "SLOT_TIME_PRIOR", "DISABLE_COPY_PASTE")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fusion", default=AUDIO.FUSION, choices=["spot", "xattn", "concat"])
    parser.add_argument("--slots", type=int, default=AUDIO.NUM_SLOTS)
    parser.add_argument("--no_rsk", action="store_true")
    parser.add_argument("--no_prosody", action="store_true")
    parser.add_argument("--no_ot", action="store_true",
                        help="Row-softmax instead of Sinkhorn (the spot-softmax arm).")
    parser.add_argument("--no_bidirectional", action="store_true")
    parser.add_argument("--no_route_gate", action="store_true")
    parser.add_argument("--no_frame_residual", action="store_true")
    parser.add_argument("--no_role_election", action="store_true")
    parser.add_argument("--no_time_prior", action="store_true")
    parser.add_argument("--disable_copy_paste", action="store_true")
    parser.add_argument("--graph_k", type=int, default=AUDIO.GRAPH_K)
    parser.add_argument("--prosody_channels", type=str, default=AUDIO.PROSODY_CHANNELS,
                        choices=list(PROSODY_GROUPS))
    parser.add_argument("--all_arms", action="store_true",
                        help="Run the self-test once per ablation arm. This is the check "
                             "that every arm in experiments/audio/fusion_arms.py actually "
                             "builds and trains, which is cheaper to find out here than "
                             "six hours into a sweep.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lang", type=int, default=3)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--proposals", type=int, default=64)
    _args = parser.parse_args()
    sys.exit(run_all_arms(_args) if _args.all_arms else run(_args))
