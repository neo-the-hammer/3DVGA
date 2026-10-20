"""
CPU-only analyses over finished 3D-VGA runs. Needs no GPU and no dataset.

    python experiments/audio/analysis.py results-table   --folders RUN_A RUN_B ...
    python experiments/audio/analysis.py slot-roles      --folder RUN
    python experiments/audio/analysis.py ot-sharpness    --folders SPOT XATTN
    python experiments/audio/analysis.py gain-attribution --folders BASE PROPOSED
    python experiments/audio/analysis.py error-taxonomy  --folder RUN
    python experiments/audio/analysis.py seed-aggregate  --folders SEED42 SEED1 ...

Mirrors experiments/analysis.py on the text branch: everything here reads artifacts that
scripts/audio_eval.py already wrote, so re-running an analysis costs seconds rather than a
GPU-hour, and the numbers in a paper can be regenerated from the saved run.

``scores.p`` is written by every eval. ``diagnostics.p`` needs ``--dump_diagnostics``, and
carries the per-utterance slot and transport statistics that the middle four commands read.
"""

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from lib.audio_config import AUDIO  # noqa: E402
from lib.config import CONF  # noqa: E402

#: Context rows for results-table. AP-Refer is the original benchmark; Audio-3DVG and the
#: ASR cascade are the current bar (docs/3DVGA.md section 1b). None is a controlled
#: comparison against this repository's detector -- the `concat` arm is.
REFERENCE_ROWS = [
    ("AP-Refer (Neurocomputing'25)", (48.62, 29.59), (16.94, 9.96), (23.09, 13.77)),
    ("Audio-3DVG (arXiv'25)", (78.26, 67.35), (30.82, 24.46), (40.02, 32.78)),
    ("3DVG-Trans, ASR cascade", (74.92, 56.67), (35.43, 26.92), (43.23, 33.87)),
]


# --------------------------------------------------------------------------- loading


def _load(folder, name):
    path = os.path.join(CONF.PATH.OUTPUT, folder, name)
    if not os.path.isfile(path):
        raise SystemExit(
            f"missing {path}\n"
            f"  run: python scripts/audio_eval.py --folder {folder} --force"
            + (" --dump_diagnostics" if name == "diagnostics.p" else ""))
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _np():
    try:
        import numpy as np
    except ImportError as exc:                                    # pragma: no cover
        raise SystemExit(f"these analyses need numpy (import failed: {exc})") from exc
    return np


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _sd(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _rates(ious, threshold):
    return 100.0 * sum(1 for v in ious if v >= threshold) / len(ious) if len(ious) else 0.0


# --------------------------------------------------------------------------- commands


def run_results_table(args):
    """The 3DVG-Audio table for several runs at once, against AP-Refer."""
    header = (f"{'run':28s} {'U@0.25':>8s} {'U@0.5':>8s} {'M@0.25':>8s} "
              f"{'M@0.5':>8s} {'O@0.25':>8s} {'O@0.5':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for name, u, m, o in REFERENCE_ROWS:
        print(f"{name[:28]:28s} {u[0]:8.2f} {u[1]:8.2f} {m[0]:8.2f} {m[1]:8.2f} "
              f"{o[0]:8.2f} {o[1]:8.2f}")
    print("-" * len(header))

    for folder in args.folders:
        report = _load(folder, "scores.p").get("report")
        if report is None:
            print(f"{folder[:28]:28s} {'-- no report in scores.p --':>54s}")
            continue
        u, m, o = report["unique"], report["multiple"], report["overall"]
        print(f"{folder[:28]:28s} {u[0]:8.2f} {u[1]:8.2f} {m[0]:8.2f} {m[1]:8.2f} "
              f"{o[0]:8.2f} {o[1]:8.2f}")

    print("\nThe reference rows are context, not controlled comparisons: each uses its own")
    print("detector and corpus, and Audio-3DVG additionally uses PointGroup instance")
    print("segmentation with class-filtered candidates, an easier protocol. For a")
    print("controlled number quote the `concat` arm of fusion_arms.py; for the comparison")
    print("that decides whether direct audio is worth doing, quote the ASR cascade")
    print("(experiments/audio/robustness.py asr-cascade), which currently leads.")
    return 0


def run_slot_roles(args):
    """Did the slots specialise, and which slot did each role elect?

    This is the experiment that tests the central mechanism claim, and it carries more
    weight in an audio-only model than it would otherwise: with no text teacher, nothing
    supervises the decomposition, so "the slots recover distinct roles" is an assertion
    about a latent variable that only this command can check.
    """
    np = _np()
    diag = _load(args.folder, "diagnostics.p")
    centroid = diag["slot_centroid"]                              # (N, K) in [0, 1]
    mass = diag["slot_mass"]                                      # (N, K)
    n_slots = centroid.shape[1]

    print(f"\nslot occupancy over {centroid.shape[0]} val utterances")
    print(f"{'slot':16s} {'mean pos':>9s} {'sd pos':>8s} {'mean mass':>10s} "
          f"{'sd mass':>8s} {'silent %':>9s}")
    print("-" * 64)
    for k in range(n_slots):
        silent = 100.0 * float((mass[:, k] < 0.05).mean())
        print(f"{'slot ' + str(k):16s} {centroid[:, k].mean():9.3f} "
              f"{centroid[:, k].std():8.3f} {mass[:, k].mean():10.3f} "
              f"{mass[:, k].std():8.3f} {silent:9.1f}")

    # Separation: if the slots partition the utterance, their temporal centroids are
    # ordered and spread. If they collapsed, every centroid sits on the same frame.
    order = np.argsort(centroid.mean(axis=0))
    spread = float((centroid.max(axis=1) - centroid.min(axis=1)).mean())
    print(f"\ncentroid order (early → late): {list(order)}")
    print(f"mean within-utterance spread : {spread:.3f} of the utterance")
    if spread < 0.10:
        print("  ⚠ slots are NOT separating — the decomposition claim does not hold here.")
    else:
        print("  slots occupy distinct spans, which is what the Sinkhorn column constraint")
        print("  is for. Compare against a --num_slots sweep point with the constraint off.")

    if "role_attn" in diag:
        role = diag["role_attn"]                                  # (N, R, K)
        names = {AUDIO.ROLE_REFERENT: "referent", AUDIO.ROLE_RELATION: "relation"}
        print(f"\nrole election — which slot each role pooled from")
        print(f"{'role':12s} {'modal slot':>11s} {'agreement':>10s} {'peak':>7s} "
              f"{'entropy':>8s}")
        print("-" * 52)
        for r in range(role.shape[1]):
            winner = role[:, r].argmax(axis=1)
            modal = int(np.bincount(winner, minlength=role.shape[2]).argmax())
            agree = float((winner == modal).mean())
            peak = float(role[:, r].max(axis=1).mean())
            ent = float(-(role[:, r] * np.log(np.clip(role[:, r], 1e-8, None))).sum(1).mean())
            print(f"{names.get(r, f'role {r}'):12s} {modal:11d} {agree:10.2f} "
                  f"{peak:7.3f} {ent:8.3f}")

        overlap = float((role[:, AUDIO.ROLE_REFERENT] * role[:, AUDIO.ROLE_RELATION]).sum(1).mean())
        print(f"\nreferent/relation overlap: {overlap:.3f}   (0 = disjoint, 1 = same slot)")
        if overlap > 0.5:
            print("  ⚠ the two roles are electing the SAME slot. RSK is then reading the")
            print("    referent vector, and the geometric kernel is a second copy of the")
            print("    target pathway rather than a relation-conditioned prior.")
            print("    Raise VGA_LOSS_ROLE_SEPARATION (sweeps.py role_separation).")
        print("\nAgreement below ~0.6 means the election is unstable across utterances:")
        print("the roles are not resting on a consistent decomposition, and an accuracy")
        print("gain is coming from somewhere other than the mechanism claimed.")
    else:
        print("\n(no role_attn in diagnostics.p — re-run the eval to get the election)")

    if args.by_outcome:
        iou = diag["iou"]
        hit, miss = iou >= 0.25, iou < 0.25
        print(f"\n{'slot':16s} {'mass | hit':>12s} {'mass | miss':>12s} {'Δ':>8s}")
        print("-" * 52)
        for k in range(n_slots):
            a, b = float(mass[hit, k].mean()), float(mass[miss, k].mean())
            print(f"{'slot ' + str(k):16s}{a:12.3f} {b:12.3f} {a - b:8.3f}")
        print("\nA slot whose mass differs between hits and misses is one the model is")
        print("actually using. A slot with no difference is dead weight — evidence for a")
        print("smaller K.")
    return 0


def run_ot_sharpness(args):
    """Does optimal transport concentrate the grounding score more than attention does?

    The direct measurement of Figure 14's claim. Attention arms have no transport plan, so
    they appear here only as the reference the SPOT arms are compared against.
    """
    np = _np()
    print(f"\n{'run':28s} {'OT peak':>9s} {'margin':>9s} {'plan H':>9s} "
          f"{'Acc@0.25':>9s} {'Acc@0.5':>9s}")
    print("-" * 78)

    for folder in args.folders:
        diag = _load(folder, "diagnostics.p")
        iou = diag["iou"]
        if "ot_peak" not in diag:
            print(f"{folder[:28]:28s} {'— no transport plan (attention arm) —':>38s} "
                  f"{_rates(iou, 0.25):9.2f} {_rates(iou, 0.5):9.2f}")
            continue
        print(f"{folder[:28]:28s} {diag['ot_peak'].mean():9.4f} "
              f"{diag['ot_margin'].mean():9.4f} {diag['ot_entropy'].mean():9.3f} "
              f"{_rates(iou, 0.25):9.2f} {_rates(iou, 0.5):9.2f}")

    print("\npeak   = the target slot's largest transport mass over proposals")
    print("margin = best minus second-best. THIS is the same-class tie-break, so it is the")
    print("         number Figure 14 is about — a larger margin on the Multiple subset is")
    print("         the mechanism working, not just the accuracy moving.")

    for folder in args.folders:
        diag = _load(folder, "diagnostics.p")
        if "ot_margin" not in diag:
            continue
        um = diag["unique_multiple"]
        print(f"\n{folder}")
        for label, sel in (("unique", um == 0), ("multiple", um == 1)):
            if sel.sum() == 0:
                continue
            hit = diag["iou"][sel] >= 0.25
            margin = diag["ot_margin"][sel]
            print(f"  {label:9s} margin {margin.mean():.4f}   "
                  f"hit {margin[hit].mean():.4f} vs miss {margin[~hit].mean():.4f}   "
                  f"n={int(sel.sum())}")
        print("  If margin separates hits from misses, the transport plan is carrying the")
        print("  decision rather than decorating a score the MLP already made.")
    return 0


def run_gain_attribution(args):
    """Where a gain actually comes from: subset, scene density, and class correctness.

    A headline improvement that turns out to be concentrated in easy scenes is a different
    result from one concentrated in crowded ones, and the Overall column cannot tell them
    apart.
    """
    np = _np()
    if len(args.folders) != 2:
        raise SystemExit("gain-attribution compares exactly two runs: baseline then proposed")
    base_name, prop_name = args.folders
    base, prop = _load(base_name, "diagnostics.p"), _load(prop_name, "diagnostics.p")

    if base["iou"].shape != prop["iou"].shape:
        raise SystemExit(
            f"the two runs scored different numbers of utterances "
            f"({base['iou'].shape[0]} vs {prop['iou'].shape[0]}). They must be evaluated on "
            f"the same split with the same --audio_lang_num_max for a paired comparison.")

    print(f"\nbaseline : {base_name}\nproposed : {prop_name}\n")

    um = base["unique_multiple"]
    n_obj = base["n_objects"]
    # Scene density is the axis the Multiple subset conflates: "several chairs" and "one of
    # forty objects" are both `multiple`, and they are not the same problem.
    bands = [(0, 16, "sparse ≤16"), (16, 32, "16–32"), (32, 64, "32–64"), (64, 10 ** 6, ">64")]

    print(f"{'slice':18s} {'n':>7s} {'base@.25':>9s} {'prop@.25':>9s} {'Δ':>7s} "
          f"{'base@.5':>9s} {'prop@.5':>9s} {'Δ':>7s}")
    print("-" * 80)

    def row(label, sel):
        if sel.sum() == 0:
            return
        b25, p25 = _rates(base["iou"][sel], 0.25), _rates(prop["iou"][sel], 0.25)
        b50, p50 = _rates(base["iou"][sel], 0.5), _rates(prop["iou"][sel], 0.5)
        print(f"{label:18s} {int(sel.sum()):7d} {b25:9.2f} {p25:9.2f} {p25 - b25:+7.2f} "
              f"{b50:9.2f} {p50:9.2f} {p50 - b50:+7.2f}")

    row("overall", np.ones_like(um, dtype=bool))
    row("unique", um == 0)
    row("multiple", um == 1)
    print("-" * 80)
    for lo, hi, label in bands:
        row(label, (n_obj >= lo) & (n_obj < hi))
    print("-" * 80)

    right = base["audio_pred"] == base["audio_true"]
    row("AudioCls correct", right)
    row("AudioCls wrong", ~right)

    flips = int(((base["iou"] < 0.25) & (prop["iou"] >= 0.25)).sum())
    breaks = int(((base["iou"] >= 0.25) & (prop["iou"] < 0.25)).sum())
    print(f"\nfixed by the proposed model : {flips}")
    print(f"broken by it                : {breaks}")
    print(f"net                         : {flips - breaks:+d}")
    print("\nA large fixed AND a large broken count with a small net is a model that moved")
    print("sideways — report both, not the net alone.")
    return 0


def run_error_taxonomy(args):
    """Split failures into recognition vs grounding, which need different fixes."""
    np = _np()
    diag = _load(args.folder, "diagnostics.p")
    iou, um = diag["iou"], diag["unique_multiple"]
    class_ok = diag["audio_pred"] == diag["audio_true"]
    hit = iou >= 0.25

    categories = [
        ("correct", hit, "the referent was localised"),
        ("grounding failure", ~hit & class_ok,
         "the class was heard correctly, the wrong instance was picked"),
        ("recognition failure", ~hit & ~class_ok,
         "the object class itself was misheard — the acoustic branch, not the fusion"),
    ]

    print(f"\nfailure taxonomy over {len(iou)} utterances — {args.folder}")
    print(f"\n{'category':22s} {'all %':>8s} {'unique %':>9s} {'multiple %':>11s}")
    print("-" * 54)
    for name, sel, _ in categories:
        allp = 100.0 * float(sel.mean())
        up = 100.0 * float(sel[um == 0].mean()) if (um == 0).sum() else 0.0
        mp = 100.0 * float(sel[um == 1].mean()) if (um == 1).sum() else 0.0
        print(f"{name:22s} {allp:8.1f} {up:9.1f} {mp:11.1f}")
    print()
    for name, _, why in categories:
        print(f"  {name:22s} {why}")

    print("\nThe split matters because the two failures have different remedies:")
    print("  recognition → a better encoder, more prosody, cleaner audio, L_AudioCls")
    print("  grounding   → the fusion operator, the geometric prior, the slot budget")
    print("A model whose losses are mostly RECOGNITION has a fusion problem it does not")
    print("have, and vice versa — which is exactly the mistake a single Overall number")
    print("invites.")
    return 0


def run_seed_aggregate(args):
    """Mean ± sd over seeds, and the size a claimed gain has to clear."""
    rows = []
    for folder in args.folders:
        report = _load(folder, "scores.p").get("report")
        if report is None:
            print(f"  {folder}: no report, skipped")
            continue
        rows.append((folder, report))

    if len(rows) < 2:
        raise SystemExit("seed-aggregate needs at least two runs")

    print(f"\n{'run':28s} {'U@0.25':>8s} {'U@0.5':>8s} {'M@0.25':>8s} "
          f"{'M@0.5':>8s} {'O@0.25':>8s} {'O@0.5':>8s}")
    print("-" * 82)
    for folder, report in rows:
        u, m, o = report["unique"], report["multiple"], report["overall"]
        print(f"{folder[:28]:28s} {u[0]:8.2f} {u[1]:8.2f} {m[0]:8.2f} {m[1]:8.2f} "
              f"{o[0]:8.2f} {o[1]:8.2f}")
    print("-" * 82)

    columns = [("unique", 0), ("unique", 1), ("multiple", 0), ("multiple", 1),
               ("overall", 0), ("overall", 1)]
    means = [_mean(r[key][i] for _, r in rows) for key, i in columns]
    sds = [_sd(r[key][i] for _, r in rows) for key, i in columns]
    print("mean".ljust(28) + "".join(f"{v:8.2f}" for v in means))
    print("sd".ljust(28) + "".join(f"{'±' + format(v, '.2f'):>8s}" for v in sds))

    print(f"\nn = {len(rows)} seeds.")
    print(f"A gain on Overall@0.25 has to clear roughly {2 * sds[4]:.2f} points "
          f"(2 sd) before it is")
    print("distinguishable from run-to-run noise. Report every ablation gap against this.")
    return 0



def run_significance(args):
    """Paired permutation test between two runs over the same utterances.

    An ablation table of point estimates invites the reader to treat a 0.4-point gap as a
    result. Accuracy here is a mean over ~9.5k paired Bernoulli outcomes, so whether a gap
    is real is a question with an answer, and this is the cheapest correct way to get it:
    the two systems are evaluated on the same utterances in the same order, so the pairing
    is exact and a permutation test needs no distributional assumption at all.

    Under the null the two systems are interchangeable, so flipping a random subset of the
    pairs leaves the difference unchanged in distribution. p is the fraction of flips that
    reach the observed difference.
    """
    np = _np()
    a = _load(args.folders[0], "scores.p")
    b = _load(args.folders[1], "scores.p")
    ious_a, ious_b = np.asarray(a["ious"]), np.asarray(b["ious"])
    if ious_a.shape != ious_b.shape:
        raise SystemExit(
            f"the two runs scored different numbers of utterances "
            f"({ious_a.shape[0]} vs {ious_b.shape[0]}). A paired test needs the same "
            f"evaluation set in the same order -- re-run audio_eval.py for both with the "
            f"same --lang_num_max and without --repeat.")
    masks = np.asarray(a["masks"])

    rng = np.random.RandomState(args.seed)
    print(f"\n  A = {args.folders[0]}\n  B = {args.folders[1]}")
    print(f"  {args.iters} permutations, paired over {ious_a.shape[0]} utterances\n")
    header = (f"{'subset':12s} {'n':>7s} {'A':>8s} {'B':>8s} {'B-A':>8s} "
              f"{'95% CI':>17s} {'p':>9s}")
    print(header); print("-" * len(header))

    for name, sel in (("Unique", masks == 0), ("Multiple", masks == 1),
                      ("Overall", np.ones_like(masks, dtype=bool))):
        for thr in (0.25, 0.5):
            hit_a = (ious_a[sel] >= thr).astype(np.float64)
            hit_b = (ious_b[sel] >= thr).astype(np.float64)
            n = hit_a.shape[0]
            if n == 0:
                continue
            diff = hit_b.mean() - hit_a.mean()

            # Permutation: swap each pair with probability 1/2.
            d = hit_b - hit_a
            signs = rng.randint(0, 2, size=(args.iters, n)) * 2 - 1
            null = (signs * d).mean(axis=1)
            p = float((np.abs(null) >= abs(diff) - 1e-12).mean())

            # Paired bootstrap for the interval, resampling utterances not systems.
            idx = rng.randint(0, n, size=(args.iters, n))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])

            print(f"{name + '@' + str(thr):12s} {n:7d} {hit_a.mean() * 100:8.2f} "
                  f"{hit_b.mean() * 100:8.2f} {diff * 100:+8.2f} "
                  f"{'[' + format(lo * 100, '+.2f') + ', ' + format(hi * 100, '+.2f') + ']':>17s} "
                  f"{p:9.4f}")

    print("\np is two-sided. A confidence interval that straddles zero means the two "
          "systems are\nnot distinguishable on that subset at this sample size, whatever "
          "the point estimates say.")
    print("This is a test against sampling noise over utterances only. It does NOT cover "
          "seed\nvariance -- for that, run several seeds and use seed-aggregate; report "
          "both.\n")
    return 0


def run_complexity(args):
    """Accuracy against utterance duration and against scene clutter.

    The audio counterpart of the predecessor paper's "effect of linguistic complexity on
    grounding". A longer clip is a longer description: more attributes, more relations,
    more chances for the decomposition to matter -- and more frames for a mean-pooling
    baseline to wash out. If the proposed operator earns its place, its advantage should
    *grow* with duration and with the number of same-class distractors, and a flat curve
    would mean the gain comes from somewhere other than the stated mechanism.
    """
    np = _np()
    diag = _load(args.folder, "diagnostics.p")
    if "audio_len" not in diag:
        raise SystemExit(
            "this diagnostics.p predates the audio_len field; re-run audio_eval.py "
            "--dump_diagnostics")
    iou = diag["iou"]
    length = diag["audio_len"].astype(float)
    fps = 16000.0 / (320.0 * max(1, AUDIO.FRAME_POOL))
    seconds = length / fps

    ref = None
    if args.baseline:
        other = _load(args.baseline, "diagnostics.p")
        if other["iou"].shape == iou.shape:
            ref = other["iou"]
        else:
            print(f"  baseline has {other['iou'].shape[0]} utterances vs "
                  f"{iou.shape[0]}; skipping the paired column\n")

    def report(title, values, edges, unit):
        print(f"\n  {title}")
        header = (f"  {'bucket':>16s} {'n':>7s} {'Acc@0.25':>9s} {'Acc@0.5':>9s}"
                  + (f" {'vs base':>9s}" if ref is not None else ""))
        print(header); print("  " + "-" * (len(header) - 2))
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (values >= lo) & (values < hi)
            if sel.sum() < args.min_bucket:
                continue
            line = (f"  {f'{lo:g}-{hi:g} {unit}':>16s} {int(sel.sum()):7d} "
                    f"{(iou[sel] >= 0.25).mean() * 100:9.2f} "
                    f"{(iou[sel] >= 0.5).mean() * 100:9.2f}")
            if ref is not None:
                delta = ((iou[sel] >= 0.25).mean() - (ref[sel] >= 0.25).mean()) * 100
                line += f" {delta:+9.2f}"
            print(line)

    print(f"\n{args.folder}"
          + (f"   vs baseline {args.baseline}" if args.baseline else ""))
    report("by utterance duration", seconds, [0, 2, 3, 4, 5, 6, 8, 1e9], "s")
    report("by objects in the scene", diag["n_objects"].astype(float),
           [0, 20, 40, 60, 80, 1e9], "obj")

    if ref is not None:
        print("\n  The 'vs base' column is the point of this table. A gain that is flat "
              "across\n  duration is not coming from the decomposition; a gain that grows "
              "with duration\n  and with clutter is the behaviour the method predicts.")
    else:
        print("\n  Pass --baseline <run> to get the paired difference column, which is "
              "what\n  makes this an argument rather than a description.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    table = sub.add_parser("results-table", help="metric table for several runs")
    table.add_argument("--folders", nargs="+", required=True)
    table.set_defaults(func=run_results_table)

    roles = sub.add_parser("slot-roles", help="did the slots specialise?")
    roles.add_argument("--folder", required=True)
    roles.add_argument("--by_outcome", action="store_true",
                       help="Split slot mass by whether the utterance was grounded.")
    roles.set_defaults(func=run_slot_roles)

    sharp = sub.add_parser("ot-sharpness", help="does transport concentrate the score?")
    sharp.add_argument("--folders", nargs="+", required=True)
    sharp.set_defaults(func=run_ot_sharpness)

    gain = sub.add_parser("gain-attribution", help="where a gain comes from")
    gain.add_argument("--folders", nargs=2, required=True, metavar=("BASELINE", "PROPOSED"))
    gain.set_defaults(func=run_gain_attribution)

    tax = sub.add_parser("error-taxonomy", help="recognition vs grounding failures")
    tax.add_argument("--folder", required=True)
    tax.set_defaults(func=run_error_taxonomy)

    sig = sub.add_parser("significance",
                         help="paired permutation test + bootstrap CI between two runs")
    sig.add_argument("folders", nargs=2, metavar=("BASELINE", "PROPOSED"))
    sig.add_argument("--iters", type=int, default=10000)
    sig.add_argument("--seed", type=int, default=0)
    sig.set_defaults(func=run_significance)

    cx = sub.add_parser("complexity",
                        help="accuracy by utterance duration and scene clutter")
    cx.add_argument("--folder", required=True)
    cx.add_argument("--baseline", default=None,
                    help="A second run, scored on the same utterances, for the paired "
                         "difference column.")
    cx.add_argument("--min_bucket", type=int, default=30)
    cx.set_defaults(func=run_complexity)

    seeds = sub.add_parser("seed-aggregate", help="mean ± sd over seeds")
    seeds.add_argument("--folders", nargs="+", required=True)
    seeds.set_defaults(func=run_seed_aggregate)

    args = parser.parse_args()
    # Every --folder also accepts an arm name ('spot') or a run tag, meaning its newest run.
    from experiments.audio.fusion_arms import resolve_run
    for name in ("folder", "baseline"):
        if getattr(args, name, None):
            setattr(args, name, resolve_run(getattr(args, name)))
    if getattr(args, "folders", None):
        args.folders = [resolve_run(f) for f in args.folders]
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
