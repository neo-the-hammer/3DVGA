"""
Hyperparameter sweeps for the 3D-VGA branch.

    python experiments/audio/sweeps.py list
    python experiments/audio/sweeps.py run slots --eval
    python experiments/audio/sweeps.py run slots --values 4 8 --eval     # a lighter grid
    python experiments/audio/sweeps.py run sinkhorn --dry_run
    python experiments/audio/sweeps.py table slots

The default point of every sweep IS the `spot` arm of fusion_arms.py, so it is not trained
a second time: `run` skips it once a VGA-SPOT run exists and `table` reads that run.

Separate from fusion_arms.py on purpose. That file answers "does each component earn its
place"; this one answers "is each component's setting defensible". A reviewer asks both,
and conflating them produces a table where two things changed between adjacent rows.

Every sweep varies ONE knob with everything else at the default, under a frozen detector.
Each entry names the question it settles, because a sweep whose result changes nothing is a
sweep not worth a GPU-week.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments.audio.fusion_arms import (  # noqa: E402
    COMMON, arm_tag, build_command, evaluate_tag, latest_run, read_report,
    warn_if_detector_unfrozen)


SWEEPS = {
    # ---------------------------------------------------------------- the proposed method
    "slots": dict(
        question="How many latent role slots? The text parse has three fields; K=6 leaves "
                 "headroom. Too few and roles share a slot; too many and the Sinkhorn "
                 "budget is spread over slots that claim noise.",
        flag="--num_slots",
        values=[4, 5, 6, 8, 12],
        default=6,
    ),
    "sinkhorn": dict(
        question="Entropic temperature and iteration count for BOTH transport stages. "
                 "Measured offline for marginal convergence (docs/3DVGA.md §6); this sweep "
                 "checks that convergence and accuracy actually agree.",
        # Both stages move together: they share the bounded-cosine cost design, so pinning
        # one and sweeping the other would test a configuration nobody would ship.
        env_grid=[
            {"VGA_OT_EPS": "0.15", "VGA_OT_ITERS": "8"},
            {"VGA_OT_EPS": "0.30", "VGA_OT_ITERS": "8"},
            {"VGA_OT_EPS": "0.30", "VGA_OT_ITERS": "4"},
            {"VGA_OT_EPS": "0.50", "VGA_OT_ITERS": "6"},
            {"VGA_OT_EPS": "1.00", "VGA_OT_ITERS": "4"},
        ],
        default="eps 0.30 / 8 iters",
        default_index=1,
    ),
    "depth": dict(
        question="How many stacked SPOT layers? The predecessor paper's Table 5 shows the "
                 "text model gaining with depth at negligible cost; this is the audio "
                 "counterpart, and it is what justifies the 3 in 'SPOT x 3'. Report "
                 "parameters and latency beside accuracy, which is why measurements.py "
                 "params takes --fusion_depth.",
        flag="--fusion_depth",
        values=[1, 2, 3, 4],
        default=3,
    ),
    "graph_k": dict(
        question="Neighbours per node in SpatialGraphNet's message passing. k=0 removes "
                 "the graph entirely, which is the spot-nograph arm; the rest ask how far "
                 "relational context should reach. The predecessor paper swept the same "
                 "parameter (its Figure 12) and found the optimum well below the proposal "
                 "count.",
        flag="--graph_k",
        values=[0, 5, 10, 15],
        default=5,
    ),
    "ot_weight": dict(
        question="How much of the score should come from the transport plan's target row "
                 "rather than the MLP head? The parameter is learnable, so this sweeps its "
                 "initialisation — a check that the result is not initialisation-bound.",
        flag="--ot_logit_weight",
        values=[0.0, 0.25, 0.5, 1.0],
        default=0.5,
    ),

    # ---------------------------------------------------------------- the acoustic front end
    "encoder_layer": dict(
        question="Which transformer layer of the frozen encoder to read. The last layer of "
                 "a self-supervised speech model is specialised for its pretraining "
                 "objective; mid layers carry more phonetic and lexical content. Layer 8 "
                 "of 12 is the default and it was chosen from the literature rather than "
                 "measured here, which is exactly why it needs a sweep. One cache per "
                 "point.",
        # The default layer reads the ordinary cache; every other point gets its own file.
        env_grid=[{"VGA_ENCODER_LAYER": str(v),
                   **({} if v == 8 else {"VGA_CACHE_SUFFIX": f"-layer{v}"})}
                  for v in (4, 6, 8, 10, 12)],
        default="layer 8",
        default_index=2,
        needs_cache=True,
    ),
    "prosody_channels": dict(
        question="Which prosodic cue group carries the signal. The spot-noprosody arm says "
                 "whether prosody helps at all; this says WHICH prosody, which is the "
                 "difference between a result and an explanation. energy = stress, "
                 "pitch = phrase melody, spectral = segmentation.",
        flag="--prosody_channels",
        values=["none", "energy", "pitch", "spectral", "all"],
        default="all",
    ),
    "encoder": dict(
        question="Which frozen speech encoder. wav2vec2 reproduces the paper's choice; "
                 "WavLM and HuBERT are drop-ins with identical geometry. Needs one feature "
                 "cache per encoder.",
        flag="--audio_encoder",
        values=["wav2vec2", "wav2vec2-large", "wavlm", "hubert"],
        default="wav2vec2",
        needs_cache=True,
    ),
    "frames": dict(
        question="The masked-length counterpart of the paper's num_frame ablation (their "
                 "Table 4, where accuracy peaks then FALLS). With real lengths carried as "
                 "an attention mask this curve should flatten — that flattening is the "
                 "claim, and this sweep is what tests it.",
        # The cap is applied when the cache is built, so each length is its own cache.
        env_grid=[{"VGA_MAX_AUDIO_LEN": str(v),
                   **({} if v == 256 else {"VGA_CACHE_SUFFIX": f"-len{v}"})}
                  for v in (64, 128, 256, 384)],
        default="256 frames",
        default_index=2,
        needs_cache=True,
    ),

    # ---------------------------------------------------------------- unsupervised shaping
    # With no text teacher, these regularisers and the time prior are the ONLY things
    # steering the decomposition. They matter more here than they would in a distilled
    # model, so they get swept rather than assumed.
    "slot_shaping": dict(
        question="Weights on the slot entropy and orthogonality terms. Too weak and the "
                 "slots collapse onto one span; too strong and they partition the audio "
                 "evenly regardless of what was said.",
        env_grid=[
            {"VGA_LOSS_SLOT_ENTROPY": "0.0", "VGA_LOSS_SLOT_ORTHO": "0.0"},
            {"VGA_LOSS_SLOT_ENTROPY": "0.02", "VGA_LOSS_SLOT_ORTHO": "0.01"},
            {"VGA_LOSS_SLOT_ENTROPY": "0.05", "VGA_LOSS_SLOT_ORTHO": "0.05"},
            {"VGA_LOSS_SLOT_ENTROPY": "0.1", "VGA_LOSS_SLOT_ORTHO": "0.1"},
        ],
        default="entropy 0.02 / ortho 0.01",
        default_index=1,
    ),
    "role_separation": dict(
        question="Weight on the penalty keeping the referent and relation elections "
                 "disjoint. At 0 both role queries may elect the same slot, which hands "
                 "RSK the referent vector and silently turns the geometric kernel into a "
                 "second copy of the target pathway.",
        env_grid=[
            {"VGA_LOSS_ROLE_SEPARATION": "0.0"},
            {"VGA_LOSS_ROLE_SEPARATION": "0.05"},
            {"VGA_LOSS_ROLE_SEPARATION": "0.2"},
        ],
        default="0.05",
        default_index=1,
    ),
    "target_contrastive": dict(
        question="Weight on the in-batch InfoNCE tying the elected referent to the "
                 "ground-truth proposal. This is the ONLY direct supervision any role "
                 "receives, and it comes from geometry -- at 0, role election is driven "
                 "entirely by the grounding gradient.",
        env_grid=[
            {"VGA_LOSS_TARGET_CONTRASTIVE": "0.0"},
            {"VGA_LOSS_TARGET_CONTRASTIVE": "0.2"},
            {"VGA_LOSS_TARGET_CONTRASTIVE": "0.5"},
        ],
        default="0.2",
        default_index=1,
    ),
    "audio_cls": dict(
        question="Weight gamma on the audio classification loss, the one auxiliary term "
                 "inherited from the 3DVG-Audio paper (their Eq. 1, where gamma = 1 was "
                 "chosen by ablation). At 0 the acoustic branch is supervised by grounding "
                 "alone; this checks that their choice carries over to a different fusion.",
        env_grid=[
            {"VGA_LOSS_AUDIO_CLS": "0.0"},
            {"VGA_LOSS_AUDIO_CLS": "0.5"},
            {"VGA_LOSS_AUDIO_CLS": "1.0"},
        ],
        default="1.0",
        default_index=2,
    ),

    # ---------------------------------------------------------------- reliability
    "seeds": dict(
        question="Run-to-run variance. The single most common reviewer objection to a "
                 "one-point improvement, and the only sweep that is not optional: every "
                 "gap in the ablation table must be reported against this spread.",
        flag="--seed",
        values=[42, 1, 2, 3, 4],
        default=42,
        aggregate="spread",
    ),
}


def _tag(name, value):
    safe = str(value).replace(" ", "").replace("/", "").replace(".", "p")
    return f"{COMMON['RUN_PREFIX']}SWEEP-{name}-{safe}".upper()


def _points(name, values=None):
    """[(tag, extra_cli, extra_env, is_default)] for one sweep.

    ``values`` narrows a flag sweep to a lighter grid. ``is_default`` marks the point that
    is the proposed configuration, i.e. the `spot` arm.
    """
    spec = SWEEPS[name]
    out = []
    if "env_grid" in spec:
        if values:
            raise SystemExit(f"--values does not apply to '{name}': its points set more "
                             f"than one knob. Edit SWEEPS['{name}']['env_grid'] instead.")
        for index, grid in enumerate(spec["env_grid"]):
            label = "-".join(f"{k.split('_')[-1].lower()}{v}" for k, v in grid.items()
                             if k != "VGA_CACHE_SUFFIX")
            out.append((_tag(name, label), [], grid, index == spec.get("default_index")))
    else:
        kind = type(spec["default"])
        for value in ([kind(v) for v in values] if values else spec["values"]):
            out.append((_tag(name, value), [spec["flag"], str(value)], {},
                        value == spec["default"]))
    return out


def ensure_cache(extra, env, dry_run=False):
    """Build the feature cache a sweep point reads, unless it is already on disk.

    Three sweeps change what the frozen encoder produced -- which encoder, which layer,
    how many frames -- so their points cannot share the ordinary cache. Needs the WAVs.
    """
    from lib.audio_config import AUDIO

    encoder = extra[extra.index("--audio_encoder") + 1] if "--audio_encoder" in extra \
        else COMMON["ENCODER"]
    suffix = env.get("VGA_CACHE_SUFFIX", "")
    for split, scenes in (("train", COMMON["NUM_SCENES"]), ("val", COMMON["NUM_VAL_SCENES"])):
        path = os.path.join(AUDIO.FEAT_DIR, encoder, f"{split}{suffix}.h5")
        if os.path.isfile(path):
            continue
        cmd = [sys.executable, os.path.join("scripts", "audio_encode", "cache_audio_features.py"),
               "--splits", split, "--encoder", encoder]
        if suffix:
            cmd += [f"--cache_suffix={suffix}"]
        if scenes != -1:
            cmd += ["--num_scenes", str(scenes)]
        shown = " ".join(f"{k}={v}" for k, v in env.items())
        print(f"[3D-VGA] no {path}; building it\n$ {shown} {' '.join(cmd)}".rstrip(),
              flush=True)
        if not dry_run:
            code = subprocess.call(cmd, cwd=REPO_ROOT, env={**os.environ, **env})
            if code != 0:
                return code
    return 0


def _spot_stands_in(is_default):
    """True when the default point can be read from the finished `spot` arm."""
    return is_default and latest_run(arm_tag("spot")) is not None


def run_sweep(args):
    spec = SWEEPS[args.sweep]
    print(f"\n{args.sweep}: {spec['question']}\n")
    if spec.get("needs_cache"):
        print("[3D-VGA] NOTE: every point of this sweep reads its own feature cache, built\n"
              "         here from the WAVs when it is missing (minutes per point, and\n"
              "         several GB of disk each).\n")
    warn_if_detector_unfrozen()

    codes = {}
    for tag, extra, env, is_default in _points(args.sweep, args.values):
        if _spot_stands_in(is_default) and not args.include_default:
            print(f"[3D-VGA] {tag}: the default point is the spot arm "
                  f"({latest_run(arm_tag('spot'))}); not trained again", flush=True)
            continue
        if spec.get("needs_cache"):
            codes[tag] = ensure_cache(extra, env, dry_run=args.dry_run)
            if codes[tag] != 0:
                print(f"[3D-VGA] {tag}: could not build its feature cache, skipped")
                continue
        cmd = build_command("spot", os.path.join("scripts", "audio_train.py"), extra)
        # build_command tags every run VGA-SPOT; replace it with the sweep point's tag so
        # the table can find the runs again.
        cmd[cmd.index("--tag") + 1] = tag
        environment = {**os.environ, **env}
        shown = " ".join(f"{k}={v}" for k, v in env.items())
        print(f"$ {shown} {' '.join(cmd)}".strip(), flush=True)
        codes[tag] = 0 if args.dry_run else subprocess.call(cmd, cwd=REPO_ROOT,
                                                            env=environment)
        if codes[tag] == 0 and args.eval:
            codes[tag] = evaluate_tag(tag, force=True, dry_run=args.dry_run)

    failed = [t for t, c in codes.items() if c != 0]
    print(f"\n{len(codes) - len(failed)}/{len(codes)} point(s) succeeded"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def run_eval(args):
    """Score every trained point of a sweep that has no scores yet."""
    codes = {}
    for tag, _, _, is_default in _points(args.sweep, args.values):
        if latest_run(tag) is None and _spot_stands_in(is_default):
            tag = arm_tag("spot")
        codes[tag] = evaluate_tag(tag, force=args.force, dry_run=args.dry_run)
    failed = [t for t, c in codes.items() if c != 0]
    print(f"\n{len(codes) - len(failed)}/{len(codes)} point(s) evaluated"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def run_table(args):
    spec = SWEEPS[args.sweep]
    print(f"\n{args.sweep}: {spec['question']}")
    print(f"default: {spec['default']}\n")

    header = (f"{'point':22s} {'U@0.25':>8s} {'U@0.5':>8s} {'M@0.25':>8s} "
              f"{'M@0.5':>8s} {'O@0.25':>8s} {'O@0.5':>8s}")
    print(header)
    print("-" * len(header))

    rows = []
    for tag, extra, env, is_default in _points(args.sweep, args.values):
        label = tag.split(f"SWEEP-{args.sweep.upper()}-", 1)[-1].lower()
        report = read_report(tag)
        if report is None and is_default:
            report = read_report(arm_tag("spot"))
            label += " (spot)"
        if report is None:
            print(f"{label:22s} {'-- not evaluated --':>54s}")
            continue
        u, m, o = report["unique"], report["multiple"], report["overall"]
        rows.append((label, o))
        print(f"{label:22s} {u[0]:8.2f} {u[1]:8.2f} {m[0]:8.2f} {m[1]:8.2f} "
              f"{o[0]:8.2f} {o[1]:8.2f}")

    if spec.get("aggregate") == "spread" and len(rows) > 1:
        at25 = [o[0] for _, o in rows]
        at50 = [o[1] for _, o in rows]
        mean25 = sum(at25) / len(at25)
        mean50 = sum(at50) / len(at50)
        sd25 = (sum((x - mean25) ** 2 for x in at25) / (len(at25) - 1)) ** 0.5
        sd50 = (sum((x - mean50) ** 2 for x in at50) / (len(at50) - 1)) ** 0.5
        print("-" * len(header))
        print(f"{'mean ± sd':22s} {'':8s} {'':8s} {'':8s} {'':8s} "
              f"{mean25:8.2f} {mean50:8.2f}")
        print(f"{'':22s} {'':8s} {'':8s} {'':8s} {'':8s} "
              f"{'±' + format(sd25, '.2f'):>8s} {'±' + format(sd50, '.2f'):>8s}")
        print(f"\nReport every ablation gap against this spread. A 0.4-point gain over "
              f"{sd25:.2f} points of seed noise is not a result.")

    if not rows:
        print(f"\nrun `sweeps.py eval {args.sweep}` first.")
    return 0


def run_list(args):
    print()
    for name, spec in SWEEPS.items():
        values = (spec.get("values")
                  or [" · ".join(f"{k}={v}" for k, v in g.items()) for g in spec["env_grid"]])
        print(f"  {name}")
        print(f"      {spec['question']}")
        print(f"      points : {values}")
        print(f"      default: {spec['default']}")
        if spec.get("needs_cache"):
            print("      NOTE   : needs one feature cache per point")
        print()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=run_list)

    def common(p):
        p.add_argument("sweep", choices=list(SWEEPS))
        p.add_argument("--values", nargs="+", default=None,
                       help="Run only these points of a single-flag sweep (a lighter grid).")
        return p

    run = common(sub.add_parser("run"))
    run.add_argument("--dry_run", action="store_true")
    run.add_argument("--eval", action="store_true",
                     help="Evaluate each point as soon as it has trained.")
    run.add_argument("--include_default", action="store_true",
                     help="Train the default point even though the spot arm covers it.")
    run.set_defaults(func=run_sweep)

    evaluate = common(sub.add_parser("eval", help="score the trained points of a sweep"))
    evaluate.add_argument("--force", action="store_true")
    evaluate.add_argument("--dry_run", action="store_true")
    evaluate.set_defaults(func=run_eval)

    common(sub.add_parser("table")).set_defaults(func=run_table)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
