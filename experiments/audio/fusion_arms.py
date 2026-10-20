"""
The 3D-VGA fusion comparison, driven from one config table.

    python experiments/audio/fusion_arms.py list
    python experiments/audio/fusion_arms.py run spot --eval
    python experiments/audio/fusion_arms.py run-all --eval
    python experiments/audio/fusion_arms.py eval-all
    python experiments/audio/fusion_arms.py table

Every arm shares COMMON and differs only
in its row of ARMS, so a gap between two rows is attributable to the one thing that row
changes. Overridable from the environment as VGA_<NAME>, which is how the notebooks set it.

Every arm trains the acoustic branch and the fusion head only. Both inputs come off disk:
proposals from the scene cache (cached_scenes/), encoder frames from the audio feature
cache (data_audio/feat/). No detector and no speech encoder runs.

Two groups of arms.

**Progressive** -- each adds a single component to the one above it, so the table reads as
an ablation rather than a list of numbers:

    concat        AP-Refer's fusion (mean-pooled audio, concat, MLP)
    xattn         + attention over frames and slots
    spot-k2       + OT routing, with only as many slots as there are roles
    spot          + slot induction beyond the two roles (proposed)

**Leave-one-out** -- the proposed model with exactly one component removed. These answer
"does this piece earn its place?", which is a different question from "how far did we get?",
and they are the rows a reviewer checks first:

    spot-softmax     Sinkhorn -> row-wise softmax        the central claim
    spot-unidir      no slots-read-proposals direction
    spot-fixedroles  fixed slot indices, no role election
    spot-norsk       fixed -distance prior, no RSK
    spot-nograph     no SpatialGraphNet
    spot-nogate      plain residual instead of the read-out gate
    spot-noframes    no residual pass over raw encoder frames
    spot-noprosody   no prosody stream
    spot-nocopypaste no proposal copy-paste augmentation
    spot-notimeprior no Gaussian time prior on the slots
    spot-nospecaug   no SpecAugment on the cached frames

`spot-softmax` is the one that matters most. Every other arm removes a module; this one
keeps the module, the cost matrix, the temperature, the read-out and the parameter count,
and changes only whether the proposals compete for a slot's mass. If the paper's argument
is right, this is where the largest single drop on the Multiple subset appears -- and if it
is not, the argument is wrong and no other arm can rescue it.

Run them on the scene cache (the default below) or nothing is comparable: the detection
branch contributes most of the variance, and refitting it per arm would bury the fusion
effect under detector noise.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def _env(name, default):
    raw = os.environ.get(f"VGA_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, list):
        return [x for x in (p.strip() for p in raw.split(",")) if x]
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


COMMON = dict(
    USE_CACHED_SCENES=_env("USE_CACHED_SCENES", True),
    CACHED_SCENES_ROOT=_env("CACHED_SCENES_ROOT", "cached_scenes"),
    EPOCH=_env("EPOCH", 30),
    BATCH_SIZE=_env("BATCH_SIZE", 8),
    LANG_NUM_MAX=_env("LANG_NUM_MAX", 8),
    SEED=_env("SEED", 42),
    LR=_env("LR", 1e-3),
    VAL_STEP=_env("VAL_STEP", 2000),
    VERBOSE=_env("VERBOSE", 50),
    NUM_WORKERS=_env("NUM_WORKERS", 4),
    # -1 = every scene. A small number is a pilot run, not a result.
    NUM_SCENES=_env("NUM_SCENES", -1),
    NUM_VAL_SCENES=_env("NUM_VAL_SCENES", -1),
    # True trains and scores only the utterances that have been synthesised so far.
    ONLY_CACHED_AUDIO=_env("ONLY_CACHED_AUDIO", False),
    ENCODER=_env("ENCODER", "wav2vec2"),
    EXTRA_ARGS=_env("EXTRA_ARGS", ["--use_color", "--use_normal", "--coslr"]),
    # Which checkpoint of a run `eval` scores; the same default as scripts/audio_eval.py.
    CHECKPOINT=_env("CHECKPOINT", "model_criteria_25.pth"),
    # Prepended to every run tag, so pilot and smoke runs are never read as results.
    RUN_PREFIX=_env("RUN_PREFIX", ""),
    CPU=_env("CPU", False),
    # Only without the scene cache: freeze a live detector warm-started from this run.
    PRETRAINED=_env("PRETRAINED", ""),
    FREEZE_DETECTOR=_env("FREEZE_DETECTOR", True),
)

ARMS = {
    "concat": dict(
        flags=["--audio_fusion", "concat", "--no_prosody"],
        note="AP-Refer's fusion: mean-pooled audio, concatenated, masked, MLP.",
    ),
    "xattn": dict(
        flags=["--audio_fusion", "xattn"],
        note="Attentive fusion: proposals cross-attend over frames and slots.",
    ),
    "spot-k2": dict(
        flags=["--audio_fusion", "spot", "--num_slots", "2"],
        note="Proposed fusion with only as many slots as roles -- the floor for slot "
             "induction, since the two role queries have nothing left to choose between.",
    ),
    "spot-norsk": dict(
        flags=["--audio_fusion", "spot", "--no_rsk"],
        note="Proposed fusion, fixed -distance geometric prior instead of RSK.",
    ),
    "spot-noprosody": dict(
        flags=["--audio_fusion", "spot", "--no_prosody"],
        note="Everything except the prosody stream.",
    ),
    "spot": dict(
        flags=["--audio_fusion", "spot"],
        note="Proposed: SPOT + RSK + prosody, audio-only.",
    ),

    # ---- leave-one-out: the proposed model minus exactly one component ----
    "spot-softmax": dict(
        flags=["--audio_fusion", "spot", "--no_ot"],
        note="THE central-claim arm. Sinkhorn replaced by a row-wise softmax over "
             "proposals: same cost matrix, same temperature, same slot masses, same "
             "read-out, same parameters. The only thing removed is the column constraint, "
             "i.e. whether proposals compete. Report the Multiple subset separately.",
    ),
    "spot-unidir": dict(
        flags=["--audio_fusion", "spot", "--no_bidirectional"],
        note="No slots-read-proposals direction: slots summarise the utterance without "
             "ever seeing the scene. Isolates the bidirectional update.",
    ),
    "spot-fixedroles": dict(
        flags=["--audio_fusion", "spot", "--no_role_election"],
        note="Roles read off fixed slot indices instead of being elected. Measures what "
             "the election is worth now that no text teacher pins those indices.",
    ),
    "spot-nograph": dict(
        flags=["--audio_fusion", "spot", "--graph_k", "0"],
        note="No SpatialGraphNet: proposals reach the fusion head without relational "
             "message passing.",
    ),
    "spot-nogate": dict(
        flags=["--audio_fusion", "spot", "--no_route_gate"],
        note="Plain residual add instead of the learned gate on the routed read-out.",
    ),
    "spot-noframes": dict(
        flags=["--audio_fusion", "spot", "--no_frame_residual"],
        note="No residual pass over raw encoder frames: the K slots become the only "
             "acoustic pathway, so this measures what the slots compress away.",
    ),
    "spot-nocopypaste": dict(
        flags=["--audio_fusion", "spot", "--disable_copy_paste"],
        note="No proposal copy-paste augmentation. The text branch's version of this "
             "ablation is Section 4.6.5 of the predecessor paper.",
    ),
    "spot-notimeprior": dict(
        flags=["--audio_fusion", "spot", "--no_time_prior"],
        note="No Gaussian time prior on the slots. With no text teacher the prior is one "
             "of the few things that shape the decomposition, and at initialisation the "
             "only one that separates the slots in time.",
    ),
    "spot-nospecaug": dict(
        flags=["--audio_fusion", "spot", "--no_spec_augment"],
        note="No SpecAugment on the cached encoder frames -- the only augmentation the "
             "acoustic stream gets.",
    ),
}

#: Arms that belong in the progressive table vs the leave-one-out table. `table` prints
#: them as two blocks, because mixing them invites reading a leave-one-out row as a step.
PROGRESSIVE = ("concat", "xattn", "spot-k2", "spot")
LEAVE_ONE_OUT = ("spot-softmax", "spot-unidir", "spot-fixedroles", "spot-norsk",
                 "spot-nograph", "spot-nogate", "spot-noframes", "spot-noprosody",
                 "spot-nocopypaste", "spot-notimeprior", "spot-nospecaug")

#: The arms worth running first: the baselines, the central claim, and the two components
#: the method adds on the acoustic side. `run-all --core` runs these and stops.
CORE = ("spot", "concat", "xattn", "spot-softmax", "spot-norsk", "spot-noprosody")


def arm_tag(arm):
    return f"{COMMON['RUN_PREFIX']}VGA-{arm}".upper()


def _shared_flags():
    """Flags a training and an evaluation command have in common."""
    cmd = list(COMMON["EXTRA_ARGS"])
    if COMMON["USE_CACHED_SCENES"]:
        cmd += ["--use_cached_scenes", "--cached_scenes_root", COMMON["CACHED_SCENES_ROOT"]]
    if COMMON["ONLY_CACHED_AUDIO"]:
        cmd += ["--only_cached_audio"]
    if COMMON["CPU"]:
        cmd += ["--cpu"]
    return cmd


def build_command(arm, script, extra=()):
    cfg = ARMS[arm]
    cmd = [sys.executable, script,
           "--tag", arm_tag(arm),
           "--epoch", str(COMMON["EPOCH"]),
           "--batch_size", str(COMMON["BATCH_SIZE"]),
           "--audio_lang_num_max", str(COMMON["LANG_NUM_MAX"]),
           "--seed", str(COMMON["SEED"]),
           "--lr", str(COMMON["LR"]),
           "--val_step", str(COMMON["VAL_STEP"]),
           "--verbose", str(COMMON["VERBOSE"]),
           "--num_workers", str(COMMON["NUM_WORKERS"]),
           "--audio_encoder", COMMON["ENCODER"]]
    if COMMON["NUM_SCENES"] != -1:
        cmd += ["--num_scenes", str(COMMON["NUM_SCENES"])]
    if COMMON["NUM_VAL_SCENES"] != -1:
        cmd += ["--num_val_scenes", str(COMMON["NUM_VAL_SCENES"])]
    cmd += _shared_flags()
    if COMMON["PRETRAINED"] and not COMMON["USE_CACHED_SCENES"]:
        cmd += ["--use_pretrained", COMMON["PRETRAINED"]]
        if COMMON["FREEZE_DETECTOR"]:
            cmd += ["--no_detection"]
    cmd += cfg["flags"]
    cmd += list(extra)
    return cmd


def latest_run(tag):
    """Newest run directory under outputs/ whose name ends in ``_<TAG>``, or None."""
    outputs = os.path.join(REPO_ROOT, "outputs")
    if not os.path.isdir(outputs):
        return None
    runs = sorted(d for d in os.listdir(outputs) if d.endswith("_" + tag.upper()))
    return runs[-1] if runs else None


def resolve_run(name):
    """A run directory from a directory name, an arm name ('spot') or a run tag.

    Lets the analyses be pointed at an arm instead of a timestamped folder.
    """
    if name is None or os.path.isdir(os.path.join(REPO_ROOT, "outputs", name)):
        return name
    if name in ARMS:
        return latest_run(arm_tag(name)) or name
    return latest_run(COMMON["RUN_PREFIX"] + name) or latest_run(name) or name


def eval_command(folder, extra=(), out_subdir=""):
    """The scripts/audio_eval.py argv for one finished run.

    The architecture comes back from the run's info.json, so nothing about the arm is
    repeated here; ``extra`` is for flags that deliberately differ from training.
    """
    cmd = [sys.executable, os.path.join("scripts", "audio_eval.py"),
           "--folder", folder, "--force", "--dump_diagnostics",
           "--checkpoint", COMMON["CHECKPOINT"],
           "--batch_size", str(COMMON["BATCH_SIZE"]),
           "--audio_lang_num_max", str(COMMON["LANG_NUM_MAX"])]
    if COMMON["NUM_VAL_SCENES"] != -1:
        cmd += ["--num_scenes", str(COMMON["NUM_VAL_SCENES"])]
    cmd += _shared_flags()
    if out_subdir:
        cmd += ["--out_subdir", out_subdir]
    return cmd + list(extra)


def evaluate_tag(tag, force=False, dry_run=False):
    """Score the newest run carrying ``tag``. Returns an exit code."""
    folder = latest_run(tag)
    if folder is None:
        print(f"[3D-VGA] no run tagged {tag} under outputs/ -- train it first")
        return 0 if dry_run else 1
    root = os.path.join(REPO_ROOT, "outputs", folder)
    if os.path.isfile(os.path.join(root, "scores.p")) and not force:
        print(f"[3D-VGA] {folder}: scores.p exists, skipping (--force to redo)")
        return 0
    if not os.path.isfile(os.path.join(root, COMMON["CHECKPOINT"])) and not dry_run:
        print(f"[3D-VGA] {folder} has no {COMMON['CHECKPOINT']}: training did not reach a "
              f"validation pass. Lower VGA_VAL_STEP, or set VGA_CHECKPOINT=model_last.pth.")
        return 1
    cmd = eval_command(folder)
    print("$", " ".join(cmd), flush=True)
    return 0 if dry_run else subprocess.call(cmd, cwd=REPO_ROOT)


def run_arm(args):
    if args.arm not in ARMS:
        raise SystemExit(f"unknown arm {args.arm!r}; try: {', '.join(ARMS)}")
    warn_if_detector_unfrozen()
    cmd = build_command(args.arm, os.path.join("scripts", "audio_train.py"), args.extra)
    print("$", " ".join(cmd), flush=True)
    code = 0 if args.dry_run else subprocess.call(cmd, cwd=REPO_ROOT)
    if code == 0 and getattr(args, "eval", False):
        code = evaluate_tag(arm_tag(args.arm), force=True, dry_run=args.dry_run)
    return code


def warn_if_detector_unfrozen():
    if COMMON["USE_CACHED_SCENES"] or COMMON["PRETRAINED"]:
        return
    print("\n[3D-VGA] WARNING: no scene cache and no VGA_PRETRAINED, so this run trains\n"
          "         its own detector. Runs are then not comparable to each other -- the\n"
          "         detector contributes more variance than the fusion does.\n")


def _selected(args):
    return CORE if getattr(args, "core", False) else tuple(ARMS)


def run_all(args):
    arms = _selected(args)
    codes = {}
    for arm in arms:
        print(f"\n{'=' * 70}\n{arm}: {ARMS[arm]['note']}\n{'=' * 70}", flush=True)
        args.arm = arm
        codes[arm] = run_arm(args)
    failed = [a for a, c in codes.items() if c != 0]
    print(f"\ndone. {len(arms) - len(failed)}/{len(arms)} arm(s) succeeded"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def run_eval(args):
    if getattr(args, "arm", None):
        arms = (args.arm,)
    else:
        # eval-all scores what has been trained; an arm nobody ran is not a failure.
        arms = tuple(a for a in _selected(args) if latest_run(arm_tag(a)) or args.dry_run)
        print(f"{len(arms)} trained arm(s): {', '.join(arms) or 'none'}")
    codes = {arm: evaluate_tag(arm_tag(arm), force=args.force, dry_run=args.dry_run)
             for arm in arms}
    failed = [a for a, c in codes.items() if c != 0]
    print(f"\n{len(arms) - len(failed)}/{len(arms)} arm(s) evaluated"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def run_list(args):
    print(f"\nCOMMON: {json.dumps(COMMON, indent=2)}\n")
    for title, group in (("progressive", PROGRESSIVE), ("leave-one-out", LEAVE_ONE_OUT)):
        print(f"--- {title} " + "-" * (60 - len(title)))
        for arm in group:
            cfg = ARMS[arm]
            print(f"  {arm:17s} {cfg['note']}")
            print(f"  {'':17s} {' '.join(cfg['flags'])}\n")
    return 0


def read_report(tag, subdir=""):
    """The ``report`` block of the newest run tagged ``tag``, or None."""
    folder = latest_run(tag)
    if folder is None:
        return None
    path = os.path.join(REPO_ROOT, "outputs", folder, subdir, "scores.p")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle).get("report")


def _report_for(arm):
    return read_report(arm_tag(arm))


def run_table(args):
    """Collect scores.p from every finished arm into the two tables."""
    reports = {arm: _report_for(arm) for arm in ARMS}
    baseline = reports.get("spot")

    header = (f"{'arm':17s} {'U@0.25':>8s} {'U@0.5':>8s} {'M@0.25':>8s} "
              f"{'M@0.5':>8s} {'O@0.25':>8s} {'O@0.5':>8s}")

    def emit(arm, delta=False):
        report = reports[arm]
        if report is None:
            print(f"{arm:17s} {'-- not evaluated --':>54s}")
            return
        u, m, o = report["unique"], report["multiple"], report["overall"]
        line = (f"{arm:17s} {u[0]:8.2f} {u[1]:8.2f} {m[0]:8.2f} {m[1]:8.2f} "
                f"{o[0]:8.2f} {o[1]:8.2f}")
        if delta and baseline is not None and arm != "spot":
            # The leave-one-out number that matters is the drop, not the level, and the
            # Multiple drop is the one the method's argument predicts.
            dm = m[0] - baseline["multiple"][0]
            do = o[0] - baseline["overall"][0]
            line += f"    dM@0.25 {dm:+6.2f}   dO@0.25 {do:+6.2f}"
        print(line)

    print("\n=== progressive: each row adds one component ===")
    print(header); print("-" * len(header))
    print(f"{'AP-Refer (paper)':17s} {48.62:8.2f} {29.59:8.2f} {16.94:8.2f} "
          f"{9.96:8.2f} {23.09:8.2f} {13.77:8.2f}")
    for arm in PROGRESSIVE:
        emit(arm)

    print("\n=== leave-one-out: proposed minus one component ===")
    print(header); print("-" * len(header))
    emit("spot")
    for arm in LEAVE_ONE_OUT:
        emit(arm, delta=True)

    if baseline is None:
        print("\nNOTE: the 'spot' arm has no scores.p, so no deltas could be computed.")
    print("\nDeltas are against 'spot'. Report them against the seed spread from "
          "sweeps.py seeds --\na drop smaller than that spread is not evidence that a "
          "component earns its place.\n")
    print("run `fusion_arms.py eval-all` for any arm marked 'not evaluated'.\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list").set_defaults(func=run_list)
    sub.add_parser("table").set_defaults(func=run_table)

    run = sub.add_parser("run")
    run.add_argument("arm", choices=list(ARMS))
    run.add_argument("--dry_run", action="store_true")
    run.add_argument("--eval", action="store_true",
                     help="Evaluate the run as soon as it has trained.")
    run.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    run.set_defaults(func=run_arm)

    every = sub.add_parser("run-all")
    every.add_argument("--core", action="store_true",
                       help=f"Only the core arms: {', '.join(CORE)}.")
    every.add_argument("--dry_run", action="store_true")
    every.add_argument("--eval", action="store_true",
                       help="Evaluate each arm as soon as it has trained.")
    every.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    every.set_defaults(func=run_all)

    one = sub.add_parser("eval", help="score the newest run of one arm")
    one.add_argument("arm", choices=list(ARMS))
    one.add_argument("--force", action="store_true")
    one.add_argument("--dry_run", action="store_true")
    one.set_defaults(func=run_eval)

    many = sub.add_parser("eval-all", help="score every trained arm that has no scores yet")
    many.add_argument("--core", action="store_true")
    many.add_argument("--force", action="store_true")
    many.add_argument("--dry_run", action="store_true")
    many.set_defaults(func=run_eval)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
