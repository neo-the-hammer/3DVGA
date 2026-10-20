"""
Evaluation-only experiments on a trained 3D-VGA run. Nothing here trains.

    python experiments/audio/interventions.py list
    python experiments/audio/interventions.py controls            # newest spot run
    python experiments/audio/interventions.py knockout --folder <run>
    python experiments/audio/interventions.py table

Each experiment re-scores one checkpoint with something changed at test time and archives
the result under ``outputs/<run>/interventions/<name>/``, so the run's own scores.p is
never overwritten -- the same layout as the text branch's ``parser_swap/``.

Two groups, answering two different questions.

**controls** -- is the model listening? The scene, the proposals and the labels stay; only
the audio changes. ``silence`` and ``mismatch`` measure what the scene alone is worth: a
model that scores nearly as well on another object's utterance is reading the room, not the
speech, and its headline number is not a statement about grounding from audio.
``reverse`` and the two halves ask whether word order and position carry the decision.

**knockout** -- what does the trained model rely on? One component is switched off at test
time, weights untouched. This is NOT the leave-one-out table of fusion_arms.py: a retrained
arm shows whether a component is *needed*, because the rest of the model can adapt to its
absence; a knockout shows whether the trained model *uses* it. A component with a small
leave-one-out drop and a large knockout drop is used but replaceable, which is a different
finding from one that is idle.
"""

import argparse
import os
import pickle
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments.audio.fusion_arms import (  # noqa: E402
    COMMON, arm_tag, eval_command, latest_run, resolve_run)

ARCHIVE = "interventions"

CONTROLS = {
    "silence": dict(
        flags=["--audio_control", "silence"],
        note="All audio zeroed. The floor: what the scene and the proposals give alone."),
    "mismatch": dict(
        flags=["--audio_control", "mismatch"],
        note="Each annotation hears an utterance about a different object class."),
    "reverse": dict(
        flags=["--audio_control", "reverse"],
        note="Frames in reverse order. Same content, no word order."),
    "first-half": dict(
        flags=["--audio_control", "first_half"],
        note="Only the first half of each clip -- usually the head noun and its modifiers."),
    "second-half": dict(
        flags=["--audio_control", "second_half"],
        note="Only the second half -- usually the spatial relation and its anchor."),
}

# Only switches that are read in forward() or that remove a module outright: the checkpoint
# still loads in full. A switch that resizes a layer (--prosody_channels, --num_slots)
# cannot be knocked out and needs its own training run.
KNOCKOUTS = {
    "no-ot": dict(
        flags=["--no_ot"],
        note="Sinkhorn -> row-wise softmax: proposals stop competing for a slot's mass."),
    "no-rsk": dict(
        flags=["--no_rsk"],
        note="Fixed -distance prior instead of the relation-conditioned kernel."),
    "no-bidir": dict(
        flags=["--no_bidirectional"],
        note="Slots no longer read the proposals."),
    "no-gate": dict(
        flags=["--no_route_gate"],
        note="Plain residual instead of the learned read-out gate."),
    "no-frames": dict(
        flags=["--no_frame_residual"],
        note="No residual pass over raw encoder frames; slots are the only audio path."),
    "no-graph": dict(
        flags=["--graph_k", "0"],
        note="No message passing between proposals."),
    "no-timeprior": dict(
        flags=["--no_time_prior"],
        note="Slot induction without the time prior."),
    "prosody-flat": dict(
        flags=["--prosody_keep", "none"],
        note="Every prosodic channel flattened to its per-utterance mean."),
    "prosody-energy": dict(
        flags=["--prosody_keep", "energy"],
        note="Only loudness and its slope keep their variation."),
    "prosody-pitch": dict(
        flags=["--prosody_keep", "pitch"],
        note="Only F0 and voicing keep their variation."),
    "prosody-spectral": dict(
        flags=["--prosody_keep", "spectral"],
        note="Only spectral flux and boundary strength keep their variation."),
}

GROUPS = {"controls": CONTROLS, "knockout": KNOCKOUTS}


def resolve_folder(args):
    folder = resolve_run(args.folder) or latest_run(arm_tag(args.arm))
    if folder is None:
        raise SystemExit(f"no run tagged {arm_tag(args.arm)} under outputs/; train it "
                         f"(fusion_arms.py run {args.arm} --eval) or pass --folder")
    return folder


def run_group(args):
    table = GROUPS[args.command]
    names = args.only or list(table)
    unknown = [n for n in names if n not in table]
    if unknown:
        raise SystemExit(f"unknown {args.command} name(s) {unknown}; try: {', '.join(table)}")

    folder = resolve_folder(args)
    print(f"\n{args.command} on {folder} ({COMMON['CHECKPOINT']})\n")
    codes = {}
    for name in names:
        out = os.path.join(REPO_ROOT, "outputs", folder, ARCHIVE, name, "scores.p")
        if os.path.isfile(out) and not args.force:
            print(f"[3D-VGA] {name}: already scored, skipping (--force to redo)")
            codes[name] = 0
            continue
        print(f"--- {name}: {table[name]['note']}", flush=True)
        cmd = eval_command(folder, extra=table[name]["flags"],
                           out_subdir=os.path.join(ARCHIVE, name))
        print("$", " ".join(cmd), flush=True)
        codes[name] = 0 if args.dry_run else subprocess.call(cmd, cwd=REPO_ROOT)

    failed = [n for n, c in codes.items() if c != 0]
    print(f"\n{len(codes) - len(failed)}/{len(codes)} {args.command} experiment(s) done"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def _report(folder, subdir=""):
    path = os.path.join(REPO_ROOT, "outputs", folder, subdir, "scores.p")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle).get("report")


def run_table(args):
    folder = resolve_folder(args)
    clean = _report(folder)

    header = (f"{'':17s} {'U@0.25':>8s} {'U@0.5':>8s} {'M@0.25':>8s} "
              f"{'M@0.5':>8s} {'O@0.25':>8s} {'O@0.5':>8s} {'dM@0.25':>9s} {'dO@0.25':>9s}")

    def emit(name, report):
        if report is None:
            print(f"{name:17s} {'-- not evaluated --':>54s}")
            return
        u, m, o = report["unique"], report["multiple"], report["overall"]
        line = (f"{name:17s} {u[0]:8.2f} {u[1]:8.2f} {m[0]:8.2f} {m[1]:8.2f} "
                f"{o[0]:8.2f} {o[1]:8.2f}")
        if clean is not None and report is not clean:
            line += (f" {m[0] - clean['multiple'][0]:+9.2f}"
                     f" {o[0] - clean['overall'][0]:+9.2f}")
        print(line)

    print(f"\nrun: {folder}")
    for title, table in (("controls: is the model listening?", CONTROLS),
                         ("knockout: what does the trained model rely on?", KNOCKOUTS)):
        print(f"\n=== {title} ===")
        print(header)
        print("-" * len(header))
        emit("clean", clean)
        for name in table:
            emit(name, _report(folder, os.path.join(ARCHIVE, name)))

    if clean is None:
        print("\nNOTE: the run itself has no scores.p, so no deltas could be computed.")
    print("\nDeltas are against the clean row of the same checkpoint. The silence and "
          "mismatch rows are the\nfloor: accuracy above them is what the audio contributes.\n")
    return 0


def run_list(args):
    for title, table in (("controls", CONTROLS), ("knockout", KNOCKOUTS)):
        print(f"\n--- {title} " + "-" * (60 - len(title)))
        for name, cfg in table.items():
            print(f"  {name:17s} {cfg['note']}")
            print(f"  {'':17s} {' '.join(cfg['flags'])}")
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=run_list)

    def target(p):
        p.add_argument("--folder", type=str, default=None,
                       help="Run under outputs/. Default: the newest run of --arm.")
        p.add_argument("--arm", type=str, default="spot")
        return p

    for name in GROUPS:
        group = target(sub.add_parser(name))
        group.add_argument("--only", nargs="+", default=None,
                           help=f"Subset of: {', '.join(GROUPS[name])}")
        group.add_argument("--force", action="store_true")
        group.add_argument("--dry_run", action="store_true")
        group.set_defaults(func=run_group)

    target(sub.add_parser("table")).set_defaults(func=run_table)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
