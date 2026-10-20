"""
End-to-end smoke test of the 3D-VGA pipeline on a real but tiny corpus. A few minutes.

    python scripts/audio/smoketest.py            # GPU if one is usable
    python scripts/audio/smoketest.py --cpu
    python scripts/audio/smoketest.py --keep     # leave the artefacts for inspection

Where scripts/audio/selftest.py checks the model on synthetic tensors, this runs the
commands the notebooks run, in the same order, on one train and one val scene:

    text -> speech -> frozen-encoder cache        (real TTS, real encoder)
    train on cached scenes + cached audio         (does the loss fall? does it learn?)
    evaluate -> scores.p / predictions.p / diagnostics.p
    evaluation-only controls and knockouts, every analysis, every table

Nothing it writes can be mistaken for a result: the corpus lives in data_audio/_smoke/
and every run is tagged SMOKE-. Both are removed at the end unless --keep is passed.
It trains for seconds, so its accuracies mean nothing -- it checks that the pipeline runs
and that the model can fit forty utterances, not how well it grounds.
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

SMOKE_ROOT = os.path.join(REPO_ROOT, "data_audio", "_smoke")
LOG_DIR = os.path.join(REPO_ROOT, "outputs", "smoke_logs")
PREFIX = "SMOKE-"
VOICES = ["en_US-amy-medium", "en_US-lessac-medium"]
UTTERANCES = 40                # per split

RESULTS = []


def step(name, cmd, env=None, expect=0):
    """Run one command, log it, and record whether it exited as expected."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"{len(RESULTS) + 1:02d}_{name.replace(' ', '_').replace('/', '-')}.log")
    start = time.time()
    with open(log, "w") as out:
        code = subprocess.call(cmd, cwd=REPO_ROOT, env=env, stdout=out,
                               stderr=subprocess.STDOUT)
    seconds = time.time() - start
    ok = code == expect
    RESULTS.append((name, ok, seconds))
    print(f"  {'ok  ' if ok else 'FAIL'} {seconds:6.1f}s  {name}", flush=True)
    if not ok:
        print(f"       exit code {code}, expected {expect}; last lines of {log}:")
        for line in open(log).read().splitlines()[-25:]:
            print("       | " + line)
    return ok


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), 0.0))
    print(f"  {'ok  ' if condition else 'FAIL'}   --     {name}" + ("" if condition else f"  {detail}"),
          flush=True)
    return bool(condition)


def gpu_usable():
    """A real kernel launch: is_available() is True on cards the build cannot drive."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.randn(8, 8, device="cuda")
        float((x @ x).sum())
        return True
    except Exception:                                     # noqa: BLE001
        return False


def validate_commands():
    """Parse every arm, sweep point and intervention against the real argument parsers."""
    from experiments.audio import fusion_arms, interventions, sweeps
    from scripts.audio_eval import build_parser as eval_parser
    from scripts.audio_train import build_parser as train_parser

    train, evaluate = train_parser(), eval_parser()
    count = 0
    for arm in fusion_arms.ARMS:
        train.parse_args(fusion_arms.build_command(arm, "x")[2:])
        count += 1
    for name in sweeps.SWEEPS:
        for tag, extra, env, _ in sweeps._points(name):
            train.parse_args(fusion_arms.build_command("spot", "x", extra)[2:])
            unknown = [k for k in env if k[4:] not in fusion_arms_env_keys()]
            if unknown:
                raise SystemExit(f"sweep {name}: {unknown} is not a VGA_ key audio_config reads")
            count += 1
    for group in interventions.GROUPS.values():
        for cfg in group.values():
            evaluate.parse_args(fusion_arms.eval_command("run", extra=cfg["flags"])[2:])
            count += 1
    return count


def fusion_arms_env_keys():
    from lib.audio_config import _ENV_KEYS
    return set(_ENV_KEYS)


def newest(tag):
    runs = sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", f"*_{PREFIX}{tag}")))
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="Do not use the GPU.")
    parser.add_argument("--keep", action="store_true", help="Keep the smoke artefacts.")
    parser.add_argument("--fit_epochs", type=int, default=None,
                        help="Epochs for the can-it-learn run (default 20 on GPU, 3 on CPU).")
    args = parser.parse_args()

    on_gpu = (not args.cpu) and gpu_usable()
    fit_epochs = args.fit_epochs or (20 if on_gpu else 3)
    started = time.time()
    print(f"\n3D-VGA smoke test  device={'cuda' if on_gpu else 'cpu'}  "
          f"corpus={UTTERANCES} utterances per split, 1 scene each\n")

    py = sys.executable
    base = {**os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "VGA_AUDIO_ROOT": SMOKE_ROOT,
            "VGA_RUN_PREFIX": PREFIX,
            "VGA_BATCH_SIZE": "2", "VGA_NUM_SCENES": "1", "VGA_NUM_VAL_SCENES": "1",
            "VGA_NUM_WORKERS": "0", "VGA_ONLY_CACHED_AUDIO": "1",
            "VGA_HELD_OUT_VOICES": "1",
            "VGA_CPU": "0" if on_gpu else "1"}
    if not on_gpu:
        base["CUDA_VISIBLE_DEVICES"] = ""
    device_flag = ["--device", "cuda" if on_gpu else "cpu"]
    arms = os.path.join("experiments", "audio", "fusion_arms.py")
    analysis = os.path.join("experiments", "audio", "analysis.py")
    inter = os.path.join("experiments", "audio", "interventions.py")
    robust = os.path.join("experiments", "audio", "robustness.py")
    sweeps = os.path.join("experiments", "audio", "sweeps.py")
    measure = os.path.join("experiments", "audio", "measurements.py")

    print("model wiring, every arm (synthetic tensors)")
    step("selftest --all_arms", [py, "scripts/audio/selftest.py", "--all_arms", *device_flag], base)
    try:
        count = validate_commands()
        check(f"{count} arm / sweep / intervention commands parse", True)
    except SystemExit as exc:
        check("arm / sweep / intervention commands parse", False, str(exc))

    print("\ntext -> speech -> encoder features")
    from lib.audio_config import AUDIO
    encoder = AUDIO.ENCODER
    subset = ["--num_scenes", "1", "--limit", str(UTTERANCES)]
    step("build_audioset (piper)",
         [py, "scripts/audio_tts/build_audioset.py", "--splits", "train", "val",
          "--voice_list", *VOICES, *subset], base)
    step(f"cache_audio_features ({encoder})",
         [py, "scripts/audio_encode/cache_audio_features.py", "--splits", "train", "val",
          "--batch_size", "4", *subset], base)
    try:
        import h5py
        with h5py.File(os.path.join(SMOKE_ROOT, "feat", encoder, "train.h5"), "r") as h:
            keys, frames = h["keys"].shape[0], h["features"].shape
            lengths = h["offsets"][1:] - h["offsets"][:-1]
        check("feature cache holds every utterance",
              keys == UTTERANCES and frames[1] in (768, 1024) and lengths.min() > 0
              and lengths.max() <= AUDIO.MAX_AUDIO_LEN, f"keys={keys} frames={frames}")
    except Exception as exc:                              # noqa: BLE001
        check("feature cache holds every utterance", False, repr(exc))

    print("\ntrain + evaluate on cached scenes and cached audio")
    fit = {**base, "VGA_EPOCH": str(fit_epochs), "VGA_VAL_STEP": str(max(2, fit_epochs * 2)),
           "VGA_VERBOSE": "5"}
    quick = {**base, "VGA_EPOCH": "2", "VGA_VAL_STEP": "2", "VGA_VERBOSE": "5"}
    step(f"train spot, {fit_epochs} epochs + eval", [py, arms, "run", "spot", "--eval"], fit)
    spot = newest("VGA-SPOT")
    if spot:
        try:
            log = json.load(open(os.path.join(spot, "my_log.json")))
            losses, acc = log["train_loss"], log["train"]["25"]
            if on_gpu:
                # Each entry is one epoch's running loss list, so this is an early epoch
                # against a late one.
                first = sum(losses[0]) / len(losses[0])
                last = sum(losses[-1]) / len(losses[-1])
                check(f"training loss falls ({first:.2f} -> {last:.2f})", last < first)
                check(f"model fits the training utterances (Acc@0.25 {max(acc):.2f})",
                      max(acc) >= 0.30, "expected >= 0.30 on 40 utterances")
            else:
                print("        --     can-it-learn check skipped on CPU (needs ~75 iterations)")
        except Exception as exc:                          # noqa: BLE001
            check("training log is readable", False, repr(exc))
        for name in ("scores.p", "predictions.p", "diagnostics.p"):
            check(f"{name} written", os.path.isfile(os.path.join(spot, name)))
        try:
            scores = pickle.load(open(os.path.join(spot, "scores.p"), "rb"))
            check("scores cover every val utterance",
                  len(scores["ious"]) == UTTERANCES and "overall" in scores["report"],
                  f"n={len(scores['ious'])}")
        except Exception as exc:                          # noqa: BLE001
            check("scores cover every val utterance", False, repr(exc))
    step("train xattn + eval", [py, arms, "run", "xattn", "--eval"], quick)
    step("sweep point (slots=4) + eval",
         [py, sweeps, "run", "slots", "--values", "4", "--eval"], quick)

    print("\nevaluation-only experiments")
    step("controls: silence, mismatch",
         [py, inter, "controls", "--only", "silence", "mismatch"], base)
    step("knockout: no-ot, no-rsk, prosody-flat",
         [py, inter, "knockout", "--only", "no-ot", "no-rsk", "prosody-flat"], base)
    step("noise 10 dB (re-encode + eval)",
         [py, robust, "noise", "--folder", "spot", "--snr", "10", "--run"], base)
    step("speaker-ood", [py, robust, "speaker-ood", "--folder", "spot", "--per_voice"], base)

    print("\ntables and analyses")
    for name, cmd in (
            ("fusion_arms table", [arms, "table"]),
            ("interventions table", [inter, "table"]),
            ("sweeps table slots", [sweeps, "table", "slots", "--values", "4", "6"]),
            ("results-table", [analysis, "results-table", "--folders", "spot", "xattn"]),
            ("slot-roles", [analysis, "slot-roles", "--folder", "spot", "--by_outcome"]),
            ("ot-sharpness", [analysis, "ot-sharpness", "--folders", "spot", "xattn"]),
            ("gain-attribution", [analysis, "gain-attribution", "--folders", "xattn", "spot"]),
            ("error-taxonomy", [analysis, "error-taxonomy", "--folder", "spot"]),
            ("significance", [analysis, "significance", "xattn", "spot", "--iters", "500"]),
            ("complexity", [analysis, "complexity", "--folder", "spot", "--baseline", "xattn",
                            "--min_bucket", "5"]),
            ("seed-aggregate", [analysis, "seed-aggregate", "--folders", "spot", "xattn"]),
            ("measurements params", [measure, "params"]),
            ("measurements latency", [measure, "latency", *device_flag]),
    ):
        step(name, [py, *cmd], base)

    failed = [name for name, ok, _ in RESULTS if not ok]
    total = time.time() - started
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {total:.0f}s"
          + (f"\nFAILED: {', '.join(failed)}" if failed else ""))

    if args.keep or failed:
        print(f"artefacts kept: {SMOKE_ROOT}, outputs/*_{PREFIX}*, {LOG_DIR}")
    else:
        shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
        shutil.rmtree(LOG_DIR, ignore_errors=True)
        for run in glob.glob(os.path.join(REPO_ROOT, "outputs", f"*_{PREFIX}*")):
            shutil.rmtree(run, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
