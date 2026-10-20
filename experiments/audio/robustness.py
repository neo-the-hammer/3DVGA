"""
Robustness and baseline experiments the fusion ablation does not cover.

    python experiments/audio/robustness.py speaker-ood   --folder <run>
    python experiments/audio/robustness.py noise         --folder <run> --snr 20 10 5 0
    python experiments/audio/robustness.py asr-cascade   --build
    python experiments/audio/robustness.py asr-cascade   --wer

Three questions, none of which the ablation table answers:

  speaker-ood   Did the model learn the content, or the synthesiser? Two of the eight
                voices never appear in training (AUDIO.HELD_OUT_VOICES), and the manifest
                records which voice produced every clip, so the val split partitions into
                seen and unseen speakers with no retraining at all. A model that keyed on
                timbre falls off a cliff here; one that learned the words does not.

  noise         TTS audio is clean in a way no deployment is. Noise is added to the
                waveform BEFORE the frozen encoder — adding it to cached features would
                measure nothing, since the encoder is what confers acoustic robustness.

  asr-cascade   The obvious alternative, and the first thing a reviewer proposes:
                transcribe, then run the existing text model. This builds that baseline
                honestly — real ASR output, word-error rate reported — so the comparison
                is against a system somebody would actually build, not a strawman.

All three run against a finished checkpoint. None needs retraining except the cascade,
which needs no training at all.
"""

import argparse
import collections
import json
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from lib.audio_config import AUDIO, utterance_key, wav_path  # noqa: E402
from lib.config import CONF  # noqa: E402


# --------------------------------------------------------------------- speaker-OOD


def load_manifest(split):
    path = AUDIO.MANIFEST.format(split=split)
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path} — run scripts/audio_tts/build_audioset.py first")
    return json.load(open(path))


def run_speaker_ood(args):
    """Partition the val metric by whether the speaker was seen in training."""
    train_manifest = load_manifest("train")
    val_manifest = load_manifest("val")

    seen_voices = {e["voice"] for e in train_manifest["entries"] if "voice" in e}
    voice_of = {e["key"]: e["voice"] for e in val_manifest["entries"] if "voice" in e}
    held_out = sorted({v for v in voice_of.values()} - seen_voices)

    print(f"\nvoices in training : {len(seen_voices)}  {sorted(seen_voices)}")
    print(f"held out of training: {len(held_out)}  {held_out}")
    if not held_out:
        raise SystemExit(
            "No val voice is absent from training, so there is no speaker gap to measure.\n"
            "Set AUDIO.HELD_OUT_VOICES > 0 and re-run build_audioset.py.")

    diag_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "diagnostics.p")
    preds_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions.p")
    if not os.path.isfile(preds_path):
        raise SystemExit(f"missing {preds_path} — run scripts/audio_eval.py --folder "
                         f"{args.folder} --force first")
    with open(preds_path, "rb") as handle:
        predictions = pickle.load(handle)

    # predictions is scene -> object -> ann -> {iou, bbox}; the manifest key is the same
    # triple, so the two join without any ordering assumption.
    buckets = collections.defaultdict(list)
    missing = 0
    for scene, objects in predictions.items():
        for object_id, anns in objects.items():
            for ann_id, record in anns.items():
                key = utterance_key(scene, str(object_id), str(ann_id))
                voice = voice_of.get(key)
                if voice is None:
                    missing += 1
                    continue
                buckets["unseen" if voice in held_out else "seen"].append(record["iou"])
                buckets[f"voice:{voice}"].append(record["iou"])

    def rates(values):
        if not values:
            return 0.0, 0.0, 0
        at25 = 100.0 * sum(1 for v in values if v >= 0.25) / len(values)
        at50 = 100.0 * sum(1 for v in values if v >= 0.5) / len(values)
        return at25, at50, len(values)

    print(f"\n{'group':22s} {'Acc@0.25':>10s} {'Acc@0.5':>10s} {'n':>8s}")
    print("-" * 54)
    seen, unseen = rates(buckets["seen"]), rates(buckets["unseen"])
    print(f"{'seen speakers':22s} {seen[0]:10.2f} {seen[1]:10.2f} {seen[2]:8d}")
    print(f"{'UNSEEN speakers':22s} {unseen[0]:10.2f} {unseen[1]:10.2f} {unseen[2]:8d}")
    print("-" * 54)
    print(f"{'gap (seen − unseen)':22s} {seen[0] - unseen[0]:10.2f} "
          f"{seen[1] - unseen[1]:10.2f}")

    if args.per_voice:
        print()
        for name in sorted(k for k in buckets if k.startswith("voice:")):
            at25, at50, n = rates(buckets[name])
            marker = "  ← held out" if name.split(":", 1)[1] in held_out else ""
            print(f"{name:22s} {at25:10.2f} {at50:10.2f} {n:8d}{marker}")

    if missing:
        print(f"\n{missing} prediction(s) had no manifest entry and were skipped.")

    print("\nRead it this way: a large gap means the acoustic branch is keying on timbre,")
    print("and the headline number is inflated by the synthesiser. A small gap is the")
    print("claim that multi-speaker synthesis was worth the extra corpus build.")
    if os.path.isfile(diag_path):
        print(f"\n(diagnostics.p is present; experiments/audio/analysis.py slot-roles will")
        print(f" tell you WHICH part of the pipeline degrades on unseen speakers.)")
    return 0


# --------------------------------------------------------------------- noise


def run_noise(args):
    """Score a trained run on noisy audio, one feature cache per SNR.

    Without --run this prints the commands; with it, each SNR builds its val cache and is
    scored into ``<run>/noise/snr<N>/``, so the clean scores are not overwritten.
    """
    import subprocess

    from experiments.audio.fusion_arms import eval_command

    print("\nNoise is injected into the WAVEFORM, before the frozen encoder.")
    print("Injecting it into cached features would measure nothing: the encoder is")
    print("exactly what confers acoustic robustness, so corrupting its output tests")
    print("the fusion head against a perturbation no real microphone produces.\n")

    def name(snr):
        return f"snr{snr:g}"

    failed = []
    for snr in args.snr:
        suffix = f"-{name(snr)}"
        build = [sys.executable, os.path.join("scripts", "audio_encode", "cache_audio_features.py"),
                 "--splits", "val", "--encoder", AUDIO.ENCODER,
                 "--noise_snr", f"{snr:g}", f"--cache_suffix={suffix}"]   # '=': it starts with '-'
        score = eval_command(args.folder, out_subdir=os.path.join("noise", name(snr)))
        print(f"# SNR {snr:g} dB")
        print("$", " ".join(build))
        print(f"$ VGA_CACHE_SUFFIX={suffix}", " ".join(score), "\n", flush=True)
        if not args.run:
            continue
        code = subprocess.call(build, cwd=REPO_ROOT)
        if code == 0:
            code = subprocess.call(score, cwd=REPO_ROOT,
                                   env={**os.environ, "VGA_CACHE_SUFFIX": suffix})
        if code != 0:
            failed.append(name(snr))

    def report(subdir=""):
        path = os.path.join(CONF.PATH.OUTPUT, args.folder, subdir, "scores.p")
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle).get("report")

    print(f"{'condition':12s} {'U@0.25':>8s} {'M@0.25':>8s} {'O@0.25':>8s} {'O@0.5':>8s}")
    for label, rep in [("clean", report())] + [
            (f"{snr:g} dB", report(os.path.join("noise", name(snr)))) for snr in args.snr]:
        if rep is None:
            print(f"{label:12s} {'-- not evaluated --':>35s}")
        else:
            print(f"{label:12s} {rep['unique'][0]:8.2f} {rep['multiple'][0]:8.2f} "
                  f"{rep['overall'][0]:8.2f} {rep['overall'][1]:8.2f}")
    if not args.run:
        print("\npass --run to execute the commands above.")
    if failed:
        print(f"\nfailed: {', '.join(failed)}")
    return 1 if failed else 0


# --------------------------------------------------------------------- ASR cascade


def _transcribe_split(split, limit=None):
    """Run CTC decoding over the split's WAVs, returning {key: transcript}."""
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
    except ImportError as exc:
        raise SystemExit("the cascade baseline needs torch + transformers\n"
                         f"(import failed: {exc})") from exc
    import wave

    import numpy as np

    name = "facebook/wav2vec2-base-960h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading ASR model {name} on {device}")
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForCTC.from_pretrained(name).eval().to(device)

    source = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{split}.json")
    annotations = json.load(open(source))
    if limit:
        annotations = annotations[:limit]

    out = {}
    for i, ann in enumerate(annotations):
        path = wav_path(ann["scene_id"], ann["object_id"], ann["ann_id"])
        if not os.path.isfile(path):
            continue
        with wave.open(path, "rb") as src:
            audio = np.frombuffer(src.readframes(src.getnframes()),
                                  dtype=np.int16).astype(np.float32) / 32768.0
        inputs = processor(audio, sampling_rate=AUDIO.SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        text = processor.batch_decode(logits.argmax(-1))[0].lower().strip()
        out[utterance_key(ann["scene_id"], ann["object_id"], ann["ann_id"])] = text
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(annotations)}", flush=True)
    return out


def _words_only(text):
    import re
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", str(text).lower()).split())


def _word_error_rate(reference, hypothesis):
    """Levenshtein distance over words, normalised by reference length."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def run_asr_cascade(args):
    out_dir = os.path.join(AUDIO.ROOT, "asr")
    os.makedirs(out_dir, exist_ok=True)
    transcript_path = os.path.join(out_dir, f"transcripts_{args.split}.json")

    if args.build:
        transcripts = _transcribe_split(args.split, limit=args.limit)
        with open(transcript_path, "w") as handle:
            json.dump(transcripts, handle)
        print(f"\n{len(transcripts)} transcript(s) -> {transcript_path}")

        # Emit a ScanRefer-shaped file with descriptions replaced by ASR output, so the
        # EXISTING text pipeline can consume it with no code change at all.
        source = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{args.split}.json")
        annotations = json.load(open(source))
        replaced, dropped = 0, 0
        for ann in annotations:
            key = utterance_key(ann["scene_id"], ann["object_id"], ann["ann_id"])
            text = transcripts.get(key)
            if not text:
                dropped += 1
                continue
            ann["description"] = text
            ann["token"] = text.split()
            replaced += 1
        shaped = os.path.join(out_dir, f"ScanRefer_asr_{args.split}.json")
        with open(shaped, "w") as handle:
            json.dump(annotations, handle)
        print(f"{replaced} replaced, {dropped} left as ground-truth text -> {shaped}")
        print("\nNext: run a text-grounding model over this file to get the cascade's accuracy,")
        print("then train/eval the TEXT model on it. That run is the cascade baseline.")
        return 0

    if args.wer:
        if not os.path.isfile(transcript_path):
            raise SystemExit(f"missing {transcript_path} — run with --build first")
        transcripts = json.load(open(transcript_path))
        source = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{args.split}.json")
        rates, lengths = [], []
        for ann in json.load(open(source)):
            key = utterance_key(ann["scene_id"], ann["object_id"], ann["ann_id"])
            hyp = transcripts.get(key)
            if hyp is None:
                continue
            # CTC output has no punctuation, so score words only: otherwise every full
            # stop in the reference counts as a deleted word.
            ref, hyp = _words_only(ann["description"]), _words_only(hyp)
            rates.append(_word_error_rate(ref, hyp))
            lengths.append(len(ref.split()))

        if not rates:
            raise SystemExit("no transcript matched an annotation")
        rates.sort()
        mean = sum(rates) / len(rates)
        print(f"\nASR word-error rate over {len(rates)} utterances")
        print(f"  mean   {mean * 100:.2f} %")
        print(f"  median {rates[len(rates) // 2] * 100:.2f} %")
        print(f"  p90    {rates[int(len(rates) * 0.9)] * 100:.2f} %")
        print(f"  perfect (WER 0) {100.0 * sum(1 for r in rates if r == 0) / len(rates):.1f} %")
        print(f"  mean reference length {sum(lengths) / len(lengths):.1f} words")
        print("\nThis number is the cascade's error floor. Every word the recogniser drops")
        print("is a word the parser never sees and the grounding model never recovers —")
        print("and object nouns are exactly the low-frequency words ASR gets wrong.")
        return 0

    raise SystemExit("pass --build or --wer")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ood = sub.add_parser("speaker-ood")
    ood.add_argument("--folder", required=True, help="Run under outputs/ that was evaluated.")
    ood.add_argument("--per_voice", action="store_true", help="Break down by voice.")
    ood.set_defaults(func=run_speaker_ood)

    noise = sub.add_parser("noise")
    noise.add_argument("--folder", required=True)
    noise.add_argument("--snr", type=float, nargs="+", default=[20, 10, 5, 0],
                       help="Signal-to-noise ratios in dB.")
    noise.add_argument("--run", action="store_true",
                       help="Build each noisy cache and score it, instead of only "
                            "printing the commands.")
    noise.set_defaults(func=run_noise)

    asr = sub.add_parser("asr-cascade")
    asr.add_argument("--split", default="val")
    asr.add_argument("--build", action="store_true", help="Transcribe and emit the dataset.")
    asr.add_argument("--wer", action="store_true", help="Report word-error rate.")
    asr.add_argument("--limit", type=int, default=None)
    asr.set_defaults(func=run_asr_cascade)

    args = parser.parse_args()
    if getattr(args, "folder", None):
        from experiments.audio.fusion_arms import resolve_run
        args.folder = resolve_run(args.folder)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
