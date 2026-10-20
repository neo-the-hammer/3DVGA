"""
Complexity and latency of the 3D-VGA branch.

    python experiments/audio/measurements.py params
    python experiments/audio/measurements.py latency --use_color --use_normal
    python experiments/audio/measurements.py deployment --use_color --use_normal

Counterpart of experiments/measurements.py on the text branch. Three questions:

  params      Where the parameters are. The paper's pitch for AP-Refer is that it is
              "simple"; if SPOT buys its accuracy with ten times the fusion head, that
              belongs in the comparison.

  latency     Forward-pass time per utterance, by module, on synthetic tensors — no
              dataset needed. The detector dominates, which is the point: the fusion
              head's cost has to be read against that, not in isolation.

  deployment  The number that matters for the task's own motivation. 3DVG-Audio exists
              for hands-free use, so the honest figure is end-to-end from waveform to box,
              INCLUDING the frozen encoder — which is cached away during training and
              cannot be during deployment. It also prices the ASR-cascade alternative,
              since the cascade pays an extra recogniser pass the direct model does not.
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# lib/config.py lists the ScanNet scene directory at import time; an empty directory is
# enough, and this keeps `params` runnable on a machine with no dataset.
os.makedirs(os.path.join(REPO_ROOT, "data", "scannet", "scans"), exist_ok=True)

from lib.audio_config import AUDIO, add_arguments as add_audio_arguments  # noqa: E402
from lib.audio_config import apply as apply_audio  # noqa: E402


def _torch():
    try:
        import torch
    except ImportError as exc:                                    # pragma: no cover
        raise SystemExit(f"these measurements need torch (import failed: {exc})") from exc
    return torch


def _count(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _build_audio_side(device):
    """AudioModule + AudioMatchModule only — no detector, so this needs no CUDA kernels."""
    _torch()                                   # fail early with a useful message
    from models.audio_match_module import AudioMatchModule
    from models.audio_module import AudioModule

    class Args:
        detector = "GF"

    audio = AudioModule(num_text_classes=18, in_dim=AUDIO.ENCODER_DIM).to(device).eval()
    match = AudioMatchModule(Args(), num_proposals=256, det_channel=288).to(device).eval()
    return audio, match


def run_params(args):
    audio, match = _build_audio_side("cpu")

    rows = [
        ("AudioModule", audio),
        ("  encoder (PA-AFE)", audio.encoder),
        ("    FiLM", audio.encoder.film if AUDIO.USE_PROSODY else None),
        ("  slot induction", audio.slots),
        ("  AudioCls head", audio.audio_cls),
        ("AudioMatchModule", match),
        ("  features_concat", match.features_concat),
        ("  scene self-attn", match.self_attn),
        ("  SpatialGraphNet", match.graph),
        ("  fusion", match.fusion),
        ("  scoring head", match.match),
    ]
    if AUDIO.FUSION == "spot" and AUDIO.USE_RSK:
        rows.append(("    RSK hypernet", match.fusion.rsk))

    print(f"\nfusion='{AUDIO.FUSION}'  slots={AUDIO.NUM_SLOTS}  rsk={AUDIO.USE_RSK}  "
          f"prosody={AUDIO.USE_PROSODY}")
    print(f"\n{'module':28s} {'params':>12s} {'share':>8s}")
    print("-" * 50)
    grand = _count(audio)[0] + _count(match)[0]
    for name, module in rows:
        if module is None:
            continue
        total = _count(module)[0]
        print(f"{name:28s} {total:12,d} {100.0 * total / grand:7.1f}%")
    print("-" * 50)
    print(f"{'audio branch + fusion':28s} {grand:12,d}")

    print(f"\nFrozen and NOT counted above: the speech encoder "
          f"({AUDIO.ENCODER}, ~95M params).")
    print("It is cached to disk, so it costs nothing per training step — but it is part of")
    print("the deployed system, and `deployment` below prices it.")

    print("\nFor the comparison a reader wants, run this with --audio_fusion concat and")
    print("--audio_fusion xattn too: that is AP-Refer's fusion and the attentive baseline")
    print("under identical surrounding code.")
    return 0


def run_latency(args):
    torch = _torch()
    device = torch.device(args.device)
    audio, match = _build_audio_side(device)

    batch, lang = args.batch, args.lang
    flat, frames, proposals = batch * lang, AUDIO.MAX_AUDIO_LEN, 256

    def fresh():
        return {
            "audio_feat_list": torch.randn(batch, lang, frames, AUDIO.ENCODER_DIM,
                                           device=device),
            "prosody_list": torch.randn(batch, lang, frames, AUDIO.PROSODY_DIM,
                                        device=device),
            "audio_len_list": torch.full((batch, lang), frames, dtype=torch.long,
                                         device=device),
            "audio_cat_list": torch.zeros(batch, lang, dtype=torch.long, device=device),
            "features": torch.randn(batch, 288, proposals, device=device),
            "center": torch.randn(batch, proposals, 3, device=device) * 2.0,
            "objectness_scores": torch.stack([
                torch.zeros(batch, proposals, device=device),
                torch.ones(batch, proposals, device=device)], dim=-1),
            "istrain": torch.zeros(batch, device=device),
            "lang_num": torch.full((batch,), lang, dtype=torch.long, device=device),
        }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def timed(fn, iters, warmup):
        for _ in range(warmup):
            fn()
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        return (time.perf_counter() - start) / iters

    with torch.no_grad():
        data = fresh()
        t_audio = timed(lambda: audio(dict(data)), args.iters, args.warmup)
        primed = audio(dict(data))
        t_match = timed(lambda: match(dict(primed)), args.iters, args.warmup)

    per_utt_audio = t_audio / flat * 1000
    per_utt_match = t_match / flat * 1000

    print(f"\ndevice={device}  batch={batch}×{lang}={flat} utterances  "
          f"frames={frames}  proposals={proposals}")
    print(f"fusion='{AUDIO.FUSION}'  slots={AUDIO.NUM_SLOTS}  rsk={AUDIO.USE_RSK}")
    print(f"\n{'stage':24s} {'per batch (ms)':>16s} {'per utterance (ms)':>20s}")
    print("-" * 62)
    print(f"{'AudioModule':24s} {t_audio * 1000:16.2f} {per_utt_audio:20.3f}")
    print(f"{'AudioMatchModule':24s} {t_match * 1000:16.2f} {per_utt_match:20.3f}")
    print("-" * 62)
    print(f"{'audio side total':24s} {(t_audio + t_match) * 1000:16.2f} "
          f"{per_utt_audio + per_utt_match:20.3f}")

    print("\nThe detector is NOT in this measurement — it needs real point clouds and "
          "CUDA ops.")
    print("Run experiments/measurements.py model for it. The reason to read them together:")
    print("the detector dominates, so a fusion head that doubles in cost may not move the")
    print("end-to-end number at all — and claiming otherwise either way needs both halves.")
    return 0


def run_deployment(args):
    torch = _torch()
    device = torch.device(args.device)

    print("\nEnd-to-end deployment cost, per spoken query.")
    print("Training caches the frozen encoder; deployment cannot, so it is counted here.\n")

    try:
        from transformers import AutoModel
        name = AUDIO.ENCODER_CKPT[AUDIO.ENCODER]
        encoder = AutoModel.from_pretrained(name).eval().to(device)
        audio_in = torch.randn(1, int(4.25 * AUDIO.SAMPLE_RATE), device=device)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(args.warmup):
                encoder(audio_in)
            sync()
            start = time.perf_counter()
            for _ in range(args.iters):
                encoder(audio_in)
            sync()
            t_encoder = (time.perf_counter() - start) / args.iters
        print(f"  frozen encoder ({AUDIO.ENCODER}, 4.25 s clip) : "
              f"{t_encoder * 1000:8.2f} ms")
        params = sum(p.numel() for p in encoder.parameters())
        print(f"  its parameters                               : {params:,}")
    except ImportError:
        print("  frozen encoder: transformers not installed, skipped")
        t_encoder = None

    audio, match = _build_audio_side(device)
    n_branch = _count(audio)[0] + _count(match)[0]
    print(f"  audio branch + fusion parameters             : {n_branch:,}")

    print("\nThe cascade alternative pays, per query:")
    print("  ASR forward pass          (a second encoder-sized model)")
    print("  + text tokenisation and embedding lookup")
    print("  + the LLM parse, if the text model uses one — which this repository's does,")
    print("    and which is an API round-trip unless a local parser is substituted")
    print("  + the text grounding model")
    print("\nThe direct model pays one encoder pass and one fusion head. That difference is")
    print("the deployment argument for 3DVG-Audio, and it is worth measuring rather than")
    print("asserting — experiments/audio/robustness.py asr-cascade builds the baseline,")
    print("and experiments/measurements.py parsing-latency already times the parse.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--device", default=None)
        p.add_argument("--iters", type=int, default=30)
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--lang", type=int, default=8)
        p.add_argument("--use_color", action="store_true")
        p.add_argument("--use_normal", action="store_true")
        add_audio_arguments(p)
        return p

    common(sub.add_parser("params")).set_defaults(func=run_params)
    common(sub.add_parser("latency")).set_defaults(func=run_latency)
    common(sub.add_parser("deployment")).set_defaults(func=run_deployment)

    args = parser.parse_args()
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    apply_audio(args)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
