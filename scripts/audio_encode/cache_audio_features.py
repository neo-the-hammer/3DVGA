"""
Freeze the speech encoder: turn data_audio/wav/ into one HDF5 per split.

    python scripts/audio_encode/cache_audio_features.py --splits train val --encoder wav2vec2

The encoder is frozen in every 3DVG-Audio setup (the paper's AP-Refer included), so running
it inside the training loop pays a transformer forward pass per annotation per epoch for a
tensor that never changes. Caching is the same move this repo already makes for the
detector (scripts/audio/build_scene_cache.py), and it is
what makes the fusion ablations cheap enough to sweep.

Layout -- one contiguous block plus an index, rather than one HDF5 dataset per utterance,
because ~60k small datasets makes opening the file slower than reading it:

    features  (sum_T, C)  float16   frozen encoder frames, pooled to 25 fps
    prosody   (sum_T, P)  float32   log-energy, dE, log-F0, voicing, flux, boundary
    offsets   (N + 1,)    int64     features[offsets[i]:offsets[i+1]] is utterance i
    keys      (N,)        bytes     "<scene_id>/<object_id>/<ann_id>"
    attrs                           encoder, layer, fps, dims, pooling, max_len

The prosody block is the part a text branch cannot have. It is computed from the waveform
directly (not from the encoder), because wav2vec-style models are trained to be invariant
to exactly the pitch and loudness cues that mark which word is the referent.
"""

import argparse
import json
import os
import sys
import wave

import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from lib.audio_config import (AUDIO, apply_env, encoder_ckpt, encoder_layer, feature_cache,
                              pilot_subset,
                              utterance_key, wav_path)
from lib.config import CONF


# --------------------------------------------------------------------------- prosody


def read_wav(path):
    with wave.open(path, "rb") as src:
        rate = src.getframerate()
        width = src.getsampwidth()
        frames = src.readframes(src.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, rate


def frame_signal(audio, frame_len, hop):
    """(n_frames, frame_len) view of the signal, zero-padded at the tail."""
    if audio.shape[0] < frame_len:
        audio = np.pad(audio, (0, frame_len - audio.shape[0]))
    n_frames = 1 + (audio.shape[0] - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return audio[idx]


def estimate_f0(frames, rate, fmin=60.0, fmax=400.0):
    """Autocorrelation F0 with a voicing score, one value per frame.

    Deliberately not YIN or CREPE: F0 here is a *cue*, feeding a 6-dimensional stream that
    a FiLM layer consumes, not a measurement anyone reports. Autocorrelation is exact
    enough for "is this syllable accented", needs no extra dependency, and is fast enough
    to run over 60k clips.
    """
    n_frames, frame_len = frames.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    windowed = frames * np.hanning(frame_len)[None, :]
    # Autocorrelation via FFT, per frame.
    n_fft = int(2 ** np.ceil(np.log2(2 * frame_len)))
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=1)[:, :frame_len]

    lag_min = max(1, int(rate / fmax))
    lag_max = min(frame_len - 1, int(rate / fmin))
    if lag_max <= lag_min:
        return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)

    energy = acf[:, 0:1]
    band = acf[:, lag_min:lag_max]
    best = np.argmax(band, axis=1)
    peak = band[np.arange(n_frames), best]

    # Normalised peak height is the voicing score; unvoiced frames get F0 = 0.
    voicing = np.clip(peak / (energy[:, 0] + 1e-8), 0.0, 1.0)
    f0 = rate / (best + lag_min).astype(np.float32)
    f0 = np.where(voicing > 0.3, f0, 0.0)
    return f0.astype(np.float32), voicing.astype(np.float32)


def compute_prosody(audio, rate, n_target):
    """(n_target, AUDIO.PROSODY_DIM) frame-aligned prosodic descriptors.

    Channels: log-energy, delta log-energy, log-F0, voicing, spectral flux, boundary
    strength. Boundary strength is a local dip in energy -- a proxy for the pause that
    separates the target clause from the relational clause ("the chair | next to the
    table"), which is where the slot induction downstream wants to cut.
    """
    dim = AUDIO.PROSODY_DIM
    if n_target <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    if audio.size < rate // 50:
        return np.zeros((n_target, dim), dtype=np.float32)

    frame_len = int(0.025 * rate)         # 25 ms
    hop = max(1, int(0.010 * rate))       # 10 ms -> 100 fps, resampled to n_target below
    frames = frame_signal(audio, frame_len, hop)
    if frames.shape[0] == 0:
        return np.zeros((n_target, dim), dtype=np.float32)

    energy = np.log(np.mean(frames ** 2, axis=1) + 1e-8).astype(np.float32)
    d_energy = np.gradient(energy).astype(np.float32) if energy.size > 1 else np.zeros_like(energy)

    f0, voicing = estimate_f0(frames, rate)
    log_f0 = np.log(f0 + 1.0).astype(np.float32)

    magnitude = np.abs(np.fft.rfft(frames * np.hanning(frame_len)[None, :], axis=1))
    magnitude = magnitude / (np.linalg.norm(magnitude, axis=1, keepdims=True) + 1e-8)
    flux = np.zeros(frames.shape[0], dtype=np.float32)
    if frames.shape[0] > 1:
        flux[1:] = np.linalg.norm(np.diff(magnitude, axis=0), axis=1)

    # Boundary strength: how far below the local median this frame's energy sits.
    window = min(21, energy.size if energy.size % 2 else energy.size - 1)
    if window >= 3:
        pad = window // 2
        padded = np.pad(energy, (pad, pad), mode="edge")
        local = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)
        boundary = np.clip(local - energy, 0.0, None).astype(np.float32)
    else:
        boundary = np.zeros_like(energy)

    raw = np.stack([energy, d_energy, log_f0, voicing, flux, boundary], axis=1)

    # Resample the 100 fps prosody grid onto the encoder's frame grid.
    src = np.linspace(0.0, 1.0, raw.shape[0], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, n_target, dtype=np.float64)
    out = np.stack([np.interp(dst, src, raw[:, c]) for c in range(dim)], axis=1)

    # Per-utterance standardisation of the two absolute-scale channels: speakers differ in
    # loudness and register, and the model wants "which frame is accented", not "who is
    # talking". The relative channels are already centred.
    for c in (0, 2):
        column = out[:, c]
        mask = column > (column.min() + 1e-6)
        if mask.sum() > 1:
            out[:, c] = (column - column[mask].mean()) / (column[mask].std() + 1e-6)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- encoder


def load_encoder(device):
    """Frozen speech encoder. Returns (model, feature_extractor, hidden_dim)."""
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise SystemExit(
            "caching needs torch + transformers:\n"
            "    pip install transformers soundfile\n"
            f"(import failed: {exc})") from exc

    name = encoder_ckpt()
    print(f"loading frozen encoder {name} (layer {encoder_layer()})")
    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, extractor, model.config.hidden_size


def encode_batch(model, extractor, waveforms, device, torch):
    """Run the frozen encoder over a padded batch, returning a list of (T_i, C) arrays.

    Padding is stripped per item afterwards using the encoder's own downsampling ratio, so
    a short clip in a batch with a long one does not come back with a tail of frames
    encoding silence -- which is precisely the failure the paper's fixed num_frame setting
    runs into (their Table 4: accuracy peaks and then falls as num_frame grows).
    """
    lengths = np.array([w.shape[0] for w in waveforms], dtype=np.int64)
    inputs = extractor(list(waveforms), sampling_rate=AUDIO.SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.hidden_states[encoder_layer()]      # (B, T, C)
        if getattr(model.config, "do_stable_layer_norm", False):
            # Pre-norm encoders (wav2vec 2.0 large) expose an un-normalised residual
            # stream below the last layer: values in the hundreds, a few channels carrying
            # most of the variance. Normalise each frame, as the encoder's own next layer
            # does before reading it.
            hidden = torch.nn.functional.layer_norm(hidden.float(), hidden.shape[-1:])

    if hasattr(model, "_get_feat_extract_output_lengths"):
        valid = model._get_feat_extract_output_lengths(torch.from_numpy(lengths)).numpy()
    else:                                                        # conservative fallback
        ratio = hidden.shape[1] / float(inputs["input_values"].shape[1])
        valid = np.maximum(1, np.floor(lengths * ratio)).astype(np.int64)

    hidden = hidden.float().cpu().numpy()
    return [hidden[i, :int(min(valid[i], hidden.shape[1]))] for i in range(len(waveforms))]


def pool_and_clip(frames):
    """Mean-pool by AUDIO.FRAME_POOL and clip to AUDIO.MAX_AUDIO_LEN.

    Clipping keeps the *centre* of a long utterance rather than its head: ScanRefer
    descriptions put the target noun early and the relation late, so a head crop would
    systematically drop the relational clause, which is exactly the information the
    Multiple subset needs.
    """
    stride = max(1, int(AUDIO.FRAME_POOL))
    if stride > 1 and frames.shape[0] >= stride:
        usable = (frames.shape[0] // stride) * stride
        frames = frames[:usable].reshape(-1, stride, frames.shape[1]).mean(axis=1)
    cap = int(AUDIO.MAX_AUDIO_LEN)
    if frames.shape[0] > cap:
        start = (frames.shape[0] - cap) // 2
        frames = frames[start:start + cap]
    return frames


# --------------------------------------------------------------------------- main


def add_noise(audio, snr_db, rng):
    """Additive white Gaussian noise at a given SNR, applied to the waveform.

    Before the frozen encoder, deliberately: the encoder is what confers acoustic
    robustness, so corrupting its output would test the fusion head against a perturbation
    no microphone produces.
    """
    power = float(np.mean(audio ** 2))
    if power <= 0:
        return audio
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def build_split(split, model, extractor, device, torch, batch_size=8, limit=None,
                noise_snr=None, num_scenes=-1):
    import h5py

    source = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{split}.json")
    if not os.path.isfile(source):
        raise SystemExit(f"missing {source} -- download ScanRefer into data/ first")
    annotations = pilot_subset(json.load(open(source)), num_scenes, limit)

    # De-duplicate: the split files list one entry per annotation, and a key appears once,
    # but guard anyway so a re-exported ScanRefer cannot silently double the cache.
    seen, items = set(), []
    for ann in annotations:
        key = utterance_key(ann["scene_id"], ann["object_id"], ann["ann_id"])
        if key in seen:
            continue
        seen.add(key)
        items.append((key, wav_path(ann["scene_id"], ann["object_id"], ann["ann_id"])))

    out_path = feature_cache(split)
    rng = np.random.default_rng(0xC0FFEE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[{split}] {len(items)} utterances -> {out_path}")

    # Streamed into resizable datasets rather than collected and concatenated: the train
    # split is ~6 GB of frames, and holding it twice does not fit a 12 GB Colab runtime.
    # Written beside the target and renamed at the end, so an interrupted run never
    # leaves a truncated cache where the dataset would read it.
    tmp_path = out_path + ".tmp"
    out = h5py.File(tmp_path, "w")
    feat_ds = pros_ds = None
    offsets, keys = [0], []
    missing = 0
    pending_keys, pending_waves = [], []

    def append(dataset, block):
        dataset.resize(dataset.shape[0] + block.shape[0], axis=0)
        dataset[-block.shape[0]:] = block

    def flush():
        nonlocal pending_keys, pending_waves, feat_ds, pros_ds
        if not pending_waves:
            return
        try:
            encoded = encode_batch(model, extractor, pending_waves, device, torch)
        except torch.OutOfMemoryError:
            # A batch is padded to its longest clip, so one long utterance can exceed a
            # small GPU. One clip at a time gives the same frames (padding is masked).
            torch.cuda.empty_cache()
            encoded = [encode_batch(model, extractor, [wave_data], device, torch)[0]
                       for wave_data in pending_waves]
        for key, wave_data, frames in zip(pending_keys, pending_waves, encoded):
            frames = pool_and_clip(frames)
            if frames.shape[0] == 0:
                frames = np.zeros((1, frames.shape[1]), dtype=np.float32)
            pros = (compute_prosody(wave_data, AUDIO.SAMPLE_RATE, frames.shape[0])
                    if AUDIO.USE_PROSODY
                    else np.zeros((frames.shape[0], AUDIO.PROSODY_DIM), dtype=np.float32))
            if feat_ds is None:
                feat_ds = out.create_dataset(
                    "features", shape=(0, frames.shape[1]), maxshape=(None, frames.shape[1]),
                    chunks=(256, frames.shape[1]), dtype=AUDIO.CACHE_DTYPE)
                pros_ds = out.create_dataset(
                    "prosody", shape=(0, pros.shape[1]), maxshape=(None, pros.shape[1]),
                    chunks=(1024, pros.shape[1]), dtype="float32")
            append(feat_ds, frames.astype(AUDIO.CACHE_DTYPE))
            append(pros_ds, pros.astype(np.float32))
            offsets.append(offsets[-1] + frames.shape[0])
            keys.append(key)
        pending_keys, pending_waves = [], []

    for i, (key, path) in enumerate(items):
        if not os.path.isfile(path):
            missing += 1
            continue
        try:
            audio, rate = read_wav(path)
        except Exception as exc:                          # noqa: BLE001 - log and continue
            print(f"  [{split}] unreadable {path}: {exc!r}")
            missing += 1
            continue
        if rate != AUDIO.SAMPLE_RATE:
            raise SystemExit(f"{path} is {rate} Hz; build_audioset.py writes "
                             f"{AUDIO.SAMPLE_RATE} Hz. Rebuild the WAVs.")
        if noise_snr is not None:
            audio = add_noise(audio, noise_snr, rng)
        pending_keys.append(key)
        pending_waves.append(audio)
        if len(pending_waves) >= batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"  [{split}] {i + 1}/{len(items)}  cached={len(keys)} missing={missing}",
                  flush=True)
    flush()

    if not keys:
        out.close()
        os.remove(tmp_path)
        raise SystemExit(f"[{split}] nothing cached -- run build_audioset.py first")

    features, prosody = feat_ds, pros_ds
    lengths = np.diff(np.array(offsets))
    size_gb = (features.shape[0] * features.shape[1] * features.dtype.itemsize
               + prosody.shape[0] * prosody.shape[1] * prosody.dtype.itemsize) / 1e9
    feat_dim = features.shape[1]

    with out:
        out.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
        out.create_dataset("keys", data=np.array(keys, dtype="S64"))
        out.attrs["encoder"] = AUDIO.ENCODER
        out.attrs["encoder_ckpt"] = encoder_ckpt()
        out.attrs["encoder_layer"] = encoder_layer()
        out.attrs["feat_dim"] = features.shape[1]
        out.attrs["prosody_dim"] = prosody.shape[1]
        out.attrs["frame_pool"] = AUDIO.FRAME_POOL
        out.attrs["max_audio_len"] = AUDIO.MAX_AUDIO_LEN
        out.attrs["fps"] = 50.0 / AUDIO.FRAME_POOL
        out.attrs["split"] = split
        out.attrs["noise_snr"] = "clean" if noise_snr is None else float(noise_snr)
    os.replace(tmp_path, out_path)

    print(f"[{split}] cached={len(keys)} missing={missing}  dim={feat_dim}  "
          f"frames/utt mean={lengths.mean():.1f} max={lengths.max()}  {size_gb:.2f} GB")
    if missing:
        print(f"[{split}] WARNING: {missing} annotation(s) had no WAV. The dataset treats a "
              f"missing key as one silent frame, which trains but scores near chance -- "
              f"re-run build_audioset.py if this number is not small.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--encoder", type=str, default=None, choices=list(AUDIO.ENCODER_CKPT))
    parser.add_argument("--layer", type=int, default=None,
                        help="Hidden layer to read (default: per encoder, see lib/audio_config.py).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N annotations (smoke test).")
    parser.add_argument("--num_scenes", type=int, default=-1,
                        help="Only the first N scenes per split, the same pick "
                             "audio_train.py --num_scenes makes (pilot run).")
    parser.add_argument("--no_prosody", action="store_true")
    parser.add_argument("--noise_snr", type=float, default=None,
                        help="Add white noise at this SNR (dB) before encoding. Use with "
                             "--cache_suffix so the clean cache is not overwritten.")
    parser.add_argument("--cache_suffix", type=str, default=None,
                        help="Appended to the cache filename. Write it with '=', "
                             "--cache_suffix=-snr10, or the leading dash reads as a flag.")
    args = parser.parse_args()
    apply_env()          # VGA_<NAME> overrides; the flags below still win

    if args.cache_suffix is not None:
        AUDIO.CACHE_SUFFIX = args.cache_suffix
    if args.noise_snr is not None and not AUDIO.CACHE_SUFFIX:
        raise SystemExit("--noise_snr without --cache_suffix would overwrite the clean "
                         "cache. Pass --cache_suffix=-snr<N>.")

    if args.encoder:
        AUDIO.ENCODER = args.encoder
    if args.layer is not None:
        AUDIO.ENCODER_LAYER = args.layer
    if args.no_prosody:
        AUDIO.USE_PROSODY = False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA; caching 60k clips on CPU takes hours.")
    model, extractor, dim = load_encoder(device)
    AUDIO.ENCODER_DIM = dim

    for split in args.splits:
        build_split(split, model, extractor, device, torch,
                    batch_size=args.batch_size, limit=args.limit,
                    noise_snr=args.noise_snr, num_scenes=args.num_scenes)


if __name__ == "__main__":
    main()
