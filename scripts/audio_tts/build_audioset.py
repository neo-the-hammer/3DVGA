"""
Build the 3DVG-AudioSet-style corpus: one WAV per ScanRefer annotation.

    python scripts/audio_tts/build_audioset.py --splits train val --backend piper

Reads ScanRefer_filtered_{split}.json, synthesises each ``description`` field, and writes

    data_audio/wav/<scene_id>/<object_id>_<ann_id>.wav      16 kHz, mono, 16-bit PCM
    data_audio/manifest_<split>.json                        what was synthesised, and how

which matches the spec the 3DVG-Audio paper reports for Match-TTS (Table 1). Match-TTS has
no public release, so the backend is pluggable; every backend is resampled and requantised
to the same spec, so the downstream cache does not know or care which one ran.

Synthesis is idempotent: an existing WAV is skipped unless --overwrite is passed, so an
interrupted run resumes. It is also the slowest step in the pipeline by far (~60k clips),
so it is a script and not something the data loader does.

Multi-speaker (the default) differs deliberately from the paper's single voice: with one
speaker a model can key on timbre, and the reported numbers then flatter the acoustic
branch. Voices are assigned deterministically from the annotation key, and the last
AUDIO.HELD_OUT_VOICES are reserved for val/test so a speaker-generalisation gap is
measurable rather than assumed away.
"""

import argparse
import hashlib
import json
import os
import sys
import wave

import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from lib.audio_config import AUDIO, apply_env, pilot_subset, wav_path
from lib.config import CONF


# --------------------------------------------------------------------------- backends


class TTSBackend:
    """A backend returns float32 mono audio in [-1, 1] at its own sample rate."""

    name = "base"
    native_rate = 22050

    def voices(self):
        raise NotImplementedError

    def synthesize(self, text, voice):
        raise NotImplementedError


class PiperBackend(TTSBackend):
    """Piper: small, fast, fully local neural TTS. One process per voice, reused."""

    name = "piper"
    native_rate = 22050

    #: Voice ids are resolved against the piper voice directory; override with
    #: --voice_list if a different set is installed.
    # Three female and three male, US and British. The last AUDIO.HELD_OUT_VOICES are
    # heard at validation only: one of each sex, one of each accent, and both accents
    # stay in training so the held-out gap measures the speaker and not the accent.
    DEFAULT_VOICES = [
        "en_US-amy-medium", "en_US-ryan-medium", "en_GB-jenny_dioco-medium",
        "en_US-joe-medium", "en_US-lessac-medium", "en_GB-alan-medium",
    ]

    def __init__(self, voice_list=None, model_dir=None, cuda=False, **_):
        try:
            from piper import PiperVoice  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "backend 'piper' needs the piper-tts package:\n"
                "    pip install piper-tts\n"
                f"(import failed: {exc})") from exc
        self._voice_list = voice_list or self.DEFAULT_VOICES
        self._model_dir = model_dir or AUDIO.VOICE_DIR
        if cuda:
            import onnxruntime
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                print("[piper] --cuda asked for, but this onnxruntime has no CUDA provider "
                      "(pip install onnxruntime-gpu); synthesising on CPU")
                cuda = False
            elif hasattr(onnxruntime, "preload_dlls"):
                # Finds the CUDA/cuDNN libraries that pip installed beside torch.
                onnxruntime.preload_dlls()
        self._cuda = cuda
        self._loaded = {}
        self._ensure_voices()

    def _ensure_voices(self):
        """Download any voice model that is not in the voice directory yet."""
        missing = [v for v in self._voice_list
                   if not os.path.isfile(os.path.join(self._model_dir, f"{v}.onnx"))]
        if not missing:
            return
        from pathlib import Path
        from piper.download_voices import download_voice
        os.makedirs(self._model_dir, exist_ok=True)
        for voice in missing:
            print(f"[piper] downloading voice {voice} -> {self._model_dir}", flush=True)
            download_voice(voice, Path(self._model_dir))

    def voices(self):
        return list(self._voice_list)

    def _load(self, voice):
        if voice not in self._loaded:
            from piper import PiperVoice
            # PiperVoice.load takes a model path, not a voice id.
            path = os.path.join(self._model_dir, f"{voice}.onnx")
            if self._cuda:
                # A card the CUDA provider cannot drive fails here or on the first clip
                # (seen on a GTX 1050: cuBLAS initialisation). Probe once, then stay on CPU.
                try:
                    model = PiperVoice.load(path, use_cuda=True)
                    for _ in model.synthesize("Test."):
                        pass
                    print(f"[piper] {voice}: running on {model.session.get_providers()[0]}",
                          flush=True)
                    self._loaded[voice] = model
                    return model
                except Exception as exc:                  # noqa: BLE001
                    print(f"[piper] GPU synthesis failed ({str(exc)[:90]}...); "
                          f"using the CPU for the rest of this run", flush=True)
                    self._cuda = False
            self._loaded[voice] = PiperVoice.load(path)
        return self._loaded[voice]

    def synthesize(self, text, voice):
        model = self._load(voice)
        chunks = []
        for chunk in model.synthesize(text):
            # piper >= 1.3 yields AudioChunk objects; older versions yield raw bytes.
            raw = getattr(chunk, "audio_int16_bytes", None)
            if raw is None:
                raw = bytes(chunk)
            chunks.append(np.frombuffer(raw, dtype=np.int16))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.native_rate
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        rate = getattr(self._loaded[voice].config, "sample_rate", self.native_rate)
        return audio, rate


class CoquiBackend(TTSBackend):
    """Coqui TTS (VITS). The closest public stand-in for the paper's Match-TTS."""

    name = "coqui"
    native_rate = 22050

    DEFAULT_MODEL = "tts_models/en/vctk/vits"

    def __init__(self, model_name=None, **_):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise SystemExit(
                "backend 'coqui' needs the TTS package:  pip install TTS\n"
                f"(import failed: {exc})") from exc
        self._tts = TTS(model_name or self.DEFAULT_MODEL, progress_bar=False)
        self.native_rate = int(self._tts.synthesizer.output_sample_rate)

    def voices(self):
        speakers = getattr(self._tts, "speakers", None)
        if not speakers:
            return ["default"]
        # A stable, spread-out subset rather than the first N, which are often one accent.
        step = max(1, len(speakers) // AUDIO.NUM_VOICES)
        return [speakers[i * step] for i in range(AUDIO.NUM_VOICES)]

    def synthesize(self, text, voice):
        kwargs = {} if voice == "default" else {"speaker": voice}
        audio = np.asarray(self._tts.tts(text=text, **kwargs), dtype=np.float32)
        return audio, self.native_rate


class ESpeakBackend(TTSBackend):
    """Formant synthesis via the espeak-ng binary. Robotic, but needs no model download --
    useful to smoke-test the whole pipeline before committing to 60k neural clips."""

    name = "espeak"
    native_rate = 22050

    def __init__(self, **_):
        import shutil
        if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
            raise SystemExit("backend 'espeak' needs espeak-ng on PATH "
                             "(apt-get install espeak-ng)")
        self._bin = shutil.which("espeak-ng") or shutil.which("espeak")

    def voices(self):
        # Pitch/speed variants stand in for distinct speakers.
        return [f"en+m{i}" for i in range(1, 5)] + [f"en+f{i}" for i in range(1, 5)]

    def synthesize(self, text, voice):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp = handle.name
        try:
            subprocess.run([self._bin, "-v", voice, "-w", tmp, text],
                           check=True, capture_output=True)
            with wave.open(tmp, "rb") as src:
                rate = src.getframerate()
                frames = src.readframes(src.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, rate
        finally:
            os.unlink(tmp)


class GTTSBackend(TTSBackend):
    """Online fallback. Single voice, rate-limited -- a last resort, not a default."""

    name = "gtts"
    native_rate = 24000

    def __init__(self, **_):
        try:
            import gtts  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError as exc:
            raise SystemExit("backend 'gtts' needs:  pip install gTTS soundfile\n"
                             f"(import failed: {exc})") from exc

    def voices(self):
        return ["com", "co.uk", "com.au", "ca", "ie", "co.in"]

    def synthesize(self, text, voice):
        import io
        import gtts
        import soundfile as sf
        buffer = io.BytesIO()
        gtts.gTTS(text=text, lang="en", tld=voice).write_to_fp(buffer)
        buffer.seek(0)
        audio, rate = sf.read(buffer, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, rate


class KokoroBackend(TTSBackend):
    """Kokoro-82M: a small neural TTS (StyleTTS2-based) with natural prosody, far closer to
    gTTS than the Piper voices, and fast on a GPU (a T4 does many clips per second).
    Voice ids start with the accent: a* American, b* British; second letter f/m = sex."""

    name = "kokoro"
    native_rate = 24000

    DEFAULT_VOICES = [
        "af_heart", "am_michael", "bf_emma", "bm_george",    # training: US/UK x female/male
        "af_bella", "bm_lewis",                              # held out: one US, one UK; one F, one M
    ]

    def __init__(self, voice_list=None, cuda=False, **_):
        try:
            from kokoro import KPipeline  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "backend 'kokoro' needs the kokoro package and espeak-ng:\n"
                "    pip install kokoro soundfile ; apt-get install espeak-ng\n"
                f"(import failed: {exc})") from exc
        import torch
        self._device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
        self._voice_list = voice_list or self.DEFAULT_VOICES
        self._pipelines = {}
        print(f"[kokoro] device={self._device}", flush=True)

    def voices(self):
        return list(self._voice_list)

    def _pipeline(self, voice):
        lang = voice[0]                       # 'a' American English, 'b' British English
        if lang not in self._pipelines:
            from kokoro import KPipeline
            try:
                self._pipelines[lang] = KPipeline(lang_code=lang, device=self._device)
            except TypeError:                 # older kokoro without the device argument
                self._pipelines[lang] = KPipeline(lang_code=lang)
        return self._pipelines[lang]

    def synthesize(self, text, voice):
        chunks = []
        for _, _, audio in self._pipeline(voice)(text, voice=voice, speed=1.0):
            if audio is None:
                continue
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach")
                          else np.asarray(audio))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.native_rate
        return np.concatenate(chunks).astype(np.float32), self.native_rate


BACKENDS = {
    "piper": PiperBackend,
    "kokoro": KokoroBackend,
    "coqui": CoquiBackend,
    "espeak": ESpeakBackend,
    "gtts": GTTSBackend,
}


# --------------------------------------------------------------------------- helpers


def resample_linear(audio, src_rate, dst_rate):
    """Linear resampling. Good enough here: the frozen encoder downstream is trained on
    16 kHz speech and is insensitive to the interpolation kernel at this ratio."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / float(src_rate)
    n_dst = int(round(duration * dst_rate))
    if n_dst <= 1:
        return np.zeros(0, dtype=np.float32)
    src_t = np.arange(audio.shape[0], dtype=np.float64) / src_rate
    dst_t = np.arange(n_dst, dtype=np.float64) / dst_rate
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def write_wav(path, audio, rate):
    """16-bit PCM mono, per the 3DVG-AudioSet spec."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as out:
        out.setnchannels(AUDIO.CHANNELS)
        out.setsampwidth(AUDIO.BIT_DEPTH // 8)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def pick_voice(key, voices, split):
    """Deterministic voice choice, with the last AUDIO.HELD_OUT_VOICES reserved for
    val/test. Hashing the key (not the index) keeps the assignment stable when ScanRefer
    is re-ordered or filtered."""
    held_out = AUDIO.HELD_OUT_VOICES if len(voices) > AUDIO.HELD_OUT_VOICES else 0
    pool = voices[:-held_out] if (split == "train" and held_out) else voices
    if not pool:
        pool = voices
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return pool[int.from_bytes(digest[:4], "big") % len(pool)]


def normalise_text(description):
    """ScanRefer descriptions are lower-cased, loosely punctuated and occasionally empty.
    TTS front-ends handle that badly -- an empty string yields a zero-length WAV that then
    poisons the feature cache -- so clean it up here, once, rather than in three places."""
    text = " ".join(str(description).split())
    text = text.replace(" .", ". ").replace(" ,", ", ")
    text = " ".join(text.split())
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:]


# --------------------------------------------------------------------------- main


def synthesize_one(backend, key, path, text, voice):
    """Synthesise one annotation and write its WAV. Returns its manifest entry."""
    try:
        audio, rate = backend.synthesize(text, voice)
        audio = resample_linear(audio, rate, AUDIO.SAMPLE_RATE)
        if audio.size < AUDIO.SAMPLE_RATE // 10:          # < 100 ms is a synthesis failure
            raise RuntimeError(f"degenerate output, {audio.size} samples")
        write_wav(path, audio, AUDIO.SAMPLE_RATE)
    except Exception as exc:                              # noqa: BLE001 - log and continue
        return {"key": key, "voice": voice, "error": repr(exc)}
    return {"key": key, "voice": voice, "text": text,
            "duration": round(audio.size / AUDIO.SAMPLE_RATE, 3)}


def shard_manifests(split):
    """Entries written by --shard instances, so the final manifest keeps their durations."""
    import glob
    entries = {}
    for path in glob.glob(AUDIO.MANIFEST.format(split=split) + ".shard*"):
        try:
            entries.update({e["key"]: e for e in json.load(open(path))["entries"]})
        except (ValueError, KeyError):
            pass
    return entries


def build_split(split, backend, overwrite=False, limit=None, num_scenes=-1, shard=None):
    source = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{split}.json")
    if not os.path.isfile(source):
        raise SystemExit(f"missing {source} -- download ScanRefer into data/ first")
    annotations = pilot_subset(json.load(open(source)), num_scenes, limit)

    voices = backend.voices()
    print(f"[{split}] {len(annotations)} annotations, {len(voices)} voice(s), "
          f"backend={backend.name}")

    # A resumed run skips the WAVs that exist, so start from the previous manifest or the
    # voice of every clip synthesised before the interruption is lost -- and the
    # speaker-OOD split is read from this file.
    manifest_path = AUDIO.MANIFEST.format(split=split)
    previous = {}
    if os.path.isfile(manifest_path) and not overwrite:
        try:
            previous = {e["key"]: e for e in json.load(open(manifest_path))["entries"]}
        except (ValueError, KeyError):
            previous = {}
    if not overwrite:
        previous.update(shard_manifests(split))
    if shard:
        # Several instances at once: each takes every N-th annotation and keeps its own
        # manifest; a final run without --shard merges them into manifest_<split>.json.
        index, count = shard
        annotations = annotations[index::count]
        manifest_path += f".shard{index}of{count}"
        print(f"[{split}] shard {index + 1} of {count}: {len(annotations)} annotations")

    manifest, todo = [], []
    made = skipped = failed = 0
    for ann in annotations:
        scene_id, object_id, ann_id = ann["scene_id"], ann["object_id"], ann["ann_id"]
        key = f"{scene_id}/{object_id}/{ann_id}"
        path = wav_path(scene_id, object_id, ann_id)

        if os.path.isfile(path) and not overwrite:
            skipped += 1
            # The voice is a pure function of the key, so it can be restated even when the
            # earlier run died before writing its manifest.
            manifest.append(previous.get(key) or {
                "key": key, "voice": pick_voice(key, voices, split),
                "text": normalise_text(ann["description"])})
            continue

        text = normalise_text(ann["description"])
        if not text:
            # No text means no audio; recording it keeps the count honest instead of
            # letting the cache builder trip over a missing file later.
            failed += 1
            manifest.append({"key": key, "error": "empty description"})
            continue
        todo.append((key, path, text, pick_voice(key, voices, split)))

    for i, task in enumerate(todo):
        entry = synthesize_one(backend, *task)
        manifest.append(entry)
        if "error" in entry:
            failed += 1
        else:
            made += 1
        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i + 1}/{len(todo)}  made={made} failed={failed}", flush=True)

    os.makedirs(AUDIO.ROOT, exist_ok=True)
    if not shard:
        seen = {e["key"] for e in manifest}
        manifest += [e for k, e in previous.items() if k not in seen]
    with open(manifest_path, "w") as out:
        json.dump({"backend": backend.name, "sample_rate": AUDIO.SAMPLE_RATE,
                   "voices": voices, "entries": manifest}, out)

    if not shard:
        import glob
        for stale in glob.glob(manifest_path + ".shard*"):      # merged above
            os.remove(stale)

    durations = [e["duration"] for e in manifest if "duration" in e]
    print(f"[{split}] made={made} skipped={skipped} failed={failed}")
    if durations:
        print(f"[{split}] duration mean={np.mean(durations):.2f}s "
              f"median={np.median(durations):.2f}s p95={np.percentile(durations, 95):.2f}s "
              f"(3DVG-AudioSet reports a 4.25 s mean)")
    print(f"[{split}] manifest -> {manifest_path}")
    if failed:
        print(f"[{split}] WARNING: {failed} annotation(s) have no WAV; "
              f"cache_audio_features.py will report them as missing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--backend", type=str, default=None, choices=list(BACKENDS),
                        help=f"TTS backend (default: AUDIO.TTS_BACKEND={AUDIO.TTS_BACKEND})")
    parser.add_argument("--voice_list", nargs="+", default=None,
                        help="Override the backend's voice ids.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory holding the piper voice models; missing voices are "
                             f"downloaded into it (default: {AUDIO.VOICE_DIR}).")
    parser.add_argument("--model_name", type=str, default=None, help="Model id (coqui).")
    parser.add_argument("--overwrite", action="store_true", help="Re-synthesise existing WAVs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N annotations per split (smoke test).")
    parser.add_argument("--num_scenes", type=int, default=-1,
                        help="Only the first N scenes per split, the same pick "
                             "audio_train.py --num_scenes makes (pilot run).")
    parser.add_argument("--shard", type=str, default=None, metavar="I/N",
                        help="Run as instance I of N (1-based), e.g. 1/4 .. 4/4 in four "
                             "terminals; then run once without --shard to merge.")
    parser.add_argument("--cuda", action="store_true",
                        help="Run piper on the GPU (needs onnxruntime-gpu).")
    args = parser.parse_args()
    apply_env()          # VGA_<NAME> overrides; the flags below still win

    name = args.backend or AUDIO.TTS_BACKEND
    backend = BACKENDS[name](voice_list=args.voice_list, model_dir=args.model_dir,
                             model_name=args.model_name, cuda=args.cuda)

    shard = None
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 1 <= index <= count:
            raise SystemExit(f"--shard {args.shard}: want I/N with 1 <= I <= N")
        shard = (index - 1, count)

    for split in args.splits:
        build_split(split, backend, overwrite=args.overwrite, limit=args.limit,
                    num_scenes=args.num_scenes, shard=shard)


if __name__ == "__main__":
    main()
