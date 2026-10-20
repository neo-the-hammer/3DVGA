# HOWTO — speech data on Colab, then training

Order: text → speech → features → train. Each notebook mirrors its output to Google Drive,
so the next one (or a recycled runtime) picks it up again.

| Step | Notebook | Runtime | Writes (Drive: `MyDrive/3DVGA/`) |
|---|---|---|---|
| 1 | `run_audio_tts.ipynb` | CPU (Piper runs on CPU either way) | `data_audio/wav/`, `manifest_<split>.json` |
| 2 | `run_audio_encode.ipynb` | GPU | `data_audio/feat/wav2vec2/{train,val}.h5` |
| 3 | `run_audio_train_eval.ipynb` | GPU | `outputs/<run>/` |
| 4 | `run_audio_analysis.ipynb` | GPU | `outputs/<run>/`, `outputs/reports/` |

## Setup (once)

1. Put the repo on Drive, or clone it into Colab. Run every notebook from the repo root.
2. Upload `data/ScanRefer_filtered_{train,val}.json` (the descriptions to be spoken).
3. Upload `cached_scenes/` (detector proposals; needed from step 3 on, not for steps 1–2).
4. First cell of each notebook installs the packages. Step 1 needs `piper-tts`.

## 1. Text → speech

- Defaults: 6 Piper voices, the last 2 held out (they speak validation only), `MIRROR_TO_DRIVE=True`,
  `WAVS_ON_DRIVE=True`.
- Speed: raise `TTS_INSTANCES` (4–8) to run parallel CPU processes. About 0.19 s per clip per
  process, so roughly 2–3 h on one process for the full corpus.
- Interrupted or recycled runtime: rerun the notebook. Existing WAVs are kept; only missing ones
  are synthesised (`OVERWRITE_WAVS=False`).
- Without the notebook: `python scripts/audio_tts/build_audioset.py --splits train val`
  (`--shard 1/4` … `--shard 4/4`, then one run without `--shard` to merge the manifests).

## 2. Speech → features

- Choose a GPU runtime. Defaults match the paper: `ENCODER="wav2vec2"` (base), layer 8, 25 Hz,
  `MAX_AUDIO_LEN=256` frames (10.24 s), six prosody channels.
- It reads the WAVs and manifests from Drive and writes one HDF5 file per split
  (`features`, `prosody`, `offsets`, `keys`). The file is written to `.tmp` and renamed when
  complete, so an interrupted run never leaves a half-written cache.
- Without the notebook: `python scripts/audio_encode/cache_audio_features.py --splits train val`.
- Keep `ENCODER`, `MAX_AUDIO_LEN`, `HELD_OUT_VOICES` and the voice list identical in all
  notebooks. A mismatch makes training reject the cache.

## 3. Train and evaluate

- Set `ARM` (`spot` is the proposed model; `python experiments/audio/fusion_arms.py list` shows the
  rest), `ENCODER="wav2vec2"`, `USE_CACHED_SCENES=True`.
- Only the acoustic branch and the fusion head train; the detector never runs. The notebook
  restores `data_audio/` and `cached_scenes/` from Drive.
- `EXTRA_ARGS` must match the flags the scene cache was built with (`--use_color --use_normal`).
- Commands: `python experiments/audio/fusion_arms.py run spot --eval`, or `run-all --core --eval`
  for the core ablation arms.

## 4. Ablations and analyses

`run_audio_analysis.ipynb` has phases A–E: train arms, evaluate, evaluation-only controls and
knockouts, sweeps, robustness. Switch phases on one at a time; the core tier is on by default.
Command-line equivalents: `experiments/audio/{fusion_arms,sweeps,interventions,analysis,robustness}.py`.

## Checks before the long runs

- `python scripts/audio/selftest.py --all_arms` — model wiring, seconds.
- `python scripts/audio/smoketest.py` — whole pipeline on a tiny corpus, about 5 minutes.
- `QUICK_TEST = True` in the notebooks builds and uses a 2-scene corpus in `data_audio/_quick`,
  never mixed with the real one.
- On first real run, check `role_peak` in the slot-role analysis. Near 1/K (here 1/6) means the
  role election stayed uniform and the `spot` vs `spot-softmax` comparison is not clean.

## Using the files later

- Copy `data_audio/` and `outputs/` from Drive to wherever you train or analyse. Training only
  needs `data_audio/feat/wav2vec2/*.h5` and `cached_scenes/`; the WAVs are needed only to
  re-encode or to run the noise and ASR robustness tests.
- Local runs without a usable GPU: add `--cpu` (very slow; for checks only).
- Environment overrides use the `VGA_*` prefix (`VGA_AUDIO_ROOT`, `VGA_AUDIO_FEAT_DIR` relocate
  the data tree); the notebooks export them from their config cell.
