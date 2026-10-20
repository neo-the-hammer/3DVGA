# 3D-VGA / SPOT — 3D visual grounding from speech

Given a point cloud and a spoken description, localise the referred object with a 3D box.
No transcript and no text encoder: frozen speech features are split into role slots and
fused with detected object proposals by slot–proposal optimal transport (SPOT).

Results are in the paper (`paper/`), not here. Method notes: `docs/`.

---

## Pipeline

Only the acoustic branch and the fusion head are trained. Both inputs are read from disk.

| Step | Script | Notebook | Writes |
|---|---|---|---|
| 1. text → speech | `scripts/audio_tts/build_audioset.py` | `run_audio_tts.ipynb` | `data_audio/wav/`, `manifest_<split>.json` |
| 2. speech → features | `scripts/audio_encode/cache_audio_features.py` | `run_audio_encode.ipynb` | `data_audio/feat/<encoder>/<split>.h5` |
| 3. train + evaluate | `scripts/audio_train.py`, `scripts/audio_eval.py` | `run_audio_train_eval.ipynb` | `outputs/<run>/` |
| 4. ablations, analyses | `experiments/audio/` | `run_audio_analysis.ipynb` | `outputs/<run>/`, `outputs/reports/` |

Detector proposals come from `cached_scenes/` (one file per scene); no detector runs.

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r req/requirements.txt
pip install piper-tts                       # step 1 only
pip install ./pointnet/pointnet2_ops_lib    # optional; only to rebuild the scene cache
```

---

## Data

```
data/ScanRefer_filtered_{train,val}.json
data/scannet/scannet_data/      ground-truth boxes (ScanNet v2, batch_load_scannet_data.py)
cached_scenes/{train,val}/      detector proposals
```

---

## Commands

```bash
python scripts/audio/selftest.py --all_arms                    # model wiring, seconds
python scripts/audio/smoketest.py                              # whole pipeline, ~5 min

python scripts/audio_tts/build_audioset.py --splits train val               # 6 Piper voices
python scripts/audio_encode/cache_audio_features.py --splits train val      # wav2vec 2.0 large

python experiments/audio/fusion_arms.py run spot --eval        # train + evaluate one arm
python experiments/audio/fusion_arms.py run-all --core --eval  # the core ablation arms
python experiments/audio/interventions.py controls             # evaluation-only checks
python experiments/audio/fusion_arms.py table
```

Speech synthesis resumes where it stopped. On CPU, several instances can share it:
`--shard 1/4` … `--shard 4/4`, then one run without `--shard` to merge the manifests.

---

## Experiments

Flags live in `lib/audio_config.py`; every runner reads shared settings from `VGA_<NAME>`.
The protocol and run order are in `docs/3DVGA-experiments.md`.

| Runner | For |
|---|---|
| `fusion_arms.py` | one training run per ablation arm |
| `sweeps.py` | one setting varied at a time |
| `interventions.py` | controls and knockouts on a trained model, no training |
| `analysis.py` | tables and diagnostics from saved runs |
| `robustness.py` | unseen speakers, noise, ASR error rate |
| `measurements.py` | parameters and latency |

---

## Archive

`Archive/` holds the text-grounding pipeline this work grew from and files kept from earlier
audio snapshots (`from_old_snapshots/`); nothing there is imported here.

---

Built on [ScanRefer](https://github.com/daveredrum/ScanRefer),
[VoteNet](https://github.com/facebookresearch/votenet) and
[3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer).
