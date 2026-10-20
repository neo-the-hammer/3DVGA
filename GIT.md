# Git history — branch `audio-spot`

Figure-only commits (`634547c` … `a052897`) are left out.

```
0f07fa6  2026-09-15  Init                                        text pipeline (LM3DVG)
821836a  2026-10-10  Some Changes                                adds the FRESH / OLD audio snapshots
7aa6bec  2026-10-11  Add 3D-VGA / SPOT audio pipeline            port into the main tree
08623b6  2026-10-11  6 Piper voices, separate TTS/encode, archive text pipeline
6752de5  2026-10-11  Default to wav2vec 2.0 base, 256-frame cap, align paper and handoff
```

`main` stops at `821836a`; `audio-spot` carries the rest. Line counts below are from `git show --stat`.

---

## 1. Starting point: the text model (LM3DVG) — `0f07fa6`, `821836a`

### `0f07fa6` — Init (91 files, +30 804)
The text pipeline, "Enhancing 3D Visual Grounding through Semantic-Guided Attention and Graph
Networks with LLM-Based Sentence Parsing", evaluated on ScanRefer and Sr3D. A language model
splits each description into three fields (target, adjectives, neighbours) offline; GloVe + GRU
encode them; a match module fuses them with detected proposals by attention and a spatial graph.

| Area | Files | What they do |
|---|---|---|
| Detector | `models/backbone_module.py`, `voting_module.py`, `proposal_module.py`, `models/detr/`, `models/group_free/` | PointNet++ backbone, Hough voting, proposal heads (VoteNet-style, DETR or Group-Free decoder) |
| Language side | `models/lang_module.py`, `match_module.py`, `refnet.py` | GloVe + GRU, the fusion head (LGA, A2F, TAF attention), and `RefNet` that wires everything |
| Data | `lib/dataset.py`, `lib/enet.py`, `lib/projection.py`, `scripts/multiview_features/` | ScanRefer dataset, ENet multiview features |
| Training / eval | `scripts/ScanRefer_{train,eval}.py`, `scripts/predict.py`, `lib/solver.py`, `lib/loss_helper.py`, `lib/eval_helper.py`, `lib/ap_helper.py` | training loop, evaluation, AP computation, benchmark predictions |
| Experiments | `experiments/ablation/` (`train_arms`, `parsing`, `parsing_smalllm`, `cache`, `sweep_attention_layers`, `ablation_config`, `ablation_hooks`), `experiments/analysis.py`, `experiments/measurements.py` | parser swaps (GPT, LLaMA, spaCy, small LMs), arms, sweeps, analyses, latency |
| Notebooks | `run_analysis.ipynb`, `results_analysis.ipynb`, `run_parse_quality.ipynb`, viewers | run and read the text experiments |
| Kernels | `pointnet/` (CUDA ops, a pure-Python tier, an `adv` tier) | PointNet++ operators with automatic fallback |

### `821836a` — Some Changes (178 files, +122 431)
Added the sources the port was made from: `3DVG-AUDIO -- FRESH/` (the finished audio code),
`3DVG -- AUDIO -- OLD/` (an older audio attempt, with a trained text-model run in `outputs/`),
`Z ETC Z/` (scratch notebooks), `HANDOFF.md` and `PROMPT.md`.

---

## 2. `7aa6bec` — the audio pipeline enters the main tree
76 files, +24 253 / −11. The FRESH snapshot's code was copied into the main layout and the new
pieces were wired to the existing PointNet++/VoteNet/DETR detector. The text pipeline stayed in place.

### Model (`models/`)
| File | Lines | Role |
|---|---|---|
| `audio_encoder.py` | 169 | `AudioFrameEncoder` (PA-AFE). Takes frozen wav2vec2 frames and six prosody channels. Projects to the working width (128), adds a normalised-time positional encoding, then a `FiLM` layer conditioned on prosody (zero-initialised, so it starts as identity), two self-attention layers and a saliency head. Prosody is injected here because speech encoders are trained to be invariant to exactly the pitch and loudness cues that mark the referent. |
| `audio_module.py` | 262 | `SlotInduction` and `AudioModule`, the acoustic counterpart of `lang_module.py`. The text model gets its decomposition from an LLM parser; here K=6 learned slots compete for frames through Sinkhorn transport, with a predicted per-slot mass budget, a Gaussian time prior and the saliency. Two learned queries then **elect** a referent vector and a relation vector over the slots. |
| `audio_match_module.py` | 273 | `AudioMatchModule`, the fusion head. Keeps the old `match_module.py` shape (feature projection, spatial graph, copy-paste augmentation, a Conv1d scoring stack giving `cluster_ref`) but replaces its four parallel text cross-attention streams with the SPOT fusion. Contains `SpatialGraphNet` (dense masked kNN graph, k=5, no pyg) and `copy_paste_proposals`. Score = MLP([fused; graph]) + w·log(referent row of the transport plan). |
| `fusion/sinkhorn.py` | 127 | `sinkhorn_log`, `masked_log_softmax`, `plan_to_attention`, `plan_entropy`. A log-domain Sinkhorn used twice: frames→slots and slots→proposals. Adds the column constraint that plain softmax attention lacks. |
| `fusion/geometry.py` | 124 | `RelationSlotKernel` (RSK). A hypernetwork turns the relation vector into per-head direction, band centre (≤4 m), width (≥0.1 m), sharpness and blend. It biases proposal self-attention instead of the fixed "−distance" prior, and starts at that prior. |
| `fusion/spot_fusion.py` | 312 | `SPOTLayer` and `SPOTFusion` (D=3 layers: proposals read slots, slots read proposals, transport routing, gated read-out, geometric self-attention, plus a residual pass over raw frames). Also `ConcatFusion` and `XAttnFusion` baselines and `build_fusion`. |
| `refnet_audio.py` | 203 | `AudioRefNet` (detector + audio branch + fusion, replaces `RefNet`), `CachedAudioRefNet` (audio branch + fusion only, reads cached proposals) and `build_audio_model`. |
| `model_stucture.py` (root) | 1 759 | One-file listing of the model for reading. |

### Library (`lib/`)
| File | Role |
|---|---|
| `audio_config.py` (565) | One switchboard (`AUDIO`): encoder, layer, `MAX_AUDIO_LEN`, voices, slot/Sinkhorn/RSK settings, loss weights, the arm switches, `VGA_*` environment overrides, helpers for cache paths, utterance keys and pilot subsets. |
| `audio_base_dataset.py` (470) | `ScannetReferenceDataset`, the point-cloud dataset the audio one extends. |
| `audio_dataset.py` (247) | `AudioScannetReferenceDataset`: subclasses the text dataset so point-cloud handling stays one copy; adds `AudioFeatureStore` (reads the HDF5 audio cache), SpecAugment and `keep_cached` (drops annotations with no audio). |
| `scene_cache.py` (347) | On-disk format and reader for frozen detector output: `save_scene_cache`, `load_scene_cache`, `CachedSceneDataset`, `meta.json` with a compatibility check, so a run cannot read a cache built with different settings. |
| `audio_loss_helper.py` (185) | Total loss `α·L_Det + β·L_VG + γ·L_AudioCls` plus the SPOT terms: plan-entropy, slot orthogonality, role separation, and an InfoNCE referent-to-target term. No text teacher or distillation. |
| `audio_solver.py` (80) | `AudioSolver`, a subclass of `Solver`. Only the loss and the logged metrics differ. |

### Scripts (`scripts/`)
| File | Role |
|---|---|
| `audio/build_audioset.py` (417) | Text → speech. Pluggable backend (Piper, Coqui, espeak, gTTS); one 16 kHz mono 16-bit WAV per annotation, deterministic voice per key, last voices held out for validation only, resumable. |
| `audio/cache_audio_features.py` (418) | Speech → features. Frozen encoder layer, 50→25 Hz pooling, plus six prosody channels computed from the waveform (log-energy, its change, log-F0, voicing, spectral flux, boundary strength). One HDF5 per split. |
| `audio/build_scene_cache.py` (647) | Runs the detector once per scene and stores proposals (256 boxes, 288-d `detr_features`). |
| `audio/selftest.py` (360) | Shape and gradient check on synthetic tensors for every arm. |
| `audio/smoketest.py` (262) | End-to-end test on a tiny real corpus. |
| `audio_train.py` (356), `audio_eval.py` (434) | Training and evaluation entry points; evaluation writes `scores.p`/`predictions.p` in the text pipeline's format so the existing analysis code reads them. |

### Experiments (`experiments/audio/`)
| File | Role |
|---|---|
| `fusion_arms.py` (467) | One config table of 15 arms (4 progressive: `concat`, `xattn`, `spot-k2`, `spot`; leave-one-out: `spot-softmax`, `-unidir`, `-fixedroles`, `-norsk`, `-nograph`, `-nogate`, `-noframes`, `-noprosody`, `-nocopypaste`, `-notimeprior`, `-nospecaug`); subcommands `list`, `run`, `run-all`, `eval`, `table`. |
| `sweeps.py` (416) | 14 one-setting-at-a-time sweeps (seeds, slots, depth, graph_k, sinkhorn, ot_weight, encoder, encoder_layer, prosody_channels, frames, audio_cls, …). |
| `interventions.py` (220) | Evaluation-only checks on a trained run: controls (silence, mismatch, reverse, first/second half) and test-time knockouts. |
| `analysis.py` (560) | CPU-only tables and diagnostics: results table, slot roles, OT sharpness, gain attribution, error taxonomy, significance. |
| `robustness.py` (356) | Unseen speakers, noise at several SNRs, ASR cascade. |
| `measurements.py` (263) | Parameters and latency. |

### Notebooks, paper, docs
- `run_audio_data.ipynb` (TTS + encoding in one), `run_audio_train_eval.ipynb`, `run_audio_analysis.ipynb`.
- `paper/`: `manuscript.md` (720 lines), build chain (`build_docx.py`, `latex2omml.py`, `ooxml_check.py`, `build_pdf.py`, `trim_png.py`, `fill_numbers.py`), five figures, DOCX and PDF.
- `docs/`: `3DVGA.md`, `3DVGA-experiments.md`, `3DVGA-diagrams.md`, tutorials, `figures/3dvga-figures.html`; `SPOT_HANDOFF.md`, `SPOT_PROMPT.md`.

### Edits to existing text-pipeline files (the only ones)
- `lib/loss_helper.py` — `device='cuda'` replaced by the label tensor's device (two places), so CPU runs work.
- `models/backbone_module.py` — removed two identical `use_multiview` branches (the audio parser has no such flag and would raise); `AUDIO.POINTNET_IMPL` accepted next to `ABLATION.POINTNET_IMPL`.
- `models/proposal_module.py` — same `POINTNET_IMPL` fallback.
- `README.md` rewritten; `HANDOFF.md`/`PROMPT.md` moved to `docs/survey-review/`; `req/requirements.txt` +5 lines.

---

## 3. `08623b6` — voices, separate steps, archive
66 files, +745 / −437.

- **Archive.** The whole text pipeline moved with `git mv`, content unchanged: `Archive/experiments/` (ablation, analysis, measurements), `Archive/lib/` (dataset, enet, projection), `Archive/models/` (lang_module, match_module, refnet), `Archive/scripts/` (ScanRefer train/eval, predict, visualize, multiview), the old notebooks, `Z ETC Z/`, `PROJECT_REPORT.md`; plus a new `Archive/README.md`.
- **Voices.** `AUDIO.NUM_VOICES` 8 → 6 (three female, three male, US and British; the last two held out for validation). Piper models are downloaded to `data_audio/voices/` and loaded by `.onnx` path; a GPU probe falls back to CPU automatically; `--shard I/N` runs several CPU instances; `--num_scenes` limits the corpus; the manifest is resumable (+76 lines in the TTS script).
- **Three separate steps.** `scripts/audio/build_audioset.py` → `scripts/audio_tts/`; `scripts/audio/cache_audio_features.py` → `scripts/audio_encode/`; `run_audio_data.ipynb` split into `run_audio_tts.ipynb` (505 new lines) and `run_audio_encode.ipynb`.
- **Encoder.** wav2vec 2.0 large became the default, with per-encoder default layers (`ENCODER_DEFAULT_LAYER`, `encoder_layer()`), HDF5 written to `.tmp` and renamed, per-clip retry on CUDA out-of-memory.
- **Follow-ups.** `lib/audio_config.py` (+25), `lib/audio_dataset.py`, `experiments/audio/{fusion_arms,sweeps,robustness}.py`, `models/{backbone,proposal}_module.py`, `scripts/audio_train.py`, `scripts/audio/smoketest.py`, `README.md` (281 lines cut to the audio pipeline only), paper and docs updated to the new paths.

---

## 4. `6752de5` — wav2vec 2.0 base, 256 frames, alignment
141 files, +53 / −47. Most of the file count is the folder `3DVG-AUDIO -- FRESH` renamed to `3DVG -- AUDIO -- FRESH` with no content change; the real edits are about 20 files.

- **Encoder back to base.** `AUDIO.ENCODER = "wav2vec2"`, `ENCODER_DIM = 768`, layer 8. `wav2vec2-large` remains selectable (about 4× the GPU memory to cache).
- **Frame cap 128 → 256** (5.12 s → 10.24 s). Synthesised ScanRefer descriptions average about 4.5 s with a 95th percentile near 9 s; 128 frames cropped over a quarter of them.
- **Paper, handoff, prompt and docs** edited so they state the same values as the code: `SPOT_HANDOFF.md` (T=256, six voices, 15 arms, 14 sweeps, controls and knockouts, file list), `SPOT_PROMPT.md`, `paper/manuscript.md`, `docs/3DVGA-experiments.md`, `docs/method-tutorial.md`.
- Config, sweeps, notebooks and smoke test updated for the new defaults (`lib/audio_config.py`, `experiments/audio/{fusion_arms,sweeps}.py`, `scripts/audio_train.py`, `scripts/audio/smoketest.py`, the four notebooks).

---

## 5. How the code moved from LM3DVG to 3D-VGA

| Aspect | LM3DVG (text) | 3D-VGA / SPOT (audio) |
|---|---|---|
| Query | description text, parsed by an LLM into target / adjectives / neighbours | speech synthesised from the same description; no text, no ASR in the model |
| Query encoder | GloVe + GRU (`lang_module.py`) | frozen wav2vec 2.0 + six prosody channels (`audio_encoder.py`) |
| Decomposition | external parser, offline | six learned slots by Sinkhorn transport; roles elected, not indexed (`audio_module.py`) |
| Fusion | `match_module.py` (attention + graph) | SPOT layer with transport routing and relation kernel (`models/fusion/`) |
| Network class | `RefNet` | `AudioRefNet`, and `CachedAudioRefNet` for training on cached inputs |
| Detector | runs every step | frozen; proposals read from `cached_scenes/` |
| Query input | text read from JSON each step | features read from `data_audio/feat/<encoder>/<split>.h5` |
| Trained parameters | detector + language + match | acoustic branch + fusion only (about 2.1 M) |
| Entry points | `scripts/ScanRefer_{train,eval}.py` | `scripts/audio_{train,eval}.py` |
| Experiments | `experiments/ablation/train_arms.py`, `parsing*.py`, `analysis.py` | `experiments/audio/{fusion_arms,sweeps,interventions,analysis,robustness}.py` |
| Notebooks | `run_analysis.ipynb` etc. | `run_audio_{tts,encode,train_eval,analysis}.ipynb` |

Kept unchanged: the PointNet++/VoteNet/DETR modules, `lib/ap_helper.py`, `lib/eval_helper.py`,
`utils/`, `pointnet/`, the layout of the runner scripts and notebooks (config cell, GPU probe,
registry, preflight, Drive mirror). Moved to `Archive/`: everything text-specific — language and
match modules, parser ablations, text analyses, multiview projection, old notebooks.

---

## 6. Working tree after this file
Committed together with it:
- the snapshot folders `3DVG -- AUDIO -- FRESH/` and `3DVG -- AUDIO -- OLD/` are deleted;
- `SPOT_HANDOFF.md` and `SPOT_PROMPT.md` are deleted;
- `Archive/` was renamed to lowercase `archive/` and is now listed in `.gitignore`, so its 43 tracked
  files leave the index (they stay in the history above, e.g. `git show 08623b6:Archive/...`);
- `archive/from_old_snapshots/` keeps what was worth saving from the snapshots (a June 2025
  text-model run, `attention_modules.py`, `args.ipynb`, three comment-fix files);
- `HOWTO.md` and `GIT.md` are added and the archive paragraph in `README.md` is edited.
