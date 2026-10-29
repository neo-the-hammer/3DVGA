"""
Switchboard for the 3D-VGA (3D Visual Grounding by Audio) branch.

The model is audio-only, end to end: no transcript, no ASR, no text supervision at any
point in training or inference. The text branch of this repository is a *baseline* to
compare against, never a teacher.

Every knob for the audio branch lives here, reads as an
attribute (``AUDIO.ENCODER``), and can be overridden from the CLI without editing a file.

    AUDIO.ENCODER = "wavlm"            # which frozen speech encoder built the cache
    AUDIO.FUSION = "spot"              # the proposed fusion; "xattn"/"concat" are baselines
    AUDIO.NUM_SLOTS = 8                # more latent acoustic roles

Paths follow the repo's layout convention (data/ for inputs, data_parsing/ for derived
caches), so the audio artefacts sit in their own tree and nothing existing moves:

    data_audio/
      wav/<scene_id>/<object_id>_<ann_id>.wav      # 3DVG-AudioSet-style, 16 kHz mono 16-bit
      feat/<encoder>/<split>.h5                    # frozen encoder frames + prosody
      manifest_<split>.json                        # synthesis log (voice, duration, seed)
"""

import os

from lib.config import CONF


class _Config(dict):
    """dict with attribute access, so flags read as AUDIO.ENCODER."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    __setattr__ = dict.__setitem__


AUDIO = _Config()

# ---------------------------------------------------------------- paths

#: VGA_AUDIO_ROOT relocates the whole tree (a pilot corpus, or a mounted drive) without
#: touching the default one. Read at import, because the paths below derive from it.
AUDIO.ROOT = os.environ.get("VGA_AUDIO_ROOT") or os.path.join(CONF.PATH.BASE, "data_audio")
AUDIO.WAV_DIR = os.path.join(AUDIO.ROOT, "wav")
#: VGA_AUDIO_FEAT_DIR moves the feature caches alone, for when the WAVs sit on a slow
#: mounted drive and the caches -- read at every training step -- should not.
AUDIO.FEAT_DIR = os.environ.get("VGA_AUDIO_FEAT_DIR") or os.path.join(AUDIO.ROOT, "feat")
AUDIO.MANIFEST = os.path.join(AUDIO.ROOT, "manifest_{split}.json")
# Voice models are shared by every corpus, so they do not follow VGA_AUDIO_ROOT.
AUDIO.VOICE_DIR = os.path.join(CONF.PATH.BASE, "data_audio", "voices")

# ---------------------------------------------------------------- synthesis (3DVG-AudioSet)

#: Matches the 3DVG-AudioSet WAV spec: mono, 16 kHz, 16-bit PCM.
AUDIO.SAMPLE_RATE = 16000
AUDIO.CHANNELS = 1
AUDIO.BIT_DEPTH = 16

#: Which TTS backend scripts/audio_tts/build_audioset.py drives. Every backend writes the
#: same WAV spec and the same file names, so the rest of the pipeline is backend-agnostic.
#:   "piper"    - fast local neural TTS, multi-voice (recommended, no GPU needed)
#:   "coqui"    - Coqui TTS (VITS); closest public stand-in for the paper's Match-TTS
#:   "espeak"   - formant synthesis; ugly but dependency-free, useful for a smoke test
#:   "kokoro"   - Kokoro-82M, natural voices, GPU-friendly (pip install kokoro; apt espeak-ng)
#:   "gtts"     - online fallback
AUDIO.TTS_BACKEND = "piper"

#: Multi-speaker synthesis. The paper uses a single voice, which lets a model overfit one
#: speaker's timbre; sampling a voice per annotation makes the acoustic branch learn
#: content rather than channel. Speakers are assigned deterministically from the
#: annotation key, so the split is reproducible and train/val voices can be disjoint.
AUDIO.MULTI_SPEAKER = True
AUDIO.NUM_VOICES = 6
AUDIO.HELD_OUT_VOICES = 2          # last N voices are val/test-only, for a speaker-OOD check

# ---------------------------------------------------------------- frozen speech encoder

#: "wav2vec2" reproduces the paper's feature extractor; "wavlm" and "hubert" are stronger
#: drop-ins with the same frame rate and dimensionality.
AUDIO.ENCODER = "wav2vec2"
AUDIO.ENCODER_CKPT = {
    "wav2vec2": "facebook/wav2vec2-base-960h",
    # 24 layers, 1024-d. An option for the encoder sweep: it needs ~4x the GPU memory
    # to cache and a third more disk.
    "wav2vec2-large": "facebook/wav2vec2-large-960h-lv60-self",
    "wavlm": "microsoft/wavlm-base-plus",
    "hubert": "facebook/hubert-base-ls960",
}
AUDIO.ENCODER_DIM = 768            # the real width is read from the feature cache
#: Hidden layer read from the encoder. -1 picks the encoder's entry below: two thirds of
#: the way up, because mid layers carry more phonetic/semantic content than the last.
AUDIO.ENCODER_LAYER = -1
ENCODER_DEFAULT_LAYER = {"wav2vec2-large": 16, "wav2vec2": 8, "wavlm": 8, "hubert": 8}

#: Frames are pooled by this stride when the cache is built: the encoders emit 50 fps, and
#: 25 fps loses nothing for this task while halving every downstream tensor. A 4.25 s clip
#: (the 3DVG-AudioSet mean) becomes ~106 frames.
AUDIO.FRAME_POOL = 2

#: Hard cap on cached frames per utterance, i.e. the analogue of CONF.TRAIN.MAX_DES_LEN.
#: 256 frames at 25 fps = 10.24 s. Synthesised ScanRefer descriptions average ~4.5 s with
#: a 95th percentile near 9 s; a 128-frame cap cropped over a quarter of them.
#: Unlike the paper's fixed num_frame padding, shorter clips are *masked*, not zero-padded
#: into the attention, so a short utterance is never diluted by padding.
AUDIO.MAX_AUDIO_LEN = 256

#: float16 on disk and in the batch tensor; the projection casts to float32. Halves both
#: cache size and host RAM, and the encoder features have nothing like 7 digits of signal.
AUDIO.CACHE_DTYPE = "float16"

#: Appended to the cache filename, so a corrupted or re-parameterised cache sits beside the
#: clean one instead of overwriting it. experiments/audio/robustness.py's noise sweep sets
#: it per SNR; empty means the ordinary cache.
AUDIO.CACHE_SUFFIX = ""

# ---------------------------------------------------------------- prosody stream

#: Per-frame prosodic descriptors computed alongside the encoder features:
#: log-energy, delta log-energy, log-F0, voicing probability, spectral flux, boundary
#: strength. These are what a text branch structurally cannot see.
AUDIO.USE_PROSODY = True
AUDIO.PROSODY_DIM = 6

#: Which prosody channels reach the FiLM generator. "all" is the proposed setting; the
#: groups exist so the ablation can say *which* prosodic cue carries the signal rather than
#: only that prosody as a whole helps. Channel order is fixed by
#: scripts/audio_encode/cache_audio_features.compute_prosody:
#:   0 log-energy   1 d log-energy   2 log-F0   3 voicing   4 spectral flux   5 boundary
AUDIO.PROSODY_CHANNELS = "all"
PROSODY_GROUPS = {
    "all":      (0, 1, 2, 3, 4, 5),
    "energy":   (0, 1),          # loudness and its slope -- lexical stress
    "pitch":    (2, 3),          # F0 and voicing -- phrase melody
    "spectral": (4, 5),          # flux and boundary strength -- segmentation cues
    "none":     (),
}

# ---------------------------------------------------------------- slot induction

#: Number of latent acoustic role slots. The text branch of this repo parses a sentence
#: into three fields (target / adjectives / neighbors); K slots play the same part with no
#: parser, no ASR pass and no text of any kind. K=6 leaves headroom beyond those roles.
AUDIO.NUM_SLOTS = 6

#: Roles are ELECTED, not indexed. Two learned queries attend over the slot bank and pool
#: it into a "referent" vector and a "relation" vector (models/audio_module.py).
#:
#: This used to be a fixed assignment -- slot 0 is the target, slot 3 is the relation --
#: which was only true because a distillation loss pinned it there from the text parse.
#: With the model audio-only, nothing anchors a numbered slot to a linguistic role, and
#: reading slots[:, 3] as "the relation" would have been an unenforced assumption sitting
#: underneath the geometric kernel. Election has to be learned anyway, so it is learned.
#: experiments/audio/analysis.py slot-roles reports which slot won each role.
AUDIO.NUM_ROLES = 2
AUDIO.ROLE_REFERENT = 0
AUDIO.ROLE_RELATION = 1

#: Sinkhorn settings for frame -> slot assignment. eps is the entropic temperature, and
#: what matters is its ratio to the cost range -- which is fixed, because the slot logits
#: are cosine similarities times a bounded learned scale. These values were chosen by
#: measuring marginal convergence at this problem's shapes (scripts/audio/selftest.py
#: re-checks it): eps 0.3 with 8 iterations holds the row-marginal error under 2%, where
#: eps 0.05 with 5 iterations leaves 30-50% and the "slot mass is a budget" reading of the
#: plan stops being true. Sharpen with SlotInduction's learned scale, not by lowering eps.
AUDIO.SLOT_SINKHORN_EPS = 0.3
AUDIO.SLOT_SINKHORN_ITERS = 8

#: Each slot carries a learnable Gaussian prior over normalised time. English referring
#: expressions are ordered ("the <attr> <target> <relation> <anchor>"), so initialising the
#: means evenly across the utterance gives the slots a head start; they are free to move.
AUDIO.SLOT_TIME_PRIOR = True

# ---------------------------------------------------------------- fusion

#: "spot"   - the proposed Slot-Proposal Optimal Transport fusion (docs/3DVGA.md)
#: "xattn"  - slot/frame cross-attention only, i.e. the attentive-fusion baseline
#: "concat" - AP-Refer's fusion: mean-pooled audio concatenated with proposals, masked, MLP
AUDIO.FUSION = "spot"
FUSION_VARIANTS = ("spot", "xattn", "concat")

# ----------------------------------------------------- SPOT component switches
#
# One switch per step of the operator, so an ablation can remove exactly one thing. Each
# defaults to the proposed setting; docs/3DVGA-experiments.md maps them to table rows.

#: False replaces Sinkhorn with a plain row-wise softmax over proposals, keeping the same
#: cost matrix, the same read-out and the same parameter count. This is THE ablation for
#: the central claim: it removes the competition between proposals and nothing else, so a
#: gap here is attributable to the transport constraint alone.
AUDIO.OT_ROUTING = True

#: False removes step (2), the slots-read-proposals direction. Slots then summarise the
#: utterance without ever seeing the scene.
AUDIO.BIDIRECTIONAL_SLOTS = True

#: False replaces the learned gate in the routed read-out with a plain residual add.
AUDIO.ROUTE_GATE = True

#: False removes the single residual attention pass over raw encoder frames that runs
#: after the stack, leaving the K slots as the only acoustic pathway.
AUDIO.FRAME_RESIDUAL = True

#: False replaces the learned role election with fixed slot indices (0 = referent,
#: 1 = relation). Those indices are arbitrary without a text teacher to pin them, which is
#: the whole argument for the election -- this arm measures what that argument is worth.
AUDIO.ROLE_ELECTION = True

#: Neighbours per node in SpatialGraphNet's dense masked kNN. 0 removes the graph.
AUDIO.GRAPH_K = 5
AUDIO.GRAPH_LAYERS = 6

AUDIO.HIDDEN_SIZE = 128
AUDIO.NUM_HEADS = 4
AUDIO.FUSION_DEPTH = 3

#: Sinkhorn settings for slot <-> proposal routing. This is the part that turns fusion
#: from "every proposal independently attends to audio" into an assignment with
#: competition between proposals. Same reasoning as the slot temperature above: at
#: 6 slots x 256 proposals, eps 0.3 with 8 iterations converges to ~1.6% marginal error,
#: eps 0.08 with 4 iterations to ~38%.
AUDIO.OT_EPS = 0.3
AUDIO.OT_ITERS = 8

#: Weight of the transport-plan grounding logit, mixed into the MLP score.
AUDIO.OT_LOGIT_WEIGHT = 0.5

#: Relation-Slot Geometric Kernel: the relation slot predicts the spatial attention
#: bias instead of the fixed -distance prior in models/match_module.py.
AUDIO.USE_RSK = True
AUDIO.RSK_MIN_SIGMA = 0.1         # metres; floors the distance band so gradients stay sane
AUDIO.RSK_MAX_MU = 4.0            # metres; caps the band centre to indoor-scene scale

# ---------------------------------------------------------------- losses

#: L_3DVG-Audio = alpha * L_Det + beta * L_VG + gamma * L_AudioCls  (paper Eq. 1).
#: alpha/beta follow this repo's existing get_loss weighting; gamma=1 is the paper's value.
AUDIO.LOSS_AUDIO_CLS = 1.0

#: Slot regularisers: entropy on the frame->slot plan (keeps slots from collapsing onto a
#: single loud frame), an orthogonality penalty between slot vectors, and a separation
#: penalty between the two role elections so the referent and the relation cannot resolve
#: to the same slot.
#:
#: These carry more weight than they would in a distilled model. They are, with the time
#: prior and the contrastive term below, the ONLY things shaping the decomposition -- the
#: model sees no text at any point, so nothing tells it what a "relation" is.
AUDIO.LOSS_SLOT_ENTROPY = 0.02
AUDIO.LOSS_SLOT_ORTHO = 0.01
AUDIO.LOSS_ROLE_SEPARATION = 0.05

#: In-batch InfoNCE between the elected referent vector and the ground-truth proposal
#: feature. This is the one piece of direct supervision the role election gets, and it
#: comes from GEOMETRY, not from text: the positive is the proposal the reference loss
#: already treats as correct.
AUDIO.LOSS_TARGET_CONTRASTIVE = 0.2
AUDIO.CONTRASTIVE_TEMPERATURE = 0.07

# ---------------------------------------------------------------- augmentation

#: SpecAugment-style masking applied to *cached encoder frames* (train split only).
AUDIO.SPEC_AUGMENT = True
AUDIO.SPEC_TIME_MASKS = 2
AUDIO.SPEC_TIME_WIDTH = 12
AUDIO.SPEC_CHANNEL_MASKS = 2
AUDIO.SPEC_CHANNEL_WIDTH = 48

#: Drop whole slots at random during training, so no single slot becomes load-bearing and
#: the model degrades gracefully when a word is mumbled or clipped.
AUDIO.SLOT_DROPOUT = 0.1

# ---------------------------------------------------------------- runtime

#: Hold the audio cache on disk and read slices per item instead of materialising every
#: annotation's frames in every worker.
AUDIO.LAZY_AUDIO_DATA = True

#: Hold scene arrays on disk and materialise them on demand. The eager path needs ~36 GB
#: for train+val and is SIGKILLed on smaller machines; the lazy path gives identical
#: tensors in ~1.8 GB. Not an ablation -- a host-RAM control.
AUDIO.LAZY_SCENE_DATA = True

# ------------------------------------------------- frozen detector (scene cache)
#
# Read the detection branch's output from disk instead of running it. Build the cache with
# scripts/audio/build_scene_cache.py. This is what makes an ablation sweep measure the
# fusion change rather than detector noise, so every arm in experiments/audio/ assumes it.

AUDIO.USE_CACHED_SCENES = False
AUDIO.CACHED_SCENES_ROOT = "cached_scenes"

#: Seed the 40k-point subsample from the scene id. Implied by USE_CACHED_SCENES; set it
#: alone for an end-to-end run that must reproduce the cache. Must match meta.json.
AUDIO.DETERMINISTIC_SUBSAMPLE = False
AUDIO.SUBSAMPLE_SALT = "v1"

#: True disables the proposal copy-paste augmentation in models/audio_match_module.py.
AUDIO.DISABLE_COPY_PASTE = False

#: Refuse a checkpoint that would leave part of the model randomly initialised. Eval loads
#: with strict=False, so a half-loaded fusion head is otherwise silent -- and a half-loaded
#: model reports a plausible number rather than failing.
AUDIO.STRICT_CHECKPOINT = True

#: None walks pointnet/__init__.py's three-tier chain; a tier name pins it and raises if it
#: cannot be imported. Read at model import time, so there is no CLI flag.
AUDIO.POINTNET_IMPL = None



#: AUDIO key -> argparse destination, for flags that carry a value. The three booleans
#: that default to True (USE_RSK, USE_PROSODY, SPEC_AUGMENT) are NOT here: their switches
#: are the negative form, and apply() handles them separately.
_CLI_FLAGS = {
    "ENCODER": "audio_encoder",
    "FUSION": "audio_fusion",
    "NUM_SLOTS": "num_slots",
    "MAX_AUDIO_LEN": "max_audio_len",
    "PROSODY_CHANNELS": "prosody_channels",
    "FUSION_DEPTH": "fusion_depth",
    "GRAPH_K": "graph_k",
    "TTS_BACKEND": "tts_backend",
    "OT_LOGIT_WEIGHT": "ot_logit_weight",
    "USE_CACHED_SCENES": "use_cached_scenes",
    "CACHED_SCENES_ROOT": "cached_scenes_root",
    "DETERMINISTIC_SUBSAMPLE": "deterministic_subsample",
    "SUBSAMPLE_SALT": "subsample_salt",
    "DISABLE_COPY_PASTE": "disable_copy_paste",
}

#: Scalar knobs a sweep may set as VGA_<NAME> without a CLI flag of its own.
#: experiments/audio/sweeps.py uses this for the grids it varies through the environment;
#: keeping it to an explicit list means a typo'd variable is an error, not a silent no-op.
_ENV_KEYS = (
    "OT_EPS", "OT_ITERS", "SLOT_SINKHORN_EPS", "SLOT_SINKHORN_ITERS",
    "LOSS_AUDIO_CLS", "LOSS_TARGET_CONTRASTIVE", "LOSS_SLOT_ENTROPY",
    "LOSS_SLOT_ORTHO", "LOSS_ROLE_SEPARATION",
    "SLOT_DROPOUT", "OT_LOGIT_WEIGHT", "NUM_SLOTS", "ENCODER_LAYER",
    "FRAME_POOL", "MAX_AUDIO_LEN", "NUM_VOICES", "HELD_OUT_VOICES",
    "CACHE_SUFFIX",
    "USE_CACHED_SCENES", "CACHED_SCENES_ROOT", "DETERMINISTIC_SUBSAMPLE",
    "SUBSAMPLE_SALT", "DISABLE_COPY_PASTE",
    "FUSION_DEPTH", "OT_ROUTING", "BIDIRECTIONAL_SLOTS", "ROUTE_GATE",
    "FRAME_RESIDUAL", "ROLE_ELECTION", "GRAPH_K", "GRAPH_LAYERS",
    "PROSODY_CHANNELS", "USE_RSK", "USE_PROSODY",
    "SLOT_TIME_PRIOR", "SPEC_AUGMENT", "ENCODER",
)


def apply_env():
    """Fold VGA_<NAME> overrides into AUDIO, typed from the existing default."""
    applied = []
    for key in _ENV_KEYS:
        raw = os.environ.get(f"VGA_{key}")
        if raw is None:
            continue
        current = AUDIO[key]
        try:
            if isinstance(current, bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
        except ValueError as exc:
            raise SystemExit(f"VGA_{key}={raw!r} is not a valid "
                             f"{type(current).__name__}") from exc
        AUDIO[key] = value
        applied.append(f"{key}={value!r}")
    if applied:
        print(f"[3D-VGA] environment overrides: {', '.join(applied)}")
    return applied


def encoder_layer():
    layer = int(AUDIO.ENCODER_LAYER)
    return layer if layer >= 0 else ENCODER_DEFAULT_LAYER[AUDIO.ENCODER]


def encoder_ckpt():
    return AUDIO.ENCODER_CKPT[AUDIO.ENCODER]


def feature_cache(split, suffix=None):
    """Path of the frozen-encoder cache for one split."""
    suffix = AUDIO.CACHE_SUFFIX if suffix is None else suffix
    return os.path.join(AUDIO.FEAT_DIR, AUDIO.ENCODER, f"{split}{suffix}.h5")


def wav_path(scene_id, object_id, ann_id):
    """3DVG-AudioSet's standardised naming scheme, one directory per scene."""
    return os.path.join(AUDIO.WAV_DIR, scene_id, f"{object_id}_{ann_id}.wav")


def utterance_key(scene_id, object_id, ann_id):
    """Cache key for one annotation. Kept flat so the h5 index is a plain string array."""
    return f"{scene_id}/{object_id}/{ann_id}"


def pilot_subset(annotations, num_scenes=-1, limit=None):
    """Subset for a pilot run: the first ``num_scenes`` scenes in sorted order -- the same
    pick scripts/audio_train.py makes for --num_scenes -- then the first ``limit``."""
    if num_scenes is not None and num_scenes > 0:
        scenes = set(sorted({a["scene_id"] for a in annotations})[:num_scenes])
        annotations = [a for a in annotations if a["scene_id"] in scenes]
    if limit:
        annotations = annotations[:limit]
    return annotations


def add_arguments(parser):
    """CLI overrides, so a sweep does not need to edit this file per run."""
    group = parser.add_argument_group("3D-VGA audio branch (lib/audio_config.py)")
    group.add_argument("--audio_encoder", type=str, default=None,
                       choices=list(AUDIO.ENCODER_CKPT),
                       help="Frozen speech encoder the feature cache was built with.")
    group.add_argument("--audio_fusion", type=str, default=None, choices=list(FUSION_VARIANTS),
                       help="Fusion module: 'spot' (proposed), 'xattn' or 'concat' (baselines).")
    group.add_argument("--num_slots", type=int, default=None,
                       help="Number of latent acoustic role slots.")
    group.add_argument("--max_audio_len", type=int, default=None,
                       help="Cap on cached frames per utterance (25 fps after pooling).")
    group.add_argument("--no_rsk", action="store_true", default=False,
                       help="Use the fixed -distance prior instead of the audio-conditioned "
                            "geometric kernel (ablation).")
    group.add_argument("--no_prosody", action="store_true", default=False,
                       help="Drop the prosody stream (ablation).")
    group.add_argument("--prosody_channels", type=str, default=None,
                       choices=list(PROSODY_GROUPS),
                       help="Which prosodic cue group reaches the FiLM generator.")
    group.add_argument("--fusion_depth", type=int, default=None,
                       help="Number of stacked SPOT layers.")
    group.add_argument("--graph_k", type=int, default=None,
                       help="Neighbours per node in SpatialGraphNet; 0 removes the graph.")
    group.add_argument("--no_ot", action="store_true", default=False,
                       help="Replace Sinkhorn with a row-wise softmax, keeping the cost "
                            "matrix and read-out. Isolates the transport constraint.")
    group.add_argument("--no_bidirectional", action="store_true", default=False,
                       help="Remove the slots-read-proposals direction (SPOT step 2).")
    group.add_argument("--no_route_gate", action="store_true", default=False,
                       help="Plain residual add instead of the learned read-out gate.")
    group.add_argument("--no_frame_residual", action="store_true", default=False,
                       help="Remove the residual attention pass over raw encoder frames.")
    group.add_argument("--no_role_election", action="store_true", default=False,
                       help="Use fixed slot indices for the two roles instead of electing "
                            "them.")
    group.add_argument("--no_spec_augment", action="store_true", default=False,
                       help="Disable SpecAugment on cached frames.")
    group.add_argument("--no_time_prior", action="store_true", default=False,
                       help="Remove the Gaussian time prior from slot induction (ablation).")
    group.add_argument("--tts_backend", type=str, default=None,
                       help="TTS backend for scripts/audio_tts/build_audioset.py.")
    group.add_argument("--ot_logit_weight", type=float, default=None,
                       help="Initial weight on the transport plan's grounding logit. The "
                            "parameter is learnable, so this sets where it starts.")
    group.add_argument("--audio_lang_num_max", type=int, default=8,
                       help="Annotations per scene per batch item. Kept low because the "
                            "audio frames are the bigger tensor.")

    group.add_argument("--use_cached_scenes", action="store_true", default=None,
                       help="Train/eval on cached detector output "
                            "(scripts/audio/build_scene_cache.py).")
    group.add_argument("--cached_scenes_root", type=str, default=None,
                       help="Directory holding the scene cache (train/, val/, meta.json).")
    group.add_argument("--deterministic_subsample", action="store_true", default=None,
                       help="Pin the 40k-point subsample per scene. Implied by "
                            "--use_cached_scenes.")
    group.add_argument("--subsample_salt", type=str, default=None,
                       help="Salt for the deterministic subsample seed; must match meta.json.")
    group.add_argument("--disable_copy_paste", action="store_true", default=None,
                       help="Turn off the proposal copy-paste augmentation (ablation).")
    group.add_argument("--keep_checkpoint", action="store_true", default=False,
                       help="Keep --use_checkpoint in cached mode (cleared by default so "
                            "ablation runs start from a fresh fusion head).")
    group.add_argument("--no_strict_checkpoint", action="store_true", default=False,
                       help="Allow a checkpoint to leave part of the model randomly "
                            "initialised. Off by default: half-loaded weights produce a "
                            "plausible-looking but meaningless accuracy.")
    return parser


def apply(args):
    """Fold environment and CLI overrides into AUDIO, and mirror them back onto ``args``.

    Environment first, so an explicit CLI flag still wins over a sweep's grid.
    """
    apply_env()

    for key, dest in _CLI_FLAGS.items():
        value = getattr(args, dest, None)
        if value is not None and value is not False:
            AUDIO[key] = value

    # These default to True, so their switches are the negative form.
    if getattr(args, "no_rsk", False):
        AUDIO.USE_RSK = False
    if getattr(args, "no_prosody", False):
        AUDIO.USE_PROSODY = False
    if getattr(args, "no_spec_augment", False):
        AUDIO.SPEC_AUGMENT = False
    for flag, key in (("no_ot", "OT_ROUTING"),
                      ("no_bidirectional", "BIDIRECTIONAL_SLOTS"),
                      ("no_route_gate", "ROUTE_GATE"),
                      ("no_frame_residual", "FRAME_RESIDUAL"),
                      ("no_role_election", "ROLE_ELECTION"),
                      ("no_time_prior", "SLOT_TIME_PRIOR")):
        if getattr(args, flag, False):
            AUDIO[key] = False
    if getattr(args, "no_strict_checkpoint", False):
        AUDIO.STRICT_CHECKPOINT = False

    if AUDIO.USE_CACHED_SCENES:
        # There is no detection branch left to train or supervise.
        args.no_detection = True
        args.detection = False

        # comp_weight() would copy the whole model here, so every seed would share one
        # initialisation. Warm-starting arms opt back in with --keep_checkpoint.
        if getattr(args, "use_checkpoint", "") and not getattr(args, "keep_checkpoint", False):
            print(f"\n[3D-VGA] ignoring --use_checkpoint {args.use_checkpoint!r}: in cached "
                  f"mode it would preload the fusion weights and every run would share one\n"
                  f"         initialisation. Pass --keep_checkpoint to override.\n")
            args.use_checkpoint = ""

    if hasattr(args, "detector") and args.detector is None:
        # The cache fixes the detector, so follow it instead of making every command
        # repeat it; without a cache the default is the same as the text branch's.
        args.detector = "GF"
        if AUDIO.USE_CACHED_SCENES:
            meta = os.path.join(AUDIO.CACHED_SCENES_ROOT, "meta.json")
            if os.path.isfile(meta):
                import json
                args.detector = json.load(open(meta)).get("detector") or "GF"
                print(f"[3D-VGA] --detector {args.detector} (from {meta})")

    if AUDIO.FUSION not in FUSION_VARIANTS:
        raise ValueError(f"FUSION must be one of {FUSION_VARIANTS}, got {AUDIO.FUSION!r}")
    if AUDIO.PROSODY_CHANNELS not in PROSODY_GROUPS:
        raise ValueError(f"PROSODY_CHANNELS must be one of {tuple(PROSODY_GROUPS)}, "
                         f"got {AUDIO.PROSODY_CHANNELS!r}")
    if AUDIO.GRAPH_K < 0:
        raise ValueError(f"GRAPH_K must be >= 0 (0 disables the graph), got {AUDIO.GRAPH_K}")
    if AUDIO.NUM_SLOTS < AUDIO.NUM_ROLES:
        raise ValueError(
            f"NUM_SLOTS={AUDIO.NUM_SLOTS} is below NUM_ROLES={AUDIO.NUM_ROLES}: the role "
            f"election cannot pool a referent and a relation out of fewer slots than roles.")

    for key, dest in _CLI_FLAGS.items():
        if hasattr(args, dest):
            setattr(args, dest, AUDIO[key])

    return args


def force_cpu():
    """Run without CUDA: make ``.cuda()`` a no-op for this process.

    lib/solver.py, lib/loss_helper.py and lib/eval_helper.py call ``.cuda()`` directly --
    inherited code, shared with the text branch, and not worth forking for a smoke test.
    Same shim as experiments/measurements.py uses for its CPU evaluation.
    """
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self          # noqa: E731
    torch.nn.Module.cuda = lambda self, *a, **k: self       # noqa: E731
    print("[3D-VGA] running on CPU: .cuda() is a no-op for this process")


def describe():
    keys = ("ENCODER", "FUSION", "NUM_SLOTS", "MAX_AUDIO_LEN", "USE_PROSODY", "USE_RSK",
            "SPEC_AUGMENT", "OT_EPS", "OT_ITERS")
    lines = ["3D-VGA audio config (lib/audio_config.py):"]
    lines += [f"  {key} = {AUDIO[key]!r}" for key in keys]
    return "\n".join(lines)
