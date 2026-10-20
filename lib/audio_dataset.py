"""
3D-VGA dataset: ScanRefer scenes + the frozen-encoder audio cache.

``AudioScannetReferenceDataset`` subclasses the text dataset rather than replacing it. The
point-cloud half of __getitem__ -- subsampling, augmentation, vote labels, box labels,
unique/multiple lookup -- is identical for 3DVG and 3DVG-Audio, and forking it would mean
two copies of the trickiest code in the repo drifting apart. So the parent builds the
sample and this class adds the acoustic keys:

    audio_feat_list   (L, T, C)   frozen encoder frames, zero-padded to T = MAX_AUDIO_LEN
    audio_len_list    (L,)        real frame count per annotation, before padding
    prosody_list      (L, T, P)   per-frame prosodic descriptors
    audio_cat_list    (L,)        ScanNet class of the referred object -> L_AudioCls target

The parent still populates the text keys (GloVe embeddings, the LLM parse), because this
class inherits its __getitem__ rather than forking it. **Nothing in the audio branch reads
them.** They are dead weight in the batch, kept only so the point-cloud half stays one
shared implementation; models/refnet_audio.py never constructs a language module, and
lib/audio_loss_helper.py has no term that touches them. The model is audio-only at
training time as well as inference.

Padding is not the paper's fixed num_frame: real lengths travel with the batch and become
an attention mask, so a 2 s clip is not diluted by 3 s of zeros.
"""

import os

import numpy as np

from lib.scene_cache import CachedSceneDataset
from lib.audio_config import AUDIO, feature_cache, utterance_key


class AudioFeatureStore:
    """Read-only view over one split's HDF5 cache.

    Opened lazily and per process: h5py handles do not survive a fork, and this dataset is
    read by DataLoader workers. Same reason -- and same shape of fix -- as the multiview
    handle in lib/dataset.py.
    """

    def __init__(self, path):
        self.path = path
        self._handle = None
        self._pid = None
        self._index = None
        self.feat_dim = AUDIO.ENCODER_DIM
        self.prosody_dim = AUDIO.PROSODY_DIM
        self._read_meta()

    def _read_meta(self):
        import h5py
        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                f"audio cache {self.path} not found.\n"
                f"  1. python scripts/audio_tts/build_audioset.py --splits train val\n"
                f"  2. python scripts/audio_encode/cache_audio_features.py --splits train val")
        with h5py.File(self.path, "r") as handle:
            self.feat_dim = int(handle.attrs["feat_dim"])
            self.prosody_dim = int(handle.attrs["prosody_dim"])
            self.encoder = str(handle.attrs["encoder"])
            self.fps = float(handle.attrs["fps"])
            self.max_len = int(handle.attrs["max_audio_len"])
            keys = handle["keys"][:]
            offsets = handle["offsets"][:]
        self._index = {
            key.decode("utf-8"): (int(offsets[i]), int(offsets[i + 1]))
            for i, key in enumerate(keys)
        }
        if self.max_len != AUDIO.MAX_AUDIO_LEN:
            # Silently reading a cache built with a different cap would change the
            # sequence length mid-experiment, so say so and follow the cache.
            print(f"[audio] cache {os.path.basename(self.path)} was built with "
                  f"MAX_AUDIO_LEN={self.max_len}; using that instead of "
                  f"{AUDIO.MAX_AUDIO_LEN}.")
            AUDIO.MAX_AUDIO_LEN = self.max_len

    def _file(self):
        import h5py
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self._handle = h5py.File(self.path, "r", libver="latest", swmr=False)
            self._pid = pid
        return self._handle

    def __contains__(self, key):
        return key in self._index

    def get(self, key):
        """(T, C) float32 frames and (T, P) float32 prosody. One silent frame if missing."""
        span = self._index.get(key)
        if span is None:
            return (np.zeros((1, self.feat_dim), dtype=np.float32),
                    np.zeros((1, self.prosody_dim), dtype=np.float32))
        start, end = span
        handle = self._file()
        frames = np.asarray(handle["features"][start:end], dtype=np.float32)
        prosody = np.asarray(handle["prosody"][start:end], dtype=np.float32)
        return frames, prosody


def keep_cached(annotations, split):
    """Drop the annotations whose audio is not in the feature cache for ``split``.

    For a pilot run on a corpus that is only partly synthesised: without it the missing
    utterances are heard as silence and counted in the score.
    """
    import h5py
    path = feature_cache(split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"audio cache {path} not found; build it with "
                                f"scripts/audio_encode/cache_audio_features.py")
    with h5py.File(path, "r") as handle:
        cached = {key.decode("utf-8") for key in handle["keys"][:]}
    kept = [d for d in annotations
            if utterance_key(d["scene_id"], str(d["object_id"]), str(d["ann_id"])) in cached]
    print(f"[audio] --only_cached_audio: {len(kept)} of {len(annotations)} {split} "
          f"annotation(s) have cached audio")
    return kept


def spec_augment(frames, rng):
    """SpecAugment on frozen encoder frames: mask spans of time and channels in place.

    Applied to encoder output, not to the spectrogram, because the encoder is frozen and
    cached -- so this is the only place augmentation can still reach the acoustic stream.
    It is the audio counterpart of the word-dropout that lib/lang_module.py already does
    to the text branch, and it matters more here: TTS audio has none of the speaker,
    channel or disfluency variation of real speech, so without it the acoustic branch
    memorises the synthesiser.
    """
    frames = frames.copy()
    n_frames, n_channels = frames.shape

    for _ in range(AUDIO.SPEC_TIME_MASKS):
        width = rng.integers(0, min(AUDIO.SPEC_TIME_WIDTH, max(1, n_frames // 4)) + 1)
        if width <= 0:
            continue
        start = rng.integers(0, max(1, n_frames - width))
        frames[start:start + width] = 0.0

    for _ in range(AUDIO.SPEC_CHANNEL_MASKS):
        width = rng.integers(0, AUDIO.SPEC_CHANNEL_WIDTH + 1)
        if width <= 0:
            continue
        start = rng.integers(0, max(1, n_channels - width))
        frames[:, start:start + width] = 0.0

    return frames


class AudioScannetReferenceDataset(CachedSceneDataset):
    """ScannetReferenceDataset + the acoustic modality.

    Inherits from CachedSceneDataset rather than ScannetReferenceDataset so the audio
    branch keeps the frozen-detector protocol and the lazy language loading that the
    ablation work added; with every flag off, that class is the plain dataset.
    """

    def __init__(self, *args, audio_split=None, **kwargs):
        super().__init__(*args, **kwargs)
        split = audio_split or self.split
        self.audio_store = AudioFeatureStore(feature_cache(split))
        self.audio_dim = self.audio_store.feat_dim
        self.prosody_dim = self.audio_store.prosody_dim
        self.max_audio_len = AUDIO.MAX_AUDIO_LEN
        self._augment_audio = (self.split == "train") and AUDIO.SPEC_AUGMENT

        # A missing utterance is read as one silent frame, which trains without complaint
        # and scores near chance -- so count them here, where it is still cheap to stop.
        missing = sum(utterance_key(d["scene_id"], str(d["object_id"]), str(d["ann_id"]))
                      not in self.audio_store for d in self.scanrefer)
        if missing:
            print(f"[audio] WARNING: {missing} of {len(self.scanrefer)} {self.split} "
                  f"annotation(s) are not in {os.path.basename(self.audio_store.path)} and "
                  f"will be heard as silence. Build the cache for the scenes you train on.")

    def _audio_for(self, scene_id, object_id, ann_id, rng):
        """Padded frames, padded prosody and the true length for one annotation."""
        frames, prosody = self.audio_store.get(
            utterance_key(scene_id, str(object_id), str(ann_id)))

        if self._augment_audio:
            frames = spec_augment(frames, rng)

        length = min(frames.shape[0], self.max_audio_len)
        padded = np.zeros((self.max_audio_len, self.audio_dim), dtype=np.float32)
        padded_pros = np.zeros((self.max_audio_len, self.prosody_dim), dtype=np.float32)
        padded[:length] = frames[:length]
        padded_pros[:length] = prosody[:length]
        return padded, padded_pros, max(1, length)

    def __getitem__(self, idx):
        data_dict = super().__getitem__(idx)

        # The parent already resolved which annotations this item covers; re-read them
        # from the same source so the two halves cannot fall out of step.
        entries = self.scanrefer_new[idx]
        scene_id = entries[0]["scene_id"]
        lang_num = len(entries)

        # Seeded per item, so augmentation is reproducible under a fixed seed even with
        # several loader workers -- np.random in a worker is not.
        rng = np.random.default_rng(abs(hash((scene_id, idx))) % (2 ** 32))

        feats = np.zeros((self.lang_num_max, self.max_audio_len, self.audio_dim),
                         dtype=np.float32)
        pros = np.zeros((self.lang_num_max, self.max_audio_len, self.prosody_dim),
                        dtype=np.float32)
        lens = np.ones((self.lang_num_max,), dtype=np.int64)
        cats = np.zeros((self.lang_num_max,), dtype=np.int64)

        for i in range(self.lang_num_max):
            # Past lang_num the parent repeats the last annotation's labels; mirror that
            # here so padded slots hold valid (if duplicated) audio rather than silence,
            # which would otherwise dominate the AudioCls loss.
            entry = entries[i] if i < lang_num else entries[lang_num - 1]
            frames, prosody, length = self._audio_for(
                entry["scene_id"], entry["object_id"], entry["ann_id"], rng)
            feats[i] = frames
            pros[i] = prosody
            lens[i] = length

            name = " ".join(entry["object_name"].split("_"))
            cats[i] = self.raw2label.get(name, 17)

        data_dict["audio_feat_list"] = feats
        data_dict["prosody_list"] = pros
        data_dict["audio_len_list"] = lens
        data_dict["audio_cat_list"] = cats
        return data_dict


def build_audio_dataset(**kwargs):
    """Build the audio dataset from parsed CLI args."""

    kwargs.setdefault("use_cache", bool(AUDIO.USE_CACHED_SCENES))
    kwargs.setdefault("cached_scenes_root", AUDIO.CACHED_SCENES_ROOT)
    kwargs.setdefault("deterministic", bool(AUDIO.DETERMINISTIC_SUBSAMPLE))
    kwargs.setdefault("subsample_salt", AUDIO.SUBSAMPLE_SALT)

    if AUDIO.USE_CACHED_SCENES and kwargs.get("augment", False):
        kwargs["augment"] = False
        print("\n[3D-VGA] use_cached_scenes=True -> forcing augment=False "
              "(the scene cache is augmentation-free).\n")

    return AudioScannetReferenceDataset(**kwargs)
