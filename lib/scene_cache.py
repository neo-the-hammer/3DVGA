"""
The frozen-detector scene cache: its on-disk format, and the dataset that reads it.

The detection branch never sees the referring expression -- spoken or written -- so its
output is a pure function of the scene. Computing it once per scene and reusing it for
every annotation of that scene is what makes an ablation sweep affordable: the detector
contributes more run-to-run variance than any fusion change does, and a sweep that refits
it per arm buries the effect being measured under detector noise.

Build a cache with:

    python scripts/audio/build_scene_cache.py build \\
        --use_pretrained <run under outputs/> --splits train val --use_color --use_normal

This module is the read side; that script is the write side.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.config import CONF  # noqa: E402
from lib.audio_base_dataset import SCANNET_V2_TSV, ScannetReferenceDataset  # noqa: E402


SCENE_CACHE_KEYS_REQUIRED = [
    "detr_features",                  # (K, 288)      -> MatchModule.features_concat
    "center",                         # (K, 3)
    "objectness_scores",              # (K, 2)
    "sem_cls_scores",                 # (K, num_class)
    "heading_scores",                 # (K, NH)
    "heading_residuals",              # (K, NH)
    "heading_residuals_normalized",   # (K, NH)
    "size_scores",                    # (K, NS)
    "size_residuals",                 # (K, NS, 3)
    "size_residuals_normalized",      # (K, NS, 3)
    "pred_obbs",                      # (K, 7)
    "pred_bboxes",                    # (K, 8, 3)
    "aggregated_vote_xyz",            # (K, 3)        -> compute_objectness_loss
    "aggregated_vote_features",       # (K, 128)
    "aggregated_vote_inds",           # (K,)
    "seed_xyz",                       # (num_seed, 3) -> compute_vote_loss
    "seed_inds",                      # (num_seed,)   -> compute_vote_loss
    "vote_xyz",                       # (num_seed*vote_factor, 3)
]

#: Nothing downstream reads these, and they roughly quadruple the cache size. Opt-in via
#: --include_optional, for debugging the cache itself.
SCENE_CACHE_KEYS_OPTIONAL = [
    "seed_features",
    "vote_features",
    "fp2_xyz",
    "fp2_features",
    "fp2_inds",
]

#: fp16 halves the cache size and these are activations, where the round-trip error sits
#: below the model's noise floor. Geometry and logits stay fp32 -- they feed IoU
#: thresholding and argmax, where the equivalence check needs exactness.
SCENE_CACHE_FP16_KEYS = {
    "detr_features",
    "aggregated_vote_features",
    "seed_features",
    "vote_features",
    "fp2_features",
}

CACHE_FORMAT_VERSION = 1



def scene_subsample_seed(scene_id, salt="v1"):
    """Stable 32-bit seed from the scene id. md5, not hash(), which is salted per process."""
    digest = hashlib.md5(f"{salt}:{scene_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)



def cache_dir_for(root, split):
    return os.path.join(root, split)


def cache_path_for(root, split, scene_id):
    return os.path.join(cache_dir_for(root, split), f"{scene_id}.p")


def meta_path_for(root):
    return os.path.join(root, "meta.json")



def save_scene_cache(path, tensors):
    """Persist one scene's detector output.

    ``tensors`` maps cache key -> torch tensor with the batch dimension already stripped
    (i.e. shape (K, ...), not (1, K, ...)).
    """
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for key, value in tensors.items():
        value = value.detach().cpu()
        if key in SCENE_CACHE_FP16_KEYS and value.dtype == torch.float32:
            value = value.half()
        payload[key] = value
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_scene_cache(path, keys=None, dtype=None):
    """Load one scene's cached tensors, restoring fp16 entries to ``dtype``."""
    import torch

    if dtype is None:
        dtype = torch.float32
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key, value in payload.items():
        if keys is not None and key not in keys:
            continue
        if value.dtype == torch.float16:
            value = value.to(dtype)
        out[key] = value
    return out



def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ).decode().strip()
    except Exception:
        return "unknown"


def write_meta(root, **fields):
    meta = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    meta.update(fields)
    os.makedirs(root, exist_ok=True)
    with open(meta_path_for(root), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def read_meta(root):
    path = meta_path_for(root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No meta.json at {path}. Generate the cache with\n"
            f"    python scripts/audio/build_scene_cache.py build --use_pretrained <folder> --splits train val\n"
            f"before enabling --use_cached_scenes."
        )
    with open(path) as f:
        return json.load(f)


def assert_meta_compatible(root, num_points, subsample_salt, detector, num_proposals=None):
    """Fail loudly when the training config does not match how the cache was built."""
    meta = read_meta(root)
    problems = []
    if meta.get("num_points") != num_points:
        problems.append(f"num_points: cache={meta.get('num_points')} run={num_points}")
    if meta.get("subsample_salt") != subsample_salt:
        problems.append(f"subsample_salt: cache={meta.get('subsample_salt')} run={subsample_salt}")
    if meta.get("detector") != detector:
        problems.append(f"detector: cache={meta.get('detector')} run={detector}")
    if num_proposals is not None and meta.get("num_proposals") != num_proposals:
        problems.append(f"num_proposals: cache={meta.get('num_proposals')} run={num_proposals}")
    if problems:
        raise RuntimeError(
            "Cached scenes were generated with a different configuration:\n  "
            + "\n  ".join(problems)
            + f"\nRegenerate the cache at {root} or fix the run configuration."
        )
    return meta



class _LazySceneData(dict):
    """A scene's npy arrays, loaded on first access, last few kept resident.

    Presents the mapping the base __getitem__ indexes, so lib/dataset.py works verbatim.
    """

    def __init__(self, scene_list, maxsize=4):
        super().__init__()
        self.scene_list = list(scene_list)
        self.maxsize = max(1, int(maxsize))
        self._cache = OrderedDict()

    def _load(self, scene_id):
        base = os.path.join(CONF.PATH.SCANNET_DATA, scene_id)
        return {
            "mesh_vertices": np.load(base + "_aligned_vert.npy"),
            "instance_labels": np.load(base + "_ins_label.npy"),
            "semantic_labels": np.load(base + "_sem_label.npy"),
            "instance_bboxes": np.load(base + "_aligned_bbox.npy"),
        }

    def __getitem__(self, scene_id):
        entry = self._cache.get(scene_id)
        if entry is None:
            entry = self._load(scene_id)
            self._cache[scene_id] = entry
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(scene_id)
        # The base class normalises colours in place, so hand back a fresh copy or the
        # second visit to a scene sees them normalised twice.
        return dict(entry, mesh_vertices=entry["mesh_vertices"].copy())

    def __contains__(self, scene_id):
        return scene_id in self.scene_list

    def __len__(self):
        return len(self.scene_list)

    def keys(self):
        return list(self.scene_list)



class CachedSceneDataset(ScannetReferenceDataset):
    """ScannetReferenceDataset + deterministic subsampling + cached detector output.

    Subclasses rather than edits, so lib/dataset.py keeps its original content. The two
    behaviours switch independently: ``deterministic`` pins the 40k-point subsample per
    scene, ``use_cache`` splices in that scene's cached detector tensors.
    """

    def __init__(self, *args, use_cache=False, cached_scenes_root=None,
                 deterministic=False, subsample_salt="v1", lazy_scene_data=True,
                 lazy_maxsize=4, **kwargs):
        self.use_cache = use_cache
        self.cached_scenes_root = cached_scenes_root
        self.subsample_salt = subsample_salt
        # Reading the cache only makes sense against the un-augmented, fixed subsample it
        # was generated from.
        self.deterministic = deterministic or use_cache
        self.lazy_scene_data = lazy_scene_data
        self.lazy_maxsize = lazy_maxsize

        if self.use_cache:
            if not cached_scenes_root:
                raise ValueError("use_cache=True requires cached_scenes_root")
            if kwargs.get("augment", False):
                raise ValueError(
                    "Cached scenes are incompatible with augment=True: the cache is built "
                    "from un-augmented point clouds, so augmented GT boxes would not "
                    "correspond to the cached proposals. Train cached ablations with "
                    "augment=False."
                )

        super().__init__(*args, **kwargs)

        if self.use_cache:
            assert_meta_compatible(
                cached_scenes_root,
                num_points=self.num_points,
                subsample_salt=subsample_salt,
                detector=getattr(self.args, "detector", None),
            )

    def _load_data(self):
        """ScannetReferenceDataset._load_data with the eager scene preload made lazy.

        Every other step is reproduced verbatim; set lazy_scene_data=False to fall back.
        """
        if not self.lazy_scene_data:
            return super()._load_data()

        # Mirrors ScannetReferenceDataset._load_data, except that scene arrays are
        # materialised on demand instead of up front.
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        self.scene_data = _LazySceneData(self.scene_list, maxsize=self.lazy_maxsize)

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()



    def _scene_id_for(self, idx):
        return self.scanrefer_new[idx][0]["scene_id"]

    def __getitem__(self, idx):
        scene_id = self._scene_id_for(idx)

        if self.deterministic:
            # Pin the global RNG random_sampling draws from, then restore it so no fixed
            # stream leaks into anything else in this worker.
            state = np.random.get_state()
            np.random.seed(scene_subsample_seed(scene_id, self.subsample_salt))
            try:
                data_dict = super().__getitem__(idx)
            finally:
                np.random.set_state(state)
        else:
            data_dict = super().__getitem__(idx)

        if self.use_cache:
            data_dict.update(self._load_cached_scene(scene_id))

        return data_dict

    def _load_cached_scene(self, scene_id):
        path = cache_path_for(self.cached_scenes_root, self.split, scene_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No cached scene for {scene_id} (split={self.split}) at {path}.\n"
                f"Build it first:  python scripts/audio/build_scene_cache.py build "
                f"--splits {self.split} "
                f"--use_pretrained <folder>"
            )
        return load_scene_cache(path, keys=set(SCENE_CACHE_KEYS_REQUIRED))
