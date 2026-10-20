"""
Build the frozen-detector scene cache.

    python scripts/audio/build_scene_cache.py build \\
        --use_pretrained <run under outputs/> --splits train val --use_color --use_normal

Runs the detection branch once per scene and writes its output to disk. Every annotation of
that scene then reads the same tensors, so an ablation sweep compares fusion arms against
one fixed proposal set instead of re-fitting a detector per arm. See lib/scene_cache.py for
the format and the dataset that reads it, and docs/3DVGA-experiments.md for why freezing
the detector is the precondition for every number in the protocol.

The model built here is AudioRefNet with ``no_reference=True``, which returns straight
after the detection branch -- the acoustic branch is never constructed, so no audio cache
is needed to build a scene cache. ``build`` chains into the validator when it finishes;
--no_validate skips that.
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from data.scannet.model_util_scannet import ScannetDatasetConfig  # noqa: E402
from lib.config import CONF  # noqa: E402
from lib.scene_cache import (  # noqa: E402
    SCENE_CACHE_FP16_KEYS, SCENE_CACHE_KEYS_OPTIONAL, SCENE_CACHE_KEYS_REQUIRED,
    CachedSceneDataset, cache_dir_for, cache_path_for, load_scene_cache, read_meta,
    save_scene_cache, write_meta)
from models.refnet_audio import AudioRefNet  # noqa: E402

DC = ScannetDatasetConfig()


def _load_scanrefer(split):
    path = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{split}.json")
    with open(path) as f:
        return json.load(f)


def _one_chunk_per_scene(scanrefer, num_scenes=-1):
    """Build a scanrefer_new whose every entry is a distinct scene.

    ScannetReferenceDataset indexes chunks of up to lang_num_max descriptions that all
    share a scene, and loads the point cloud by that scene alone. For caching we want
    each scene visited exactly once, so we hand it one single-description chunk per
    scene: the language fields are irrelevant here, only the point cloud is.
    """
    seen, chunks = set(), []
    for data in scanrefer:
        scene_id = data["scene_id"]
        if scene_id in seen:
            continue
        seen.add(scene_id)
        chunks.append([data])
        if num_scenes != -1 and len(chunks) >= num_scenes:
            break
    return chunks


def build_dataset(args, split, todo_scene_ids=None):
    scanrefer = _load_scanrefer(split)
    chunks = _one_chunk_per_scene(scanrefer, args.num_scenes)

    if todo_scene_ids is not None:
        # Resume: only build the dataset over scenes that are not cached yet, so a
        # restart does not re-run the detector on work already on disk.
        todo = set(todo_scene_ids)
        chunks = [c for c in chunks if c[0]["scene_id"] in todo]

    # Keep only the one description per scene `chunks` references. Not an optimisation:
    # _tranform_des costs ~22 GB over the train split, and the detection branch never
    # touches those tensors.
    scanrefer = [chunk[0] for chunk in chunks]
    scene_list = sorted({c[0]["scene_id"] for c in chunks})

    dataset = CachedSceneDataset(
        args=args,
        scanrefer=scanrefer,
        scanrefer_new=chunks,
        scanrefer_all_scene=scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        lang_num_max=1,
        augment=False,               # the cache is augmentation-free by construction
        shuffle=False,
        use_cache=False,             # this script *creates* the cache, it does not read it
        deterministic=True,          # ...and the subsample must be reproducible
        subsample_salt=args.subsample_salt,
        lazy_scene_data=True,
        lazy_maxsize=args.lazy_maxsize,
    )
    return dataset, chunks


def _pending_scenes(args, split):
    """(all_scene_ids, not_yet_cached) for this split, honouring --num_scenes."""
    scanrefer = _load_scanrefer(split)
    all_ids = [c[0]["scene_id"] for c in _one_chunk_per_scene(scanrefer, args.num_scenes)]
    if not args.resume:
        return all_ids, list(all_ids)
    todo = [s for s in all_ids
            if not os.path.isfile(cache_path_for(args.output, split, s))]
    return all_ids, todo


def _fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def build_model(args, device):
    input_channels = (
        int(args.use_normal) * 3
        + int(args.use_color) * 3
        + int(not args.no_height)
    )
    model = AudioRefNet(
        args=args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        # Returns right after the detection branch, which is all the cache stores --
        # and with no_reference=True the acoustic branch is never built, so this needs
        # neither an audio cache nor a GPU-resident encoder.
        no_reference=True,
        dataset_config=DC,
    )

    folder = args.use_pretrained or args.use_checkpoint
    if not folder:
        raise ValueError(
            "Refusing to cache features from randomly initialised weights. "
            "Pass --use_pretrained <folder> or --use_checkpoint <folder>."
        )
    ckpt_path = os.path.join(CONF.PATH.OUTPUT, folder, "model_criteria_25.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"loading detector weights from {ckpt_path}")
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # comp_weight() from ScanRefer_train.py, but reporting what was skipped: a silent
    # shape mismatch here would fill the cache with untrained-network output.
    state = model.state_dict()
    loaded, skipped = [], []
    for key, value in weights.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(state)

    # Keyed on the model's own keys, so it asserts every detection weight this model has
    # was loaded; the skipped entries are the checkpoint's lang.*/match.*, as expected.
    vision_prefixes = ("backbone_net.", "vgen.", "proposal.")
    n_vision = sum(1 for k in state if k.startswith(vision_prefixes))
    missing_vision = [k for k in state if k.startswith(vision_prefixes) and k not in loaded]
    print(f"loaded {len(loaded)} tensors ({n_vision} detection-branch), "
          f"skipped {len(skipped)} unused (lang/match) from the checkpoint")
    if missing_vision:
        raise RuntimeError(
            "Detection-branch weights failed to load (missing or shape mismatch):\n  "
            + "\n  ".join(missing_vision[:20])
            + f"\n  ... ({len(missing_vision)} total)"
            + "\nCaching would capture untrained features. Aborting."
        )

    return model.to(device).eval(), ckpt_path


def dry_run(args, keys):
    """Verify everything the real run needs, without a single forward pass.

    Checks, in the order they would otherwise fail:
      1. the checkpoint exists and every detection-branch weight loads with matching shape
      2. the parse cache and scene .npy files the dataloader will ask for are present
      3. one sample can actually be built end to end by the dataset
      4. how many scenes will be written, and roughly how much disk that needs

    Cheap enough to run on a CPU-only machine in well under a minute: it constructs the
    model on CPU (no CUDA op is touched) and builds exactly one dataset sample per split.
    """
    print("=" * 78)
    print("DRY RUN -- no forward pass, no cache written")
    print("=" * 78)

    problems = []

    print("\n[1/4] model construction and checkpoint")
    try:
        model, ckpt_path = build_model(args, torch.device("cpu"))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      OK  AudioRefNet built on CPU ({n_params/1e6:.1f}M params)")
        print(f"      OK  every backbone_net/vgen/proposal weight loaded from")
        print(f"          {ckpt_path}")
        del model
    except Exception as exc:
        problems.append(f"model/checkpoint: {exc}")
        print(f"      FAIL  {exc}")

    total_scenes = 0
    per_split_scenes = {}
    for split in args.splits:
        print(f"\n[2/4] dataset for split '{split}'")
        try:
            dataset, chunks = build_dataset(args, split)
            n = len(chunks)
            per_split_scenes[split] = n
            total_scenes += n
            print(f"      OK  {n} unique scenes, parse cache and glove loaded")
        except Exception as exc:
            problems.append(f"dataset[{split}]: {exc}")
            print(f"      FAIL  {exc}")
            continue

        print(f"[3/4] building one sample from '{split}' (exercises the .npy files)")
        try:
            t0 = time.time()
            sample = dataset[0]
            dt = time.time() - t0
            pc = sample["point_clouds"]
            print(f"      OK  scene {chunks[0][0]['scene_id']} in {dt:.2f}s, "
                  f"point_clouds {tuple(pc.shape)} {pc.dtype}")
        except Exception as exc:
            problems.append(f"sample[{split}]: {exc}")
            print(f"      FAIL  {exc}")

    print("\n[4/4] output estimate")
    # 18 required tensors at num_proposals=256, two of them fp16; ~250 KB/scene measured
    # against the tensor shapes in lib/scene_cache.py SCENE_CACHE_KEYS_REQUIRED.
    per_scene_kb = 250 if not args.include_optional else 950
    est_mb = total_scenes * per_scene_kb / 1024
    for split, n in per_split_scenes.items():
        print(f"      {split:5s}  {n:4d} scenes -> {n * per_scene_kb / 1024:.0f} MB")
    print(f"      total  {total_scenes:4d} scenes -> {est_mb:.0f} MB "
          f"in {args.output}")
    print(f"      caching {len(keys)} tensors per scene")

    print("\n" + "=" * 78)
    if problems:
        print(f"DRY RUN FAILED -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("=" * 78)
        return 1
    print("DRY RUN PASSED -- re-run without --dry_run on a GPU machine to build the cache.")
    print("=" * 78)
    return 0


@torch.no_grad()
def cache_split(args, model, split, keys, device):
    out_dir = cache_dir_for(args.output, split)
    os.makedirs(out_dir, exist_ok=True)

    all_ids, todo = _pending_scenes(args, split)
    already = len(all_ids) - len(todo)
    if already:
        print(f"[{split}] resume: {already}/{len(all_ids)} scenes already cached, "
              f"{len(todo)} to go")
    if not todo:
        print(f"[{split}] nothing to do; all {len(all_ids)} scenes are cached")
        size_mb = sum(os.path.getsize(os.path.join(out_dir, f))
                      for f in os.listdir(out_dir) if f.endswith(".p")) / (1024 ** 2)
        return {"num_scenes": 0, "num_cached_total": len(all_ids),
                "elapsed_sec": 0.0, "size_mb": round(size_mb, 1),
                "tensor_shapes": {}, "skipped_existing": already}

    dataset, chunks = build_dataset(args, split, todo_scene_ids=todo)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    scene_ids = [c[0]["scene_id"] for c in chunks]
    written, cursor = 0, 0
    shapes_seen = {}

    t0 = time.time()
    # unit="scene" with total=len(todo) so the bar and ETA count scenes, not batches --
    # with batch_size=1 on CPU a batch-based bar would be misleadingly granular.
    bar = tqdm(total=len(todo), desc=f"caching {split}", unit="scene",
               dynamic_ncols=True, smoothing=0.05)
    for data_dict in loader:
        for key in data_dict:
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].to(device)

        data_dict = model(data_dict)

        batch = data_dict["center"].shape[0]
        for i in range(batch):
            scene_id = scene_ids[cursor + i]
            tensors = {}
            for key in keys:
                value = data_dict.get(key)
                if value is None or not torch.is_tensor(value):
                    continue
                tensors[key] = value[i]
                shapes_seen.setdefault(key, list(value[i].shape))
            save_scene_cache(cache_path_for(args.output, split, scene_id), tensors)
            written += 1
        cursor += batch

        # Release the whole graph-free forward output before the next scene. On CPU with
        # batch_size=1 this is what keeps RSS flat instead of creeping over four hours.
        del data_dict, tensors
        if args.gc_every and written % args.gc_every == 0:
            gc.collect()

        rate = written / max(time.time() - t0, 1e-9)
        bar.set_postfix_str(f"{rate * 60:.1f} scene/min, "
                            f"eta {_fmt_hms((len(todo) - written) / max(rate, 1e-9))}")
        bar.update(batch)
    bar.close()

    elapsed = time.time() - t0
    missing = [k for k in SCENE_CACHE_KEYS_REQUIRED if k not in shapes_seen]
    if missing:
        raise RuntimeError(
            "The detection branch did not produce these required keys: "
            + ", ".join(missing)
            + "\nThe cache would be incomplete; check models/proposal_module.py."
        )

    size_mb = sum(os.path.getsize(os.path.join(out_dir, f))
                  for f in os.listdir(out_dir) if f.endswith(".p")) / (1024 ** 2)
    print(
        f"[{split}] {written} scenes in {_fmt_hms(elapsed)} "
        f"({elapsed / max(written, 1):.2f}s/scene), {size_mb:.1f} MB on disk"
    )
    return {
        "num_scenes": written,
        "num_cached_total": already + written,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_scene": round(elapsed / max(written, 1), 3),
        "size_mb": round(size_mb, 1),
        "tensor_shapes": shapes_seen,
        "skipped_existing": already,
    }


def _add_build_args(parser):
    parser.add_argument("--output", type=str,
                        default="cached_scenes")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Default 8 on GPU, 1 on CPU (lowest possible memory).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Default 4 on GPU, 0 on CPU (a worker copies the dataset).")
    parser.add_argument("--num_threads", type=int, default=None,
                        help="torch CPU threads. Default: leave one core free so the "
                             "machine stays responsive during a long run.")
    parser.add_argument("--lazy_maxsize", type=int, default=None,
                        help="How many scene meshes to keep resident. Default 4 on GPU, "
                             "1 on CPU.")
    parser.add_argument("--gc_every", type=int, default=32,
                        help="Force a gc.collect() every N scenes (0 disables).")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Re-cache scenes even if their .p file already exists. By "
                             "default existing scenes are skipped, so an interrupted run "
                             "can simply be restarted.")
    parser.add_argument("--include_optional", action="store_true",
                        help="Also cache large unused intermediates (seed/vote/fp2 features).")
    parser.add_argument("--no_validate", dest="validate", action="store_false",
                        help="Skip the automatic cache validation at the end.")
    parser.add_argument("--validate_samples", type=int, default=32,
                        help="Scenes per split to re-run the detector on and compare "
                             "against the cache.")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Max relative error allowed on an fp32 cached tensor. Two "
                             "forward passes of the same convolution are not bit-identical "
                             "under cuDNN autotuning, so this cannot be zero.")
    parser.add_argument("--fp16_tolerance", type=float, default=2e-3,
                        help="Same, for the keys stored as float16 (SCENE_CACHE_FP16_KEYS).")
    parser.add_argument("--subsample_salt", type=str, default="v1")
    parser.add_argument("--dry_run", action="store_true",
                        help="Check checkpoint, data and one sample per split, then report "
                             "the scene count and estimated cache size. No forward pass, "
                             "nothing written. Safe and fast on a CPU-only machine.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU. NOTE: only useful with --dry_run -- a real caching "
                             "run needs CUDA, because models/proposal_module.py and "
                             "lib/loss_helper.py call .cuda() directly.")

    # model / data configuration -- must match the training run that will read the cache
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--num_proposals", type=int, default=256)
    parser.add_argument("--num_scenes", type=int, default=-1,
                        help="Cache only the first N scenes of each split (smoke tests).")
    parser.add_argument("--detector", type=str, default="VN", choices=["VN", "GF"])
    parser.add_argument("--use_pretrained", type=str, default="")
    parser.add_argument("--use_checkpoint", type=str, default="")
    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_bidir", action="store_true")
    parser.add_argument("--no_lang_cls", action="store_true")
    # Accepted for signature compatibility with the training script, but ignored: this
    # script always builds RefNet with no_reference=True (detection branch only).
    parser.add_argument("--no_reference", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--lang_input", type=str, default="glove+parse")
    parser.add_argument("--GF_path", type=str, default=None)
    return parser



def validate(args, model, device, keys, model_is_built=True):
    """Re-run the detector on sampled scenes and compare against what is on disk.

    The older validator compared end-to-end grounding accuracy against cached-path
    accuracy. That is an indirect test: it runs the whole fusion head, needs a trained
    grounding checkpoint, and a small accuracy difference is hard to attribute. This
    compares the cached tensors element-wise against a fresh forward pass, which is the
    property the cache actually has to have, and it localises a failure to the key and the
    scene that broke.

    Exact equality is not the bar. cuDNN picks algorithms per call, so two forward passes
    of the same convolution on the same device can differ in the last bits, and half the
    cached keys are stored as fp16. The tolerance is therefore per-key and relative.
    """
    meta = read_meta(args.output)
    print(f"cache generated {meta.get('generated_at')} from {meta.get('checkpoint')}")

    rng = np.random.RandomState(args.seed)
    failures, checked = [], 0

    for split in args.splits:
        all_ids, _ = _pending_scenes(args, split)
        if not all_ids:
            continue
        n = min(args.validate_samples, len(all_ids))
        sample = [all_ids[i] for i in rng.choice(len(all_ids), size=n, replace=False)]
        print(f"\n[{split}] re-running the detector on {n} of {len(all_ids)} scenes")

        dataset, chunks = build_dataset(args, split, todo_scene_ids=sample)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
        scene_ids = [c[0]["scene_id"] for c in chunks]

        for i, data_dict in enumerate(tqdm(loader, desc=f"validating {split}",
                                           unit="scene", dynamic_ncols=True)):
            scene_id = scene_ids[i]
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].to(device)
            with torch.no_grad():
                fresh = model(data_dict)
            cached = load_scene_cache(cache_path_for(args.output, split, scene_id),
                                      keys=set(keys))
            checked += 1
            for key in keys:
                if key not in cached or key not in fresh:
                    continue
                a = cached[key].float()
                b = fresh[key][0].detach().cpu().float()
                if a.shape != b.shape:
                    failures.append((split, scene_id, key,
                                     f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
                    continue
                # fp16 keys carry ~3 decimal digits, so they get a looser bound
                tol = args.fp16_tolerance if key in SCENE_CACHE_FP16_KEYS else args.tolerance
                scale = max(float(b.abs().max()), 1e-6)
                err = float((a - b).abs().max()) / scale
                if err > tol:
                    failures.append((split, scene_id, key,
                                     f"max relative error {err:.2e} > {tol:.0e}"))
            del data_dict, fresh, cached

    print("\n" + "=" * 70)
    print(f"  scenes checked        {checked}")
    print(f"  keys per scene        {len(keys)}")
    print(f"  tolerance             {args.tolerance:.0e} (fp32) / "
          f"{args.fp16_tolerance:.0e} (fp16 keys)")
    print(f"  mismatches            {len(failures)}")
    print("=" * 70)

    report = {
        "cached_scenes_root": args.output,
        "cache_generated_at": meta.get("generated_at"),
        "cache_checkpoint": meta.get("checkpoint"),
        "scenes_checked": checked,
        "splits": list(args.splits),
        "tolerance": args.tolerance,
        "fp16_tolerance": args.fp16_tolerance,
        "num_mismatches": len(failures),
        "mismatches": [{"split": s, "scene": sc, "key": k, "detail": d}
                       for s, sc, k, d in failures[:200]],
        "passed": not failures,
    }
    out_path = os.path.join(args.output, "validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

    if failures:
        print("\nFAIL -- the cache does not reproduce the detector. First 10:")
        for split, scene_id, key, detail in failures[:10]:
            print(f"  {split}/{scene_id}  {key}: {detail}")
        print("\nDo not run ablations against this cache until the cause is found.")
        return 1

    print("\nPASS -- every sampled scene reproduces the detector within tolerance.")
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python scripts/audio/build_scene_cache.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    build = sub.add_parser("build", help="build the scene cache",
                           description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_build_args(build)

    args = parser.parse_args(argv)

    # ScannetReferenceDataset reads args.detection to decide whether to build the parsed
    # language tensors. Caching never touches the language branch, so skip that work.
    args.detection = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    keys = list(SCENE_CACHE_KEYS_REQUIRED)
    if args.include_optional:
        keys += SCENE_CACHE_KEYS_OPTIONAL

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    on_cpu = device.type == "cpu"

    # Defaults chosen per device: on CPU, minimise memory and keep the machine usable.
    if args.batch_size is None:
        args.batch_size = 1 if on_cpu else 8
    if args.num_workers is None:
        # A DataLoader worker forks the whole dataset (glove dict included), so on CPU
        # in-process loading uses far less RAM than any worker would.
        args.num_workers = 0 if on_cpu else 4
    if args.lazy_maxsize is None:
        args.lazy_maxsize = 1 if on_cpu else 4

    if on_cpu:
        if args.num_threads is None:
            args.num_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(args.num_threads)
        print(
            f"\nCPU MODE\n"
            f"  batch_size   {args.batch_size}   (lowest memory)\n"
            f"  num_workers  {args.num_workers}   (in-process; a worker would copy the dataset)\n"
            f"  threads      {args.num_threads} of {os.cpu_count()}   (one core left free)\n"
            f"  resume       {'on' if args.resume else 'OFF'}   "
            f"(already-cached scenes are {'skipped' if args.resume else 're-run'})\n"
            f"  validation   skipped on CPU -- it needs ~37 more .cuda() call sites in\n"
            f"               lib/loss_helper.py and lib/eval_helper.py. Run\n"
            f"               build_scene_cache.py validate on a GPU before\n"
            f"               these numbers for the paper.\n"
            f"  This is expected to take hours. It is safe to Ctrl-C and restart:\n"
            f"  each scene is written atomically and completed scenes are skipped.\n"
        )
        # The end-to-end validation leg cannot run here; do not pretend otherwise.
        args.validate = False
    elif args.num_threads:
        torch.set_num_threads(args.num_threads)

    if args.dry_run:
        return dry_run(args, keys)

    model, ckpt_path = build_model(args, device)

    t0 = time.time()
    per_split = {}
    incomplete = []
    for split in args.splits:
        per_split[split] = cache_split(args, model, split, keys, device)
        # meta.json is the contract the training path checks, so it must describe the
        # cache as it is on disk now -- not just what this invocation happened to write.
        expected = len(_pending_scenes(args, split)[0])
        have = per_split[split].get("num_cached_total", per_split[split]["num_scenes"])
        per_split[split]["expected_scenes"] = expected
        per_split[split]["complete"] = (have >= expected)
        if have < expected:
            incomplete.append(f"{split}: {have}/{expected}")
    total = time.time() - t0

    write_meta(
        args.output,
        checkpoint=ckpt_path,
        detector=args.detector,
        num_points=args.num_points,
        num_proposals=args.num_proposals,
        subsample_salt=args.subsample_salt,
        augmentation="none (deterministic per-scene subsample, no flip/rot/scale/translate)",
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_height=(not args.no_height),
        device=device.type,
        batch_size=args.batch_size,
        cached_keys=keys,
        splits=per_split,
        complete=(not incomplete),
        total_elapsed_sec=round(total, 2),
    )
    print(f"\nwrote {os.path.join(args.output, 'meta.json')}")
    print(f"total cache time: {_fmt_hms(total)}")

    if incomplete:
        print("\nWARNING: cache is INCOMPLETE -- " + ", ".join(incomplete))
        print("Re-run the same command to continue; finished scenes are skipped.")
        return 2

    if args.validate:
        print("\n" + "=" * 78)
        print("running cache validation -- the cache is not usable until this passes")
        print("=" * 78)
        return validate(args, model, device, keys)

    return 0


if __name__ == "__main__":
    sys.exit(main())
