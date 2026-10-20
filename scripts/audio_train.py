"""
Train the 3D-VGA model (3D visual grounding by audio).

    python scripts/audio_train.py --use_color --use_normal --batch_size 8 --epoch 50 \
        --audio_fusion spot --tag SPOT

Counterpart of scripts/ScanRefer_train.py. Same optimiser, same per-module learning rates,
same output layout (outputs/<timestamp>_<TAG>/), same checkpoint criteria -- so a 3DVG run
and a 3DVG-Audio run are directly comparable and the analysis suite reads both.

Prerequisites, in order:

    python scripts/audio_tts/build_audioset.py --splits train val
    python scripts/audio_encode/cache_audio_features.py --splits train val

Ablation arms (the fusion comparison in docs/3DVGA.md):

    --audio_fusion concat        AP-Refer's fusion
    --audio_fusion xattn         attentive fusion
    --audio_fusion spot          proposed                      (default)
    --no_rsk                    fixed -distance prior instead of the audio-conditioned one
    --no_prosody                 drop the prosody stream
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.audio_config import AUDIO, add_arguments as add_audio_arguments, apply as apply_audio
from lib.audio_config import force_cpu
from lib.audio_config import describe as describe_audio
from lib.audio_dataset import build_audio_dataset, keep_cached
from lib.audio_solver import AudioSolver
from lib.config import CONF
from models.refnet_audio import build_audio_model
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

DC = ScannetDatasetConfig()


def group_by_scene(annotations, lang_num_max):
    """Chunk a split into per-scene groups of at most ``lang_num_max`` annotations.

    Same grouping as ScanRefer_train.get_scanrefer, which is what makes the detector run
    once per scene instead of once per description. Rewritten here only to drop that
    function's reliance on a module-level ``args``.
    """
    groups, current, scene_id = [], [], None
    for data in annotations:
        if scene_id != data["scene_id"]:
            scene_id = data["scene_id"]
            if current:
                groups.append(current)
                current = []
        if len(current) >= lang_num_max:
            groups.append(current)
            current = []
        current.append(data)
    if current:
        groups.append(current)
    return groups


def get_scanrefer(num_scenes, lang_num_max, num_val_scenes=-1, only_cached=False):
    train_scenes = sorted({d["scene_id"] for d in SCANREFER_TRAIN})
    val_scenes = sorted({d["scene_id"] for d in SCANREFER_VAL})
    if num_scenes != -1:
        assert len(train_scenes) >= num_scenes
        train_scenes = train_scenes[:num_scenes]
    if num_val_scenes != -1:
        val_scenes = val_scenes[:num_val_scenes]

    train = [d for d in SCANREFER_TRAIN if d["scene_id"] in set(train_scenes)]
    val = [d for d in SCANREFER_VAL if d["scene_id"] in set(val_scenes)]
    if only_cached:
        train, val = keep_cached(train, "train"), keep_cached(val, "val")

    scanrefer = {"train": train, "val": val}
    scanrefer_new = {"train": group_by_scene(train, lang_num_max),
                     "val": group_by_scene(val, lang_num_max)}
    print(f"\ntrain on {len(train)} annotations / {len(scanrefer_new['train'])} groups, "
          f"val on {len(val)} annotations / {len(scanrefer_new['val'])} groups\n")
    return scanrefer, scanrefer_new, train_scenes + val_scenes


def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, augment,
                   shuffle=True):
    dataset = build_audio_dataset(
        args=args,
        scanrefer=scanrefer[split],
        scanrefer_new=scanrefer_new[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle,
    )

    workers = args.num_workers
    if workers is None:
        workers = min(4, max(0, (os.cpu_count() or 1) - 1))

    kwargs = {}
    if workers > 0:
        kwargs.update(num_workers=workers, prefetch_factor=args.prefetch_factor,
                      persistent_workers=True, pin_memory=torch.cuda.is_available())

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    print(f"[loader] split={split} batch={args.batch_size} workers={workers} "
          f"audio_dim={dataset.audio_dim} frames={dataset.max_audio_len}")
    return dataset, loader


def get_model(args, audio_dim):
    channels = (int(args.use_normal) * 3
                + int(args.use_color) * 3 + int(not args.no_height))
    model = build_audio_model(
        args=args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        dataset_config=DC,
        audio_dim=audio_dim,
    )

    if args.use_pretrained and AUDIO.USE_CACHED_SCENES:
        print(f"[3D-VGA] ignoring --use_pretrained {args.use_pretrained!r}: the detector's "
              f"output is read from the scene cache, so there is no detector to load")
    elif args.use_pretrained:
        # Warm-start the detector from a 3DVG run. The detection branch is identical
        # between the two tasks, so this is the cheapest way to make an audio run
        # comparable to a text one -- and with --no_detection it freezes, which is the
        # protocol every fusion ablation should use.
        path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_criteria_25.pth")
        print(f"\nloading pretrained detector from {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        own = model.state_dict()
        loaded = [k for k, v in state.items()
                  if k in own and own[k].shape == v.shape and not k.startswith(("audio.", "match."))]
        if not loaded:
            raise RuntimeError(
                f"{path} shares no detector tensor with this model. The audio branch only "
                f"wants the detector half of a 3DVG checkpoint; check --detector.")
        model.load_state_dict({k: state[k] for k in loaded}, strict=False)
        print(f"[3D-VGA] detector warm start: {len(loaded)} tensor(s) loaded")

        if args.no_detection:
            frozen = (["detector"] if args.detector == "GF"
                      else ["backbone_net", "vgen", "proposal"])
            for name in frozen:
                for param in getattr(model, name).parameters():
                    param.requires_grad = False
            print(f"[3D-VGA] detector frozen ({', '.join(frozen)})")

    return model.cuda()


def get_solver(args, dataloader, audio_dim):
    model = get_model(args, audio_dim)

    weight_dict = {
        "detr": {"lr": args.detr_lr},
        "detector": {"lr": args.detr_lr},
        "audio": {"lr": args.audio_lr},
        "match": {"lr": args.match_lr},
    }
    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd,
                                weight_dict=weight_dict)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    lr_scheduler = None
    if args.coslr:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    elif args.lr_decay_step:
        lr_scheduler = MultiStepLR(optimizer, args.lr_decay_step, 0.1)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.tag:
        stamp += "_" + args.tag.upper()
    root = os.path.join(CONF.PATH.OUTPUT, stamp)
    os.makedirs(root, exist_ok=True)

    solver = AudioSolver(
        args=args, model=model, config=DC, dataloader=dataloader, optimizer=optimizer,
        stamp=stamp, val_step=args.val_step,
        detection=not args.no_detection, reference=not args.no_reference,
        use_lang_classifier=False,
        lr_decay_step=args.lr_decay_step, lr_decay_rate=0.1,
        lr_scheduler=lr_scheduler,
    )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nnum_params (trainable): {total}")
    return solver, total, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {key: value for key, value in vars(args).items()}
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_params"] = num_params
    info["audio"] = {key: AUDIO[key] for key in sorted(AUDIO)
                     if isinstance(AUDIO[key], (int, float, str, bool, list))}
    with open(os.path.join(root, "info.json"), "w") as out:
        json.dump(info, out, indent=4)


def train(args):
    print(describe_audio())

    scanrefer, scanrefer_new, all_scene_list = get_scanrefer(
        args.num_scenes, args.lang_num_max, args.num_val_scenes, args.only_cached_audio)
    train_dataset, train_loader = get_dataloader(args, scanrefer, scanrefer_new,
                                                 all_scene_list, "train", augment=True)
    val_dataset, val_loader = get_dataloader(args, scanrefer, scanrefer_new,
                                             all_scene_list, "val", augment=False,
                                             shuffle=False)

    solver, num_params, root = get_solver(
        args, {"train": train_loader, "val": val_loader}, train_dataset.audio_dim)

    print("\nStart training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--val_step", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--num_proposals", type=int, default=256)
    parser.add_argument("--num_scenes", type=int, default=-1)
    parser.add_argument("--num_val_scenes", type=int, default=-1,
                        help="Validate on the first N val scenes only (pilot run).")
    parser.add_argument("--only_cached_audio", action="store_true",
                        help="Use only the annotations whose audio is in the feature cache "
                             "(pilot run on a partly synthesised corpus).")
    parser.add_argument("--cpu", action="store_true",
                        help="Run without CUDA (smoke test; implied when none is usable).")
    parser.add_argument("--coslr", action="store_true")
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--lr_decay_step", type=int, nargs="+", default=None)

    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_lang_cls", action="store_true", default=True)
    parser.add_argument("--no_detection", action="store_true")
    parser.add_argument("--no_reference", action="store_true")
    parser.add_argument("--detection", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_bidir", action="store_true")
    parser.add_argument("--use_pretrained", type=str, default="")
    parser.add_argument("--use_checkpoint", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=3)

    parser.add_argument("--lang_input", type=str, default="glove+parse")
    parser.add_argument("--detector", type=str, default=None, choices=["VN", "GF"],
                        help="Default: the scene cache's detector in cached mode, else GF.")
    parser.add_argument("--GF_path", type=str, default=None)
    parser.add_argument("--audio_lr", type=float, default=5e-4)
    parser.add_argument("--match_lr", type=float, default=5e-4)
    parser.add_argument("--detr_lr", type=float, default=1e-4)

    # GroupFree detector knobs, identical to ScanRefer_train.py so a detector checkpoint
    # built there loads here.
    parser.add_argument("--width", default=1, type=int)
    parser.add_argument("--num_target", type=int, default=256)
    parser.add_argument("--sampling", default="kps", type=str)
    parser.add_argument("--nhead", default=8, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_activation", default="relu", type=str)
    parser.add_argument("--self_position_embedding", default="loc_learned", type=str)
    parser.add_argument("--cross_position_embedding", default="xyz_learned", type=str)
    parser.add_argument("--query_points_generator_loss_coef", default=0.8, type=float)
    parser.add_argument("--obj_loss_coef", default=0.1, type=float)
    parser.add_argument("--box_loss_coef", default=1, type=float)
    parser.add_argument("--sem_cls_loss_coef", default=0.1, type=float)
    parser.add_argument("--center_loss_type", default="smoothl1", type=str)
    parser.add_argument("--center_delta", default=0.04, type=float)
    parser.add_argument("--size_loss_type", default="smoothl1", type=str)
    parser.add_argument("--size_delta", default=0.111111111111, type=float)
    parser.add_argument("--heading_loss_type", default="smoothl1", type=str)
    parser.add_argument("--heading_delta", default=1.0, type=float)
    parser.add_argument("--query_points_obj_topk", default=4, type=int)
    parser.add_argument("--size_cls_agnostic", action="store_true")
    parser.add_argument("--clip_norm", default=0.1, type=float)
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--syncbn", action="store_true")
    parser.add_argument("--ap_iou_thresholds", type=float, default=[0.25, 0.5], nargs="+")

    add_audio_arguments(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.detection = args.no_reference
    # Fewer annotations per scene than the text branch's 32. The input tensor is not the
    # reason -- 32 x 256 frames x 768 channels is only ~200 MB a batch -- it is that this
    # number multiplies the flattened annotation axis B*L that every fusion tensor is
    # shaped on, and those carry gradients: the RSK bias alone is (B*L, heads, M, M),
    # 67 MB at 8 and 268 MB at 32, before the attention activations over (B*L, M, H).
    args.lang_num_max = args.audio_lang_num_max

    apply_audio(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.cpu or not torch.cuda.is_available():
        force_cpu()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    train(args)
