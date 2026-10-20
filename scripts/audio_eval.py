"""
Evaluate a 3D-VGA run and report the 3DVG-Audio metric table.

    python scripts/audio_eval.py --folder <run> --use_color --use_normal --force
    python scripts/audio_eval.py --folder <run> ... --audio_control silence --out_subdir interventions/silence

Writes ``predictions.p`` and ``scores.p`` into the run directory in the same format
scripts/ScanRefer_eval.py uses, so the existing analysis suite reads an audio run without
changes, and prints the Unique / Multiple / Overall x Acc@0.25 / Acc@0.5 table that the
3DVG-Audio paper reports (their Table 2).

The model is audio-only by construction -- it has no text branch at all, at training time
or any other time -- so no number here can have been helped by a transcript.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.audio_config import AUDIO, PROSODY_GROUPS, apply as apply_audio, force_cpu
from lib.audio_dataset import build_audio_dataset, keep_cached
from lib.audio_loss_helper import compute_audio_classification_accuracy, get_audio_loss
from lib.config import CONF
from lib.eval_helper import get_eval
from models.refnet_audio import build_audio_model
from scripts.audio_train import build_parser as build_train_parser, group_by_scene

SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
DC = ScannetDatasetConfig()


def get_scanrefer(args):
    scanrefer = SCANREFER_VAL
    if args.num_scenes != -1:
        scenes = sorted({d["scene_id"] for d in scanrefer})[:args.num_scenes]
        scanrefer = [d for d in scanrefer if d["scene_id"] in scenes]
    if args.only_cached_audio:
        scanrefer = keep_cached(scanrefer, "val")
    scene_list = sorted({d["scene_id"] for d in scanrefer})
    return scanrefer, scene_list, group_by_scene(scanrefer, args.lang_num_max)


def get_dataloader(args, scanrefer, scanrefer_new, scene_list):
    dataset = build_audio_dataset(
        args=args, scanrefer=scanrefer, scanrefer_new=scanrefer_new,
        scanrefer_all_scene=scene_list, split="val",
        num_points=args.num_points, use_height=(not args.no_height),
        use_color=args.use_color, use_normal=args.use_normal,
        lang_num_max=args.lang_num_max,
        augment=False, shuffle=False,
    )
    return dataset, DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def get_model(args, audio_dim):
    channels = (int(args.use_normal) * 3
                + int(args.use_color) * 3 + int(not args.no_height))
    model = build_audio_model(
        args=args, num_class=DC.num_class, num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster, mean_size_arr=DC.mean_size_arr,
        input_feature_dim=channels, num_proposal=args.num_proposals,
        use_lang_classifier=False, use_bidir=args.use_bidir, no_reference=False,
        dataset_config=DC, audio_dim=audio_dim,
    ).cuda()

    path = os.path.join(CONF.PATH.OUTPUT, args.folder, args.checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    # A checkpoint that leaves the fusion head randomly initialised still produces a
    # plausible-looking number, so refuse it (AUDIO.STRICT_CHECKPOINT).
    critical = [k for k in missing if k.startswith(("audio.", "match."))]
    if critical and not args.no_strict_checkpoint:
        raise RuntimeError(
            f"{path} leaves {len(critical)} tensor(s) of the audio/fusion branch at random "
            f"initialisation, e.g. {critical[:3]}. The architecture does not match the "
            f"checkpoint -- check --audio_fusion / --num_slots / --no_rsk against the run's "
            f"info.json. Pass --no_strict_checkpoint to score it anyway (the result will "
            f"not mean anything).")
    print(f"[3D-VGA] loaded {args.folder}/{args.checkpoint}: "
          f"{len(missing)} missing, {len(unexpected)} unused")
    model.eval()
    return model


#: Paths belong to the machine evaluating, not to the one that trained -- a run trained
#: on Colab must still evaluate from this checkout.
_MACHINE_KEYS = ("ROOT", "WAV_DIR", "FEAT_DIR", "MANIFEST", "VOICE_DIR")

CONTROLS = ("none", "silence", "mismatch", "reverse", "first_half", "second_half")


def apply_audio_control(data, mode, generator):
    """Replace a batch's audio at test time. The model, the scene and the labels stay.

    Diagnostic controls, not a robustness test: they ask whether the score depends on
    what was said. ``silence`` and ``mismatch`` give the accuracy the scene alone
    reaches; ``reverse`` and the two halves ask whether temporal order and which part of
    the utterance matter.
    """
    if mode == "none":
        return data
    feats = data["audio_feat_list"].clone()                    # (B, L, T, C)
    pros = data["prosody_list"].clone()                        # (B, L, T, P)
    lens = data["audio_len_list"].clone()                      # (B, L)
    batch, lang = lens.shape

    if mode == "silence":
        feats.zero_()
        pros.zero_()
    elif mode == "mismatch":
        # Give every annotation the audio of one that names a different object class.
        cats = data["audio_cat_list"].reshape(-1).cpu()
        valid = (torch.arange(lang).unsqueeze(0)
                 < data["lang_num"].cpu().unsqueeze(1)).reshape(-1)
        pool = torch.where(valid)[0]
        flat_f = feats.reshape(batch * lang, *feats.shape[2:])
        flat_p = pros.reshape(batch * lang, *pros.shape[2:])
        flat_l = lens.reshape(-1)
        src_f, src_p, src_l = flat_f.clone(), flat_p.clone(), flat_l.clone()
        for index in pool.tolist():
            other = pool[cats[pool] != cats[index]]
            if other.numel() == 0:
                other = pool[pool != index]
            if other.numel() == 0:
                continue
            donor = int(other[torch.randint(other.numel(), (1,), generator=generator)])
            flat_f[index], flat_p[index], flat_l[index] = \
                src_f[donor], src_p[donor], src_l[donor]
    else:
        for i in range(batch):
            for j in range(lang):
                n = int(lens[i, j])
                if mode == "reverse":
                    feats[i, j, :n] = feats[i, j, :n].flip(0)
                    pros[i, j, :n] = pros[i, j, :n].flip(0)
                    continue
                half = max(1, n // 2)
                start = 0 if mode == "first_half" else n - half
                kept_f = feats[i, j, start:start + half].clone()
                kept_p = pros[i, j, start:start + half].clone()
                feats[i, j].zero_()
                pros[i, j].zero_()
                feats[i, j, :half] = kept_f
                pros[i, j, :half] = kept_p
                lens[i, j] = half

    data["audio_feat_list"], data["prosody_list"], data["audio_len_list"] = feats, pros, lens
    return data


def flatten_prosody(data, keep):
    """Remove the prosodic cues outside group ``keep`` at test time.

    Each removed channel is replaced by its mean over the utterance, which deletes what
    prosody carries -- the variation over time -- without feeding the model an input level
    it never saw. The trained weights are untouched, unlike --prosody_channels, which
    changes the FiLM generator's input width and so needs its own training run.
    """
    if keep == "all":
        return data
    pros = data["prosody_list"].clone()                        # (B, L, T, P)
    lens = data["audio_len_list"]
    drop = [c for c in range(pros.shape[-1]) if c not in PROSODY_GROUPS[keep]]
    for i in range(pros.shape[0]):
        for j in range(pros.shape[1]):
            n = max(1, int(lens[i, j]))
            pros[i, j, :n, drop] = pros[i, j, :n, drop].mean(dim=0, keepdim=True)
    data["prosody_list"] = pros
    return data


def restore_run_config(args):
    """Rebuild the audio config the run was trained with, from its info.json.

    Without this, evaluating a --no_rsk or --audio_fusion xattn run with default flags
    builds a different architecture, the checkpoint half-loads, and the reported accuracy
    is meaningless. Explicit flags on the command line still win.
    """
    info_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "info.json")
    if not os.path.isfile(info_path):
        print(f"[3D-VGA] no info.json in {args.folder}; using the current audio config")
        return
    info = json.load(open(info_path))
    stored = info.get("audio", {})
    explicit = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    restored = []
    for key, value in stored.items():
        if key in AUDIO and key not in _MACHINE_KEYS and AUDIO[key] != value:
            AUDIO[key] = value
            restored.append(f"{key}={value!r}")
    if args.detector is None and info.get("detector"):
        args.detector = info["detector"]
        restored.append(f"detector={args.detector!r}")
    if restored:
        print(f"[3D-VGA] restored from info.json: {', '.join(restored)}")
    if explicit & {"audio_fusion", "num_slots", "no_rsk", "no_prosody"}:
        print("[3D-VGA] note: explicit CLI flags were applied after the restore.")


def evaluate(args):
    scanrefer, scene_list, scanrefer_new = get_scanrefer(args)
    dataset, dataloader = get_dataloader(args, scanrefer, scanrefer_new, scene_list)
    model = get_model(args, dataset.audio_dim)

    post_dict = None if args.no_nms else {
        "remove_empty_box": True, "use_3d_nms": True, "nms_iou": 0.25,
        "use_old_type_nms": False, "cls_nms": True, "per_class_proposal": True,
        "conf_thresh": 0.05, "dataset_config": DC,
    }

    ref_acc, ious, masks, others, audio_acc = [], [], [], [], []
    predictions = {}
    diagnostics = []
    runtime = 0.0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    control_rng = torch.Generator().manual_seed(args.seed)

    for data in tqdm(dataloader):
        data = apply_audio_control(data, args.audio_control, control_rng)
        data = flatten_prosody(data, args.prosody_keep)
        for key in data:
            data[key] = data[key].cuda()

        with torch.no_grad():
            start = time.time()
            data = model(data)
            runtime += time.time() - start

            _, data = get_audio_loss(args=args, data_dict=data, config=DC,
                                     detection=True, reference=True, progress=1.0)
            data = get_eval(data_dict=data, config=DC, reference=True,
                            use_lang_classifier=False, post_processing=post_dict)

            ref_acc += data["ref_acc"]
            masks += data["ref_multiple_mask"]
            others += data["ref_others_mask"]
            audio_acc.append(compute_audio_classification_accuracy(data).item())

            # get_eval emits ref_iou / pred_bboxes in (i, j) order over j < lang_num[i],
            # skipping the padded slots. Walk that same order so each IoU lands on the
            # annotation it belongs to instead of on the group's first entry.
            batch_ious = data["ref_iou"]
            batch_boxes = data["pred_bboxes"]
            ids = data["scan_idx"].detach().cpu().numpy()
            lang_num = data["lang_num"].detach().cpu().numpy()
            cursor = 0
            for i in range(ids.shape[0]):
                group = scanrefer_new[int(ids[i])]
                for j in range(int(lang_num[i])):
                    entry = group[j]
                    slot = predictions.setdefault(entry["scene_id"], {}) \
                                      .setdefault(entry["object_id"], {})
                    slot[entry["ann_id"]] = {
                        "iou": float(batch_ious[cursor]),
                        "bbox": np.asarray(batch_boxes[cursor]).tolist(),
                    }
                    cursor += 1
            ious += batch_ious

            if args.dump_diagnostics:
                diagnostics.append(collect_diagnostics(data, batch_ious, lang_num))

    ious = np.array(ious)
    masks = np.array(masks)
    report = summarise(ious, masks, np.mean(audio_acc) if audio_acc else 0.0)

    root = os.path.join(CONF.PATH.OUTPUT, args.folder, args.out_subdir)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "predictions.p"), "wb") as out:
        pickle.dump(predictions, out)
    with open(os.path.join(root, "scores.p"), "wb") as out:
        pickle.dump({"ious": ious, "masks": masks, "ref_acc": np.array(ref_acc),
                     "others": np.array(others), "report": report,
                     "audio_control": args.audio_control,
                     "prosody_keep": args.prosody_keep,
                     "audio": {k: AUDIO[k] for k in sorted(AUDIO)
                               if isinstance(AUDIO[k], (int, float, str, bool))}}, out)

    if diagnostics:
        merged = {key: np.concatenate([d[key] for d in diagnostics], axis=0)
                  for key in diagnostics[0]}
        with open(os.path.join(root, "diagnostics.p"), "wb") as out:
            pickle.dump(merged, out)
        print(f"diag    -> {os.path.join(root, 'diagnostics.p')} "
              f"({merged['iou'].shape[0]} utterances)")

    print(f"\ninference: {runtime:.1f}s over {len(dataloader)} batches")
    print(f"scores  -> {os.path.join(root, 'scores.p')}")
    print(f"preds   -> {os.path.join(root, 'predictions.p')}")


def collect_diagnostics(data, batch_ious, lang_num):
    """Per-utterance slot and transport statistics, for experiments/audio/analysis.py.

    Summary statistics only, not the raw plans: the plans are (B', 6, 128) and
    (B', 6, 256) per batch, which would be gigabytes over the val split, while every
    question the analysis asks — did the slots separate, did the transport concentrate,
    which slot each role elected — is answerable from these reductions.
    """
    lang_num = np.asarray(lang_num)
    len_num_max = data["audio_feat_list"].shape[1]
    keep = np.concatenate([np.arange(len_num_max) < n for n in lang_num])

    plan = data["audio_slot_plan"]                                  # (B', K, T)
    mask = data["audio_frame_mask"]                                 # (B', T)
    rows = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    position = torch.arange(plan.shape[-1], device=plan.device).float()
    centroid = (rows * position).sum(dim=-1)                        # (B', K)
    lengths = mask.sum(dim=1).clamp_min(1).float()

    out = {
        # Slot occupancy: where in the utterance each role landed, normalised by length so
        # a 2 s and a 5 s clip are comparable.
        "slot_centroid": (centroid / lengths.unsqueeze(1)).cpu().numpy(),
        "slot_mass": data["audio_slot_mass"].cpu().numpy(),
        "slot_entropy": -(rows * torch.log(rows.clamp_min(1e-8))).sum(-1).cpu().numpy(),
        "role_attn": data["audio_role_attn"].cpu().numpy(),
        "audio_pred": data["audio_scores"].argmax(-1).cpu().numpy(),
        "audio_true": data["audio_cat_list"].reshape(-1).cpu().numpy(),
        "unique_multiple": data["unique_multiple_list"].reshape(-1).cpu().numpy(),
        # Utterance length in real (unpadded) frames. At 25 fps after pooling this is a
        # duration, and duration is the acoustic stand-in for the predecessor paper's
        # "linguistic complexity" analysis: a longer clip is a longer description with
        # more attributes and more relations to resolve.
        "audio_len": data["audio_len_list"].reshape(-1).cpu().numpy(),
        "n_objects": data["objectness_scores"].max(2)[1].sum(1).repeat_interleave(
            len_num_max).cpu().numpy(),
    }
    if "ot_plan" in data:
        weights = data["audio_role_attn"][:, AUDIO.ROLE_REFERENT]
        row = torch.einsum("bk,bkm->bm", weights, data["ot_plan"])
        row = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out["ot_peak"] = row.max(dim=-1)[0].cpu().numpy()
        out["ot_entropy"] = -(row * torch.log(row.clamp_min(1e-8))).sum(-1).cpu().numpy()
        # Margin between the best and second-best proposal: the direct measure of whether
        # the competition constraint broke the same-class tie.
        top2 = row.topk(2, dim=-1)[0]
        out["ot_margin"] = (top2[:, 0] - top2[:, 1]).cpu().numpy()

    out = {k: v[keep] for k, v in out.items()}
    out["iou"] = np.asarray(batch_ious, dtype=np.float32)
    return out


def summarise(ious, masks, audio_accuracy):
    """The 3DVG-Audio table: Unique / Multiple / Overall at IoU 0.25 and 0.5."""
    def rates(selection):
        if selection.sum() == 0:
            return 0.0, 0.0, 0
        subset = ious[selection]
        return (float((subset >= 0.25).mean() * 100),
                float((subset >= 0.5).mean() * 100),
                int(selection.sum()))

    unique = rates(masks == 0)
    multiple = rates(masks == 1)
    overall = rates(np.ones_like(masks, dtype=bool))

    print("\n" + "=" * 62)
    print(f"3DVG-Audio  fusion='{AUDIO.FUSION}'  encoder='{AUDIO.ENCODER}'  "
          f"slots={AUDIO.NUM_SLOTS}  rsk={AUDIO.USE_RSK}")
    print("=" * 62)
    print(f"{'':10s} {'Acc@0.25':>10s} {'Acc@0.5':>10s} {'n':>8s}")
    for name, (at25, at50, count) in (("Unique", unique), ("Multiple", multiple),
                                      ("Overall", overall)):
        print(f"{name:10s} {at25:10.2f} {at50:10.2f} {count:8d}")
    print("-" * 62)
    print(f"AudioCls top-1: {audio_accuracy * 100:.2f}%")
    print("reference (AP-Refer, Neurocomputing 611 2025, Table 2):")
    print(f"{'Unique':10s} {48.62:10.2f} {29.59:10.2f}")
    print(f"{'Multiple':10s} {16.94:10.2f} {9.96:10.2f}")
    print(f"{'Overall':10s} {23.09:10.2f} {13.77:10.2f}")
    print("=" * 62 + "\n")

    return {"unique": unique[:2], "multiple": multiple[:2], "overall": overall[:2],
            "audio_cls_acc": audio_accuracy}


def build_parser():
    parser = build_train_parser()
    parser.add_argument("--folder", type=str, required=True, help="Run under outputs/.")
    parser.add_argument("--checkpoint", type=str, default="model_criteria_25.pth")
    parser.add_argument("--force", action="store_true", help="Regenerate existing scores.")
    parser.add_argument("--no_nms", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--use_oracle", action="store_true")
    parser.add_argument("--use_cat_rand", action="store_true")
    parser.add_argument("--use_best", action="store_true")
    parser.add_argument("--audio_control", type=str, default="none", choices=CONTROLS,
                        help="Replace the audio at test time (diagnostic control):\n"
                             "silence, another object's utterance, time-reversed, or one "
                             "half of the clip.")
    parser.add_argument("--prosody_keep", type=str, default="all",
                        choices=list(PROSODY_GROUPS),
                        help="Keep only this prosodic cue group at test time; the other "
                             "channels are flattened to their per-utterance mean.")
    parser.add_argument("--out_subdir", type=str, default="",
                        help="Write scores/predictions under <run>/<out_subdir>/ so an "
                             "intervention does not overwrite the run's own scores.")
    parser.add_argument("--dump_diagnostics", action="store_true",
                        help="Also write diagnostics.p: per-utterance slot and transport\n"
                             "statistics that experiments/audio/analysis.py reads.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    args.detection = False
    args.lang_num_max = args.audio_lang_num_max
    restore_run_config(args)
    apply_audio(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cpu or not torch.cuda.is_available():
        force_cpu()
    evaluate(args)
