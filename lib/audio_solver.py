"""
AudioSolver: lib/solver.py with the 3D-VGA loss and the acoustic metrics.

Subclasses rather than forks. Everything the Solver does around a step -- the validation
cadence, checkpoint criteria, ETA, tensorboard, the best-model bookkeeping -- is the same
for both branches, and the only two things that differ are which loss is summed and which
extra numbers are logged. Those are exactly the two methods overridden here.
"""

import numpy as np
import torch

from lib.audio_config import AUDIO
from lib.audio_loss_helper import compute_audio_classification_accuracy, get_audio_loss
from lib.eval_helper import get_eval
from lib.solver import Solver


class AudioSolver(Solver):
    def _compute_loss(self, data_dict):
        _, data_dict = get_audio_loss(
            args=self.args,
            data_dict=data_dict,
            config=self.config,
            detection=self.detection,
            reference=self.reference,
        )

        self._running_log["ref_loss"] = data_dict["ref_loss"]
        # The Solver logs a "lang_loss" line; on this branch the comparable quantity is the
        # audio classification loss, so it goes in that slot rather than reporting a zero.
        self._running_log["lang_loss"] = data_dict["audio_cls_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["loss"] = data_dict["loss"]

        self._running_log["audio_cls_loss"] = data_dict["audio_cls_loss"]
        self._running_log["role_separation_loss"] = data_dict["role_separation_loss"]
        self._running_log["target_contrastive_loss"] = data_dict["target_contrastive_loss"]
        self._running_log["slot_entropy"] = data_dict["slot_entropy"]

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.config,
            reference=self.reference,
            use_lang_classifier=False,
        )

        # get_eval writes lang_acc from the text classifier, which this branch does not
        # have; substitute AudioCls top-1 so the logged number means something.
        self._running_log["lang_acc"] = compute_audio_classification_accuracy(data_dict).item()
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()

        self._running_log["ref_iou"] = data_dict["ref_iou"]
        self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])
        self._running_log["max_iou_rate_0.25"] = np.mean(data_dict["max_iou_rate_0.25"])
        self._running_log["max_iou_rate_0.5"] = np.mean(data_dict["max_iou_rate_0.5"])

        # Slot health, logged because a collapsed slot bank is invisible in the IoU numbers
        # until it has already cost several epochs: mass concentrated on one slot, or a
        # near-uniform transport plan, both mean the decomposition is not happening.
        with torch.no_grad():
            mass = data_dict["audio_slot_mass"]
            self._running_log["slot_mass_max"] = mass.max(dim=-1)[0].mean().item()
            if "ot_plan" in data_dict:
                weights = data_dict["audio_role_attn"][:, AUDIO.ROLE_REFERENT]
                plan = torch.einsum("bk,bkm->bm", weights, data_dict["ot_plan"])
                plan = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                self._running_log["ot_peak"] = plan.max(dim=-1)[0].mean().item()
            # How decisively the referent election settled on one slot. An election that
            # keeps drifting means the decomposition has not stabilised, and whatever
            # accuracy follows is not resting on the mechanism the method claims.
            self._running_log["role_peak"] = (
                data_dict["audio_role_attn"][:, AUDIO.ROLE_REFERENT].max(-1)[0].mean().item())
