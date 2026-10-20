"""
L_3DVG-Audio and the extra objectives the proposed model trains with.

The 3DVG-Audio paper's loss is

    L_3DVG-Audio = alpha * L_Det + beta * L_VG + gamma * L_AudioCls          (their Eq. 1)

with gamma = 1 chosen by ablation. That is reproduced here exactly -- ``get_audio_loss``
delegates the first two terms to lib/loss_helper.get_loss, so alpha/beta are this repo's
existing weights and the detection branch is supervised identically to the text model --
and then adds three terms the proposed method needs:

  L_slot   slot entropy + orthogonality. Keeps the K acoustic role slots from collapsing
           onto one loud frame or onto each other.

  L_role   separation between the two role elections, so the referent and the relation
           cannot resolve to the same slot.

  L_tgt    in-batch InfoNCE between the elected referent and the ground-truth proposal
           feature. Directly shapes the space the OT routing computes its cost in.

NO TEXT APPEARS ANYWHERE IN THIS FILE, and none should. The audio in 3DVG-AudioSet is
synthesised from ScanRefer text, so a transcript and its LLM parse are available at
training time and a distillation term would be easy to add and would probably help. It is
refused on purpose: a decomposition learned by imitating a text parse is not evidence that
the decomposition can be induced from speech, and speech that was never text is the case
the task exists for.

That refusal has a consequence worth stating plainly. The three terms above, the time
prior and the slot orthogonality are the ONLY things shaping the decomposition, and only
L_tgt is direct supervision -- from geometry, not language. Whether the roles actually
separate is therefore an empirical question every run has to answer, which is what
experiments/audio/analysis.py slot-roles is for.
"""

import torch
import torch.nn.functional as F

from lib.audio_config import AUDIO
from lib.loss_helper import get_loss as get_text_loss


def _valid_utterance_mask(data_dict):
    """(B*L,) bool over the flattened annotation axis, True on real annotations.

    The dataset pads a scene's annotation list up to ``lang_num_max`` by repeating the last
    entry, so without this every loss double-counts whatever that entry happens to be --
    badly, in scenes with two annotations and a max of eight.
    """
    lang_num = data_dict["lang_num"]                                   # (B,)
    batch_size = lang_num.shape[0]
    len_num_max = data_dict["audio_feat_list"].shape[1]
    index = torch.arange(len_num_max, device=lang_num.device).unsqueeze(0)
    return (index < lang_num.unsqueeze(1)).reshape(batch_size * len_num_max)


def compute_audio_classification_loss(data_dict):
    """L_AudioCls: cross-entropy over the 18 ScanNet classes (paper Eq. 2)."""
    scores = data_dict["audio_scores"]                                 # (B*L, C)
    target = data_dict["audio_cat_list"].reshape(-1)                   # (B*L,)
    valid = _valid_utterance_mask(data_dict)
    if valid.sum() == 0:
        return scores.sum() * 0.0
    return F.cross_entropy(scores[valid], target[valid])


def compute_slot_regularisers(data_dict):
    """Entropy on the frame->slot plan, and orthogonality between slot vectors.

    Entropy is *maximised* (hence the negative weight in the sum below) only in the sense
    of being kept away from zero: a collapsed plan early in training gives the slots no
    gradient to recover with. Orthogonality is the direct expression of "these slots should
    carry different roles".
    """
    plan = data_dict["audio_slot_plan"]                                # (B*L, K, T)
    slots = data_dict["audio_slots"]                                   # (B*L, K, H)
    valid = _valid_utterance_mask(data_dict)
    if valid.sum() == 0:
        zero = plan.sum() * 0.0
        return zero, zero

    plan = plan[valid]
    rows = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(rows * torch.log(rows.clamp_min(1e-8))).sum(dim=-1).mean()

    normed = F.normalize(slots[valid], dim=-1)                         # (N, K, H)
    gram = torch.bmm(normed, normed.transpose(1, 2))                   # (N, K, K)
    eye = torch.eye(gram.shape[1], device=gram.device).unsqueeze(0)
    ortho = ((gram - eye) ** 2).mean()

    return entropy, ortho


def compute_role_separation_loss(data_dict):
    """Penalise overlap between the referent and relation elections.

    Without it nothing stops both role queries from electing the same slot, which
    would hand RSK the referent vector and quietly turn the geometric kernel into a
    second copy of the target pathway. The penalty is the summed product of the two
    attention distributions -- zero when they are disjoint, one when identical.
    """
    attn = data_dict.get("audio_role_attn")
    if attn is None or attn.shape[1] < 2:
        return data_dict["audio_slots"].sum() * 0.0
    valid = _valid_utterance_mask(data_dict)
    if valid.sum() == 0:
        return attn.sum() * 0.0
    attn = attn[valid]
    return (attn[:, AUDIO.ROLE_REFERENT] * attn[:, AUDIO.ROLE_RELATION]).sum(-1).mean()


def compute_target_contrastive_loss(data_dict):
    """In-batch InfoNCE between the elected referent and its ground-truth proposal feature.

    The positive is the proposal the reference loss treats as correct (the highest-IoU
    proposal, from ``cluster_labels``); negatives are the other utterances' positives.
    This is what gives the OT cost matrix a metric worth transporting over -- without it,
    slot/proposal cosine similarity is only ever shaped indirectly, through the scoring MLP.
    """
    if "fused_proposals" not in data_dict or "cluster_labels" not in data_dict:
        return data_dict["audio_slots"].sum() * 0.0

    proposals = data_dict["fused_proposals"]                           # (B*L, M, H)
    labels = data_dict["cluster_labels"].reshape(proposals.shape[0], -1)
    valid = _valid_utterance_mask(data_dict) & (labels.sum(dim=-1) > 0)
    if valid.sum() < 2:
        return proposals.sum() * 0.0

    positives = proposals[valid][torch.arange(int(valid.sum()), device=proposals.device),
                                 labels[valid].argmax(dim=-1)]         # (N, H)
    anchors = F.normalize(data_dict["audio_referent"][valid], dim=-1)
    positives = F.normalize(positives, dim=-1)

    logits = anchors @ positives.t() / AUDIO.CONTRASTIVE_TEMPERATURE
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def get_audio_loss(args, data_dict, config, detection=True, reference=True, progress=0.0):
    """L_3DVG-Audio plus the proposed terms. Mirrors lib/loss_helper.get_loss's contract.

    ``progress`` is accepted and unused: it annealed the distillation weight, and there is
    no distillation any more. Kept in the signature so the solver and the eval script do
    not need a second call shape.
    """
    del progress
    # alpha * L_Det + beta * L_VG, with this repo's existing weights and detector losses.
    # use_lang_classifier=False: that head reads the text branch's GloVe embedding, which
    # this model does not have and must not have.
    _, data_dict = get_text_loss(args, data_dict, config, detection=detection,
                                 reference=reference, use_lang_classifier=False)
    base_loss = data_dict["loss"]

    audio_cls = compute_audio_classification_loss(data_dict)
    entropy, ortho = compute_slot_regularisers(data_dict)
    role_separation = compute_role_separation_loss(data_dict)
    target_contrastive = compute_target_contrastive_loss(data_dict)

    # Entropy enters with a minus sign: it is kept high, not driven to zero.
    loss = (base_loss
            + AUDIO.LOSS_AUDIO_CLS * audio_cls
            - AUDIO.LOSS_SLOT_ENTROPY * entropy
            + AUDIO.LOSS_SLOT_ORTHO * ortho
            + AUDIO.LOSS_ROLE_SEPARATION * role_separation
            + AUDIO.LOSS_TARGET_CONTRASTIVE * target_contrastive)

    data_dict["audio_cls_loss"] = audio_cls
    data_dict["slot_entropy"] = entropy
    data_dict["slot_ortho_loss"] = ortho
    data_dict["role_separation_loss"] = role_separation
    data_dict["target_contrastive_loss"] = target_contrastive
    data_dict["loss"] = loss

    return loss, data_dict


def compute_audio_classification_accuracy(data_dict):
    """Top-1 AudioCls accuracy, logged next to ref_acc so a stalled acoustic branch is
    visible in the training log rather than only in the final IoU."""
    valid = _valid_utterance_mask(data_dict)
    if valid.sum() == 0:
        return torch.zeros(1, device=data_dict["audio_scores"].device)[0]
    predicted = data_dict["audio_scores"][valid].argmax(dim=-1)
    target = data_dict["audio_cat_list"].reshape(-1)[valid]
    return (predicted == target).float().mean()
