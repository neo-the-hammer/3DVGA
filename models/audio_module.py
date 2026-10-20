"""
AudioModule: the acoustic counterpart of models/lang_module.py.

LangModule's real contribution in this repo is not the GRU -- it is that an LLM has already
split the sentence into ``target`` / ``adjectives`` / ``neighbors``, and the fusion head
then runs one cross-attention stream per field. That decomposition is where the accuracy
comes from, and it is the first thing 3DVG-Audio loses: you cannot run GPT-4o-mini on a
waveform. Transcribing first would work, but it puts an ASR system back on the critical
path and hands the task's error budget to word-error rate.

So this module induces the decomposition acoustically. K learnable slot queries compete for
the utterance's frames under a Sinkhorn (doubly-stochastic) constraint, which forces them
to *partition* the audio instead of all landing on the same loud syllable. Three biases
steer the partition toward linguistic roles without any labels:

  * a per-slot learnable Gaussian over normalised time -- English referring expressions are
    ordered, so the slots start life spread across the utterance;
  * frame saliency from the prosody-aware encoder -- accented frames are content words;
  * an orthogonality penalty between slot vectors, so they cannot converge on one role.

Nothing here reads text, at training time or any other time. There is no transcript, no
ASR pass and no distillation: the decomposition is induced from the acoustic signal alone,
and the only direct supervision any role receives comes from GEOMETRY -- the in-batch
contrastive term that ties the elected referent to the ground-truth proposal.

That is a harder problem than the supervised alternative (Audio-3DVG's Object Mention
Detection classifies mentioned object classes against a text-derived label set), and it is
the point: a class vocabulary cannot represent "brown" or "next to", and it discards word
order. It is also more fragile, which is why analysis.py slot-roles exists.

Interface mirrors LangModule so the fusion head can stay symmetric:

    data_dict["audio_fea"]            (B*L, T, H)   contextual frames
    data_dict["audio_slots"]          (B*L, K, H)   role slots
    data_dict["audio_slot_plan"]      (B*L, K, T)   frame -> slot transport plan
    data_dict["audio_slot_mass"]      (B*L, K)      how much of the utterance each slot won
    data_dict["audio_role_attn"]      (B*L, R, K)   which slot won each role
    data_dict["audio_referent"]       (B*L, H)      elected referent vector
    data_dict["audio_relation"]       (B*L, H)      elected relation vector -> RSK
    data_dict["audio_emb"]            (B*L, H)      global utterance embedding
    data_dict["audio_attention_mask"] (B*L, 1, 1, T)
    data_dict["audio_scores"]         (B*L, n_class) AudioCls head -> L_AudioCls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.audio_config import AUDIO
from models.audio_encoder import AudioFrameEncoder
from models.fusion.sinkhorn import plan_to_attention, sinkhorn_log
from models.transformer.utils import PositionWiseFeedForward


class SlotInduction(nn.Module):
    """Frames -> K role slots, by entropic optimal transport."""

    def __init__(self, hidden_size, num_slots, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots

        self.queries = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)

        # Slot logits are cosine similarities times this, so the cost range is bounded and
        # Sinkhorn's convergence does not drift as the encoder's feature norms grow. An
        # unbounded dot product is the failure here: at a cost range of +-10 the marginals
        # are 40% off after 8 iterations, whatever eps is set to. Sharpness is bought with
        # this scale (up to 2.0), not by lowering eps.
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Learnable Gaussian prior over normalised time, one per slot. Means start evenly
        # spread over the utterance; sigma starts wide (~0.3) so the prior is a nudge and
        # not a cage, and both are free to move.
        self.time_mu = nn.Parameter(torch.linspace(0.15, 0.85, num_slots))
        self.time_log_sigma = nn.Parameter(torch.full((num_slots,), -1.2))
        self.prior_gate = nn.Parameter(torch.tensor(1.0))
        self.saliency_gate = nn.Parameter(torch.tensor(0.5))

        # Per-utterance slot budget: how much of this utterance each role should claim.
        # This is the module's way of saying "no relation was spoken" -- that slot simply
        # wins very little mass, and everything downstream scales with it.
        self.mass_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_slots),
        )

        self.refine = PositionWiseFeedForward(d_model=hidden_size, d_ff=hidden_size * 2,
                                              dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features, saliency, mask):
        """
        Args:
            features: (B, T, H)
            saliency: (B, T)   -1e4 on padding
            mask:     (B, T)   bool

        Returns:
            slots (B, K, H), plan (B, K, T), mass (B, K), attention (B, K, T)
        """
        batch, max_len = features.shape[:2]
        device = features.device

        queries = F.normalize(self.to_q(self.queries), dim=-1)                 # (K, H)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)                   # (B, K, H)
        keys = F.normalize(self.to_k(features), dim=-1)                       # (B, T, H)
        values = self.to_v(features)

        scale = self.logit_scale.clamp(0.25, 2.0)
        scores = torch.bmm(queries, keys.transpose(1, 2)) * scale             # (B, K, T)

        # Every additive bias below is bounded too, for the same reason: an unbounded term
        # would put the total cost range back outside what Sinkhorn closes in 8 iterations.
        if AUDIO.SLOT_TIME_PRIOR:
            lengths = mask.sum(dim=1).clamp_min(1).float()                    # (B,)
            steps = torch.arange(max_len, device=device, dtype=torch.float32)
            position = steps.unsqueeze(0) / lengths.unsqueeze(1)              # (B, T)
            mu = self.time_mu.view(1, -1, 1)
            sigma = self.time_log_sigma.exp().clamp_min(0.05).view(1, -1, 1)
            prior = -((position.unsqueeze(1) - mu) ** 2) / (2.0 * sigma ** 2)
            scores = scores + self.prior_gate.clamp(0.0, 2.0) * prior.clamp_min(-4.0)

        # Saliency is shared across slots: it says which frames are worth claiming at all,
        # not which slot should claim them. tanh, because the encoder's saliency head is a
        # plain linear layer and nothing else bounds it.
        scores = scores + self.saliency_gate.clamp(0.0, 2.0) * torch.tanh(saliency).unsqueeze(1)

        pair_mask = mask.unsqueeze(1).expand(-1, self.num_slots, -1)          # (B, K, T)

        # Row marginal: a softmax over slots, so the budget is a *relative* allocation and
        # the row and column totals agree (both 1). Pooling is masked -- padded frames are
        # zeroed by the encoder, but the divisor still has to be the real length.
        pooled = (features * mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        row_marginal = torch.softmax(self.mass_head(pooled), dim=-1)          # (B, K)

        plan = sinkhorn_log(scores, pair_mask, row_marginal=row_marginal,
                            eps=AUDIO.SLOT_SINKHORN_EPS,
                            n_iters=AUDIO.SLOT_SINKHORN_ITERS)                # (B, K, T)

        attention = plan_to_attention(plan, dim=-1)                           # rows sum to 1
        slots = torch.bmm(attention, values)                                  # (B, K, H)
        slots = self.norm(slots + self.queries.unsqueeze(0))                  # residual on the query
        slots = self.refine(slots)

        mass = plan.sum(dim=-1)                                               # (B, K)
        return slots, plan, mass, attention


class AudioModule(nn.Module):
    """Frozen-encoder frames in, role slots and an utterance embedding out."""

    def __init__(self, num_text_classes, in_dim=None, hidden_size=None, num_slots=None,
                 dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        self.num_slots = num_slots or AUDIO.NUM_SLOTS
        self.num_text_classes = num_text_classes

        self.encoder = AudioFrameEncoder(in_dim=in_dim, hidden_size=self.hidden_size,
                                         dropout=dropout)
        self.slots = SlotInduction(self.hidden_size, self.num_slots, dropout=dropout)

        # ROLE ELECTION.  Two learned queries attend over the slot bank and pool it into a
        # referent vector and a relation vector.
        #
        # This is not decoration over a simpler `slots[:, 0]` and `slots[:, 3]`. Those
        # indices were only meaningful while a distillation loss pinned them there from the
        # text parse. The model is audio-only now, so nothing anchors slot number 3 to the
        # concept "relation" -- reading it as one would be an unenforced assumption sitting
        # directly underneath the geometric kernel, which is the component most sensitive
        # to getting the wrong vector. Which slot plays which role has to be learned, so it
        # is learned, and analysis.py reports the election so the assumption is auditable.
        self.role_queries = nn.Parameter(torch.randn(AUDIO.NUM_ROLES, self.hidden_size) * 0.02)
        self.role_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.role_scale = nn.Parameter(torch.tensor(1.0))

        # AudioCls, i.e. Eq. (3) of the 3DVG-Audio paper: MLP(Mean(F_audio)) over the 18
        # ScanNet classes. Kept identical so the L_AudioCls ablation is comparable to
        # theirs -- except that the pooled vector here is the *elected referent*,
        # concatenated with the mean, since "which object class was named" is a property of
        # the referring phrase and not of the whole utterance.
        self.audio_cls = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_text_classes),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(self, data_dict):
        frames = data_dict["audio_feat_list"]          # (B, L, T, C)
        prosody = data_dict["prosody_list"]            # (B, L, T, P)
        lengths = data_dict["audio_len_list"]          # (B, L)

        batch_size, len_num_max, max_len = frames.shape[:3]
        frames = frames.reshape(batch_size * len_num_max, max_len, -1)
        prosody = prosody.reshape(batch_size * len_num_max, max_len, -1)
        lengths = lengths.reshape(batch_size * len_num_max)

        features, saliency, mask, attn_mask = self.encoder(frames, prosody, lengths)
        slots, plan, mass, slot_attn = self.slots(features, saliency, mask)

        # Slot dropout: zero whole slots at random so no single role becomes load-bearing.
        # The text branch gets the same treatment by word masking in lang_module.py; the
        # acoustic analogue has to act at slot granularity, because masking frames would
        # just be SpecAugment again (already applied in the dataset).
        if self.training and AUDIO.SLOT_DROPOUT > 0:
            keep = (torch.rand(slots.shape[0], self.num_slots, 1, device=slots.device)
                    >= AUDIO.SLOT_DROPOUT).float()
            slots = slots * keep
            mass = mass * keep.squeeze(-1)

        pooled = (features * mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).float()

        # Elect the roles. Cosine logits times a bounded scale, for the same reason the
        # slot costs are bounded: an unbounded dot product would drive this softmax to a
        # hard one-hot early, before any slot means anything.
        queries = F.normalize(self.role_queries, dim=-1)                  # (R, H)
        keys = F.normalize(self.role_proj(slots), dim=-1)                 # (B', K, H)
        role_logits = torch.einsum("rh,bkh->brk", queries, keys)
        role_attn = torch.softmax(role_logits * self.role_scale.clamp(0.5, 4.0), dim=-1)
        if not AUDIO.ROLE_ELECTION:
            # Ablation: go back to reading fixed slot indices, which is what the earlier
            # distilled drafts did. One-hot rows keep every downstream consumer --
            # audio_role_attn contracts the transport plan, and the role-separation loss
            # reads the same tensor -- working on the same shape, so this arm differs from
            # the proposed one in exactly one thing: whether the roles are chosen.
            fixed = torch.zeros_like(role_attn)
            for r in range(AUDIO.NUM_ROLES):
                fixed[:, r, min(r, fixed.shape[-1] - 1)] = 1.0
            role_attn = fixed
        role_vecs = torch.bmm(role_attn, slots)                           # (B', R, H)
        referent = role_vecs[:, AUDIO.ROLE_REFERENT]
        relation = role_vecs[:, AUDIO.ROLE_RELATION]

        data_dict["audio_fea"] = features
        data_dict["audio_slots"] = slots
        data_dict["audio_slot_plan"] = plan
        data_dict["audio_slot_mass"] = mass
        data_dict["audio_slot_attn"] = slot_attn
        data_dict["audio_frame_mask"] = mask
        data_dict["audio_attention_mask"] = attn_mask
        data_dict["audio_saliency"] = saliency
        data_dict["audio_pooled"] = pooled
        data_dict["audio_role_attn"] = role_attn                          # (B', R, K)
        data_dict["audio_referent"] = referent                            # (B', H)
        data_dict["audio_relation"] = relation                            # (B', H)
        data_dict["audio_emb"] = self.global_proj(torch.cat([pooled, referent], dim=-1))
        data_dict["audio_scores"] = self.audio_cls(torch.cat([pooled, referent], dim=-1))

        return data_dict
