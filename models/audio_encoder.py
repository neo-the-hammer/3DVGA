"""
Prosody-Augmented Audio Frame Encoder (PA-AFE).

Takes frozen wav2vec2/WavLM/HuBERT frames from the cache and turns them into the model's
working width, with three things the frozen encoder does not provide:

  * **Prosodic conditioning.** Self-supervised speech encoders are trained to be invariant
    to pitch and loudness -- that is what makes them good phone recognisers and exactly
    what loses the cue this task needs. In "the *chair* next to the table", the pitch accent
    on the head noun is the speaker telling you which word is the referent, and the pause
    before "next to" is the speaker telling you where the relational clause starts. The
    prosody stream is injected by FiLM rather than concatenated, so six channels can
    modulate 768 without being drowned by them.

  * **Position.** Cross-attention is permutation-invariant, but a referring expression is
    strongly ordered: the target noun comes early, the anchor late. Frames carry a
    normalised-time encoding so the slot module downstream can use that order.

  * **Saliency.** A scalar per frame, learned, marking content-bearing frames. Roughly half
    of any utterance is function words and silence; this is what lets the slot module spend
    its attention budget on the rest.
"""

import torch
import torch.nn as nn

from lib.audio_config import AUDIO, PROSODY_GROUPS
from models.transformer.attention import MultiHeadAttention


class FiLM(nn.Module):
    """Feature-wise linear modulation: h <- gamma(p) * h + beta(p).

    gamma is centred at 1 and beta at 0 (the last layer is zero-initialised), so the module
    starts as the identity. Without that, a randomly initialised FiLM multiplies the
    encoder features by noise for the first few hundred steps and the run never recovers.
    """

    def __init__(self, cond_dim, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return features * (1.0 + gamma) + beta


def time_encoding(lengths, max_len, dim, device):
    """Sinusoidal encoding of *normalised* position t/T, not absolute frame index.

    Absolute position would make the model read a 2 s utterance and a 5 s one differently
    at the same word; normalised position keeps "one third of the way in" meaning the same
    thing at any speaking rate. Speaking rate itself is not lost -- it survives in the
    prosody stream.
    """
    steps = torch.arange(max_len, device=device, dtype=torch.float32)          # (T,)
    denom = lengths.clamp_min(1).to(torch.float32).unsqueeze(1)                # (B, 1)
    position = (steps.unsqueeze(0) / denom).clamp(0.0, 1.0)                    # (B, T)

    half = dim // 2
    freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                     * (-torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1)))
    angle = position.unsqueeze(-1) * freq.view(1, 1, -1) * 10.0                # (B, T, half)

    encoding = torch.zeros(position.shape[0], max_len, dim, device=device)
    encoding[..., 0:2 * half:2] = torch.sin(angle)
    encoding[..., 1:2 * half:2] = torch.cos(angle)
    return encoding


def length_mask(lengths, max_len):
    """(B, T) bool, True on real frames."""
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


class AudioFrameEncoder(nn.Module):
    """(B, T, C_enc) frozen frames + (B, T, P) prosody -> (B, T, H) contextual frames."""

    def __init__(self, in_dim=None, hidden_size=None, prosody_dim=None, num_heads=None,
                 depth=2, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim or AUDIO.ENCODER_DIM
        self.hidden_size = hidden_size or AUDIO.HIDDEN_SIZE
        # PROSODY_CHANNELS selects a cue group (energy / pitch / spectral / all), so the
        # ablation can say WHICH prosodic cue carries the signal rather than only that
        # prosody as a whole helps. The selection is on the channel axis, so the FiLM
        # generator shrinks with it and the arms differ in more than a zeroed input.
        self.prosody_channels = list(PROSODY_GROUPS[AUDIO.PROSODY_CHANNELS])
        self.prosody_dim = len(self.prosody_channels) or (prosody_dim or AUDIO.PROSODY_DIM)
        self.use_prosody = bool(AUDIO.USE_PROSODY) and bool(self.prosody_channels)
        heads = num_heads or AUDIO.NUM_HEADS

        self.project = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_size),
        )

        if self.use_prosody:
            self.prosody_norm = nn.LayerNorm(self.prosody_dim)
            self.film = FiLM(self.prosody_dim, self.hidden_size)

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=self.hidden_size,
                               d_k=self.hidden_size // heads,
                               d_v=self.hidden_size // heads,
                               h=heads, dropout=dropout)
            for _ in range(depth)
        )

        # Saliency is read off the frame *and* its prosody, so a loud stressed frame can be
        # marked even when its phonetic content is unremarkable.
        saliency_in = self.hidden_size + (self.prosody_dim if self.use_prosody else 0)
        self.saliency = nn.Sequential(
            nn.Linear(saliency_in, self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.out_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, frames, prosody, lengths):
        """
        Args:
            frames:  (B, T, C_enc) float
            prosody: (B, T, P) float
            lengths: (B,) long, real frame count

        Returns:
            features:  (B, T, H)   contextual frame features
            saliency:  (B, T)      per-frame content score (log-scale, unbounded)
            mask:      (B, T)      bool, True on real frames
            attn_mask: (B, 1, 1, T) bool, True where MultiHeadAttention should mask
        """
        batch, max_len = frames.shape[:2]
        mask = length_mask(lengths, max_len)

        features = self.project(frames)
        features = features + time_encoding(lengths, max_len, self.hidden_size, frames.device)

        if self.use_prosody:
            if len(self.prosody_channels) != prosody.shape[-1]:
                prosody = prosody[..., self.prosody_channels]
            prosody = self.prosody_norm(prosody)
            features = self.film(features, prosody)

        # MultiHeadAttention's convention: True means "mask this out".
        attn_mask = (~mask).view(batch, 1, 1, max_len)
        for layer in self.self_attn:
            features = layer(features, features, features, attn_mask)

        features = self.out_norm(features)
        # Zero the padded tail so anything pooling over frames without re-masking still
        # gets the right answer.
        features = features * mask.unsqueeze(-1).float()

        saliency_in = torch.cat([features, prosody], dim=-1) if self.use_prosody else features
        saliency = self.saliency(saliency_in).squeeze(-1)
        saliency = saliency.masked_fill(~mask, -1e4)

        return features, saliency, mask, attn_mask
