"""
RSK -- Relation-Slot Geometric Kernel.

models/match_module.py biases proposal self-attention with a fixed geometric prior:

    dist_weights = cat([-d, -d, 0, 0])          # d = pairwise centre distance

Two heads are told "nearer is more relevant", two are left alone. It is a good prior and it
is the same prior for every utterance -- which is the problem. "The chair *next to* the
table" and "the chair *on the far side of* the room" want opposite distance profiles, and
"the lamp *above* the desk" wants a direction, not a distance at all. A fixed -d kernel
answers all three the same way, and the cost lands almost entirely on the Multiple subset,
where relations are the only thing separating same-class objects (AP-Refer: 16.94 / 9.96,
against 48.62 / 29.59 on Unique).

So make the prior a function of what was said. A hypernetwork reads the *relation slot*
from the acoustic branch and emits, per attention head, a spatial kernel:

    radial      = -(d_ij - mu)^2 / (2 sigma^2)        a distance *band*, not a monotone decay
    directional = kappa * (u . r_ij)                  a preferred direction on the unit sphere
    bias_ij     = lambda * (radial + directional) + (1 - lambda) * (-d_ij)

mu is where the relation wants its anchor ("next to" -> small, "across from" -> large);
u picks out "above" (+z), "under" (-z), "left of" (in-plane); lambda blends against the
original prior. lambda is initialised at ~0, so at step zero the module *is* the repo's
existing -d prior and training only ever moves away from it if that helps. A new geometric
prior that starts out worse than the old one is a slow way to lose, and this avoids it.

Shapes are chosen to drop straight into ScaledDotProductAttention(..., way='add'), so the
existing attention stack does not change at all.
"""

import torch
import torch.nn as nn

from lib.audio_config import AUDIO


class RelationSlotKernel(nn.Module):
    """Relation slot -> per-head spatial attention bias over proposal pairs."""

    def __init__(self, hidden_size, num_heads=4, min_sigma=None, max_mu=None):
        super().__init__()
        self.num_heads = num_heads
        self.min_sigma = min_sigma if min_sigma is not None else AUDIO.RSK_MIN_SIGMA
        self.max_mu = max_mu if max_mu is not None else AUDIO.RSK_MAX_MU

        # 7 parameters per head: direction (3), log-distance-band centre, log-band width,
        # directional sharpness, blend gate.
        self.params_per_head = 7
        self.hyper = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_heads * self.params_per_head),
        )
        # Zero-init the last layer and bias the gate strongly negative: sigmoid(-4) ~ 0.018,
        # so the module opens as the original -d prior and earns its way out of it.
        nn.init.zeros_(self.hyper[-1].weight)
        bias = torch.zeros(num_heads * self.params_per_head)
        bias.view(num_heads, self.params_per_head)[:, 6] = -4.0
        with torch.no_grad():
            self.hyper[-1].bias.copy_(bias)

    def forward(self, relation_slot, centers, len_num_max, objectness_mask=None):
        """
        Args:
            relation_slot:  (B * L, H) the acoustic relation slot.
            centers:        (B, M, 3) proposal centres (scene-level, shared across the L
                            annotations of that scene).
            len_num_max:    L, annotations per scene in this batch.
            objectness_mask:(B, M) bool, True on real objects. Optional.

        Returns:
            (B * L, num_heads, M, M) additive attention bias.
        """
        batch, num_proposals = centers.shape[:2]
        heads = self.num_heads

        params = self.hyper(relation_slot)                                  # (B*L, h*7)
        params = params.view(batch, len_num_max, heads, self.params_per_head)

        direction = params[..., 0:3]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mu = torch.sigmoid(params[..., 3]) * self.max_mu                    # (B, L, h)
        sigma = torch.nn.functional.softplus(params[..., 4]) + self.min_sigma
        kappa = torch.nn.functional.softplus(params[..., 5])
        blend = torch.sigmoid(params[..., 6])

        # Pairwise geometry, computed once per scene rather than per annotation: this is
        # the tensor that would otherwise dominate memory at L > 8.
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)                 # (B, M, M, 3) = c_j - c_i
        distance = delta.norm(dim=-1)                                       # (B, M, M)
        unit = delta / distance.unsqueeze(-1).clamp_min(1e-6)

        radial = -((distance[:, None, None] - mu[..., None, None]) ** 2)
        radial = radial / (2.0 * (sigma[..., None, None] ** 2))             # (B, L, h, M, M)

        directional = torch.einsum("bmnk,blhk->blhmn", unit, direction)
        directional = kappa[..., None, None] * directional

        learned = radial + directional
        baseline = -distance[:, None, None]                                 # the repo's prior
        bias = blend[..., None, None] * learned + (1.0 - blend[..., None, None]) * baseline

        if objectness_mask is not None:
            # Keys that are not objects should not attract mass, whatever the relation says.
            key_mask = objectness_mask[:, None, None, None, :]              # (B,1,1,1,M)
            bias = bias.masked_fill(~key_mask, -1e4)

        return bias.reshape(batch * len_num_max, heads, num_proposals, num_proposals)


def default_distance_bias(centers, num_heads=4):
    """models/match_module.py's fixed prior, as a function, for the --no_rsk ablation.

    Reproduces cat([-d, -d, 0, 0]) exactly for num_heads=4, and generalises it by giving
    the first half of the heads the distance prior and leaving the rest free.
    """
    delta = centers.unsqueeze(1) - centers.unsqueeze(2)
    distance = delta.norm(dim=-1).unsqueeze(1)                              # (B, 1, M, M)
    half = max(1, num_heads // 2)
    return torch.cat([(-distance).expand(-1, half, -1, -1),
                      torch.zeros_like(distance).expand(-1, num_heads - half, -1, -1)],
                     dim=1).detach()
