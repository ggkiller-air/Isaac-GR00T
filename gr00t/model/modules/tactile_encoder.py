# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tactile (skin-suit) encoder and touch-dreaming heads for GR00T N1.7.

Grafts the three HTD (arXiv:2604.13015) components onto the GR00T action head
without touching the pretrained VLM+DiT:

(A) :class:`TactileEncoder` -- per-region MLPs + a learnable-slot cross-attention
    aggregator turn a raw ``uint8[raw_dim]`` skin packet into ``N`` tactile
    tokens of width ``embedding_dim`` (so they can be concatenated into the DiT
    ``sa_embs`` sequence). It owns the 256->112 valid-channel select and the
    ``/255`` normalization, so the data pipeline only forwards the raw packet.
(C) :class:`TactileDreamHead` + :func:`touch_dreaming_loss` + an EMA target copy
    of the encoder implement the touch-dreaming auxiliary task: predict the
    *future* tactile latent from the shared trunk features. Per the paper this
    is the component that produces a stable gain; predicting latents (not raw)
    and a magnitude term to prevent collapse are both deliberate.

The dream head is training-only; inference runs just the encoder, so there is no
deployment overhead.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from gr00t.model.modules.embodiment_conditioned_mlp import SmallMLP


class PerRegionTactileEncoder(nn.Module):
    """Encode each body region independently, then stack into region tokens.

    Input is the flat vector of valid channels (region-ordered); it is split by
    ``region_sizes`` and each slice goes through its own 2-layer MLP projecting
    to ``output_dim``. Output: ``[B, num_regions, output_dim]``. No 2D spatial
    layout is assumed (MLP, not CNN), matching the irregular per-region sensel
    counts ``[48, 40, 8, 4, 8, 4]``.
    """

    def __init__(self, region_sizes: list[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.region_sizes = list(region_sizes)
        self.branches = nn.ModuleList(
            [SmallMLP(size, hidden_dim, output_dim) for size in self.region_sizes]
        )

    def forward(self, x_valid: torch.Tensor) -> torch.Tensor:
        # x_valid: [B, sum(region_sizes)] -> [B, num_regions, output_dim]
        parts = torch.split(x_valid, self.region_sizes, dim=-1)
        tokens = [branch(part) for branch, part in zip(self.branches, parts)]
        return torch.stack(tokens, dim=1)


class TactileSlotAggregator(nn.Module):
    """Aggregate variable region tokens into a fixed set of ``N`` slot tokens.

    ``N`` learnable query vectors cross-attend the region tokens, decoupling the
    output token count from the number of regions / raw channels.
    """

    def __init__(self, embed_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, region_tokens: torch.Tensor) -> torch.Tensor:
        # region_tokens: [B, num_regions, embed_dim] -> [B, num_tokens, embed_dim]
        batch = region_tokens.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1).to(region_tokens.dtype)
        attended, _ = self.attn(query, region_tokens, region_tokens, need_weights=False)
        return self.norm(attended)


class TactileEncoder(nn.Module):
    """Raw skin packet -> tactile tokens, owning valid-select + ``/255``.

    ``forward`` maps a single-timestep raw packet ``[B, raw_dim]`` to tactile
    tokens ``[B, num_tokens, embed_dim]`` for injection into ``sa_embs``.
    ``encode_pooled`` maps one-or-many timesteps to a pooled latent (mean over
    slot tokens), used to build touch-dreaming targets from future frames.
    """

    def __init__(
        self,
        raw_dim: int,
        valid_idx: list[int],
        region_sizes: list[int],
        embed_dim: int,
        num_tokens: int,
        hidden_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        assert sum(region_sizes) == len(valid_idx), (
            f"region_sizes sum {sum(region_sizes)} != #valid channels {len(valid_idx)}"
        )
        self.raw_dim = raw_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        # Non-persistent: derived from config, no need to store in state_dict.
        self.register_buffer(
            "valid_idx", torch.as_tensor(valid_idx, dtype=torch.long), persistent=False
        )
        self.per_region = PerRegionTactileEncoder(region_sizes, hidden_dim, embed_dim)
        self.aggregator = TactileSlotAggregator(embed_dim, num_tokens, num_heads)

    def select_and_normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """``[..., raw_dim]`` (0-255) -> ``[..., num_valid]`` in ``[0, 1]``."""
        valid = raw.index_select(-1, self.valid_idx)
        return valid.to(self.aggregator.norm.weight.dtype) / 255.0

    def forward(self, raw_current: torch.Tensor) -> torch.Tensor:
        # raw_current: [B, raw_dim] -> [B, num_tokens, embed_dim]
        x_valid = self.select_and_normalize(raw_current)
        region_tokens = self.per_region(x_valid)
        return self.aggregator(region_tokens)

    def encode_pooled(self, raw: torch.Tensor) -> torch.Tensor:
        """``[B, T, raw_dim]`` (or ``[B, raw_dim]``) -> pooled latent.

        Returns ``[B, T, embed_dim]`` (or ``[B, embed_dim]``), mean-pooled over
        the slot tokens. Used to encode future frames into dreaming targets.
        """
        squeeze_time = raw.dim() == 2
        if squeeze_time:
            raw = raw.unsqueeze(1)
        batch, time = raw.shape[0], raw.shape[1]
        flat = raw.reshape(batch * time, raw.shape[-1])
        tokens = self.forward(flat)  # [B*T, num_tokens, embed_dim]
        pooled = tokens.mean(dim=1).reshape(batch, time, self.embed_dim)
        return pooled.squeeze(1) if squeeze_time else pooled


class TactileDreamHead(nn.Module):
    """Predict future tactile latents from the shared trunk feature.

    Input is a pooled trunk feature at the tactile token positions; output is
    ``[B, dream_horizon, latent_dim]`` -- one predicted latent per future step.
    """

    def __init__(self, in_dim: int, latent_dim: int, dream_horizon: int, hidden_dim: int):
        super().__init__()
        self.dream_horizon = dream_horizon
        self.latent_dim = latent_dim
        self.net = SmallMLP(in_dim, hidden_dim, dream_horizon * latent_dim)

    def forward(self, trunk_feature: torch.Tensor) -> torch.Tensor:
        # trunk_feature: [B, in_dim] -> [B, dream_horizon, latent_dim]
        out = self.net(trunk_feature)
        return out.view(out.shape[0], self.dream_horizon, self.latent_dim)


def touch_dreaming_loss(
    pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """HTD Eq. 9: direction (cosine) + magnitude (smooth-L1) over future steps.

    Args:
        pred: ``[B, dream_horizon, latent_dim]`` predicted latents.
        target: ``[B, dream_horizon, latent_dim]`` EMA-teacher latents (detached).
        beta: weight on the magnitude term (guards against representation collapse).
    """
    pred = pred.float()
    target = target.float()
    direction = 1.0 - F.cosine_similarity(pred, target, dim=-1)  # [B, dream_horizon]
    magnitude = F.smooth_l1_loss(
        pred.norm(dim=-1), target.norm(dim=-1), reduction="none"
    )  # [B, dream_horizon]
    return (direction + beta * magnitude).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """In-place EMA: ``theta_ema <- decay * theta_ema + (1 - decay) * theta`` (HTD Eq. 4)."""
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    # Keep buffers (e.g. valid_idx) in sync.
    for ema_buf, buf in zip(teacher.buffers(), student.buffers()):
        ema_buf.copy_(buf)


def build_ema_teacher(student: TactileEncoder) -> TactileEncoder:
    """Clone ``student`` into a frozen EMA target encoder initialized from it."""
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    return teacher
