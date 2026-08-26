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
    ``sa_embs`` sequence). It owns the 768->624 valid-channel select and the
    configurable deadband/region scaling, so the data pipeline only forwards
    the raw packet. The default remains the legacy ``/255`` normalization.
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
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from gr00t.model.modules.embodiment_conditioned_mlp import SmallMLP


class PerRegionTactileEncoder(nn.Module):
    """Encode each body region independently, then stack into region tokens.

    Input is the flat vector of valid channels (region-ordered); it is split by
    ``region_sizes`` and each slice goes through its own 2-layer MLP projecting
    to ``output_dim``. Output: ``[B, num_regions, output_dim]``. No 2D spatial
    layout is assumed (MLP, not CNN), matching the per-region sensel counts
    ``[48, 40, 8, 4, 8, 4, 256, 256]``.
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


class PerRegionTactileCNNEncoder(nn.Module):
    """Encode each region with a small 2D CNN over its ``(rows, cols)`` grid.

    Mirrors :class:`PerRegionTactileEncoder`'s interface (flat valid vector in,
    ``[B, num_regions, output_dim]`` out) but exploits the per-region spatial
    layout from ``tactile_layout``: each region slice is reshaped to
    ``[B, 1, rows, cols]`` (row-major), passed through a single ``3x3`` conv,
    adaptively pooled to a fixed ``(pool_h, pool_w)`` resolution, flattened, and
    fused by a linear projection to ``output_dim`` -- the recipe described for
    per-finger/region tactile embeddings. ``padding=1`` 3x3 conv handles any grid
    ``>= 1x1`` (including the degenerate ``1xN`` shoulder strips); the pool size
    is clamped to each region's grid so a ``1xN`` strip is never up-sampled.

    When ``coord`` is set, two constant CoordConv (Liu et al. 2018) channels --
    normalized row/col position in ``[-1, 1]`` -- are concatenated to the value
    grid before the conv, so the otherwise translation-equivariant + pooled CNN
    can encode *where* in the region a contact lands (a 1-row strip gets a 0
    row-coordinate). The coords are recomputed per forward to match the input's
    device/dtype; the cost is negligible for these tiny grids.
    """

    def __init__(
        self,
        region_grids: list[tuple[int, int]],
        output_dim: int,
        channels: int = 32,
        pool: tuple[int, int] = (2, 2),
        coord: bool = False,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        self.region_grids = [(int(r), int(c)) for r, c in region_grids]
        self.coord = coord
        # CoordConv channels span [-1, 1] (std ~0.7), but the value channel after
        # /255 has std ~0.04 (mostly zeros). Left at 1.0 the coords *dominate* and
        # *ill-condition* the per-region conv (grad spikes, loss stalls). Scaling
        # them to value range (~0.06-0.13 matches measured std) fixes this.
        self.coord_scale = float(coord_scale)
        in_channels = 1 + (2 if coord else 0)  # value (+ row/col coord if CoordConv)
        self.branches = nn.ModuleList()
        self.projections = nn.ModuleList()
        for rows, cols in self.region_grids:
            ph, pw = min(int(pool[0]), rows), min(int(pool[1]), cols)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((ph, pw)),
                )
            )
            # Per-region projection: pooled flat width (channels*ph*pw) differs
            # per region because ph/pw are clamped to the grid.
            self.projections.append(nn.Linear(channels * ph * pw, output_dim))

    @staticmethod
    def _axis_coord(n: int, device, dtype) -> torch.Tensor:
        # Normalized axis position in [-1, 1]; a single cell maps to 0 (center).
        if n == 1:
            return torch.zeros(1, device=device, dtype=dtype)
        return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)

    def forward(self, x_valid: torch.Tensor) -> torch.Tensor:
        # x_valid: [B, sum(rows*cols)] -> [B, num_regions, output_dim]
        sizes = [rows * cols for rows, cols in self.region_grids]
        parts = torch.split(x_valid, sizes, dim=-1)
        tokens = []
        for (rows, cols), conv, proj, part in zip(
            self.region_grids, self.branches, self.projections, parts
        ):
            batch = part.shape[0]
            grid = part.reshape(batch, 1, rows, cols)
            if self.coord:
                row_c = self._axis_coord(rows, grid.device, grid.dtype) * self.coord_scale
                col_c = self._axis_coord(cols, grid.device, grid.dtype) * self.coord_scale
                ch = row_c.view(1, 1, rows, 1).expand(batch, 1, rows, cols)
                cw = col_c.view(1, 1, 1, cols).expand(batch, 1, rows, cols)
                grid = torch.cat([grid, ch, cw], dim=1)
            feat = conv(grid).flatten(1)
            tokens.append(proj(feat))
        return torch.stack(tokens, dim=1)


class TactileSlotAggregator(nn.Module):
    """Aggregate variable region tokens into a fixed set of ``N`` slot tokens.

    ``N`` learnable query vectors cross-attend the region tokens, decoupling the
    output token count from the number of regions / raw channels.
    """

    def __init__(self, embed_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, region_tokens: torch.Tensor) -> torch.Tensor:
        # region_tokens: [B, num_regions, embed_dim] -> [B, num_tokens, embed_dim]
        batch = region_tokens.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1).to(region_tokens.dtype)
        # Force the math SDPA backend: head_dim=192 (embed_dim/num_heads) hits a NaN
        # bug in the flash-attention *backward* kernel under bf16. This attention is
        # tiny (num_tokens queries x num_regions keys), so the cost is negligible.
        with sdpa_kernel([SDPBackend.MATH]):
            attended, _ = self.attn(query, region_tokens, region_tokens, need_weights=False)
        return self.norm(attended)


class TactileEncoder(nn.Module):
    """Raw skin packet -> tactile tokens, owning selection and normalization.

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
        encoder_type: str = "mlp",
        region_grids: list[tuple[int, int]] | None = None,
        cnn_channels: int = 32,
        cnn_pool: tuple[int, int] = (2, 2),
        cnn_coord: bool = False,
        cnn_coord_scale: float = 1.0,
        deadband: float = 0.0,
        region_scales: list[float] | None = None,
        region_mask: list[float] | None = None,
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
        num_regions = len(region_sizes)
        scales = [255.0] * num_regions if region_scales is None else list(region_scales)
        mask = [1.0] * num_regions if region_mask is None else list(region_mask)
        if len(scales) != num_regions:
            raise ValueError(f"region_scales has {len(scales)} entries, expected {num_regions}")
        if len(mask) != num_regions:
            raise ValueError(f"region_mask has {len(mask)} entries, expected {num_regions}")
        if float(deadband) < 0:
            raise ValueError("tactile deadband must be non-negative")
        if any(float(scale) <= 0 for scale in scales):
            raise ValueError("all tactile region scales must be positive")
        if any(float(value) < 0 or float(value) > 1 for value in mask):
            raise ValueError("all tactile region mask values must be in [0, 1]")
        channel_scales = torch.repeat_interleave(
            torch.as_tensor(scales, dtype=torch.float32),
            torch.as_tensor(region_sizes, dtype=torch.long),
        )
        channel_mask = torch.repeat_interleave(
            torch.as_tensor(mask, dtype=torch.float32),
            torch.as_tensor(region_sizes, dtype=torch.long),
        )
        self.deadband = float(deadband)
        self.register_buffer("channel_scales", channel_scales, persistent=False)
        self.register_buffer("channel_mask", channel_mask, persistent=False)
        # Per-region encoder: flat MLP (default) or 2D CNN over each region's grid.
        # Both emit [B, num_regions, embed_dim], so the aggregator / dream path are
        # identical regardless of choice.
        if encoder_type == "cnn":
            assert region_grids is not None, "encoder_type='cnn' requires region_grids"
            assert [int(r) * int(c) for r, c in region_grids] == list(region_sizes), (
                f"region_grids {region_grids} inconsistent with region_sizes {region_sizes}"
            )
            self.per_region = PerRegionTactileCNNEncoder(
                region_grids,
                embed_dim,
                channels=cnn_channels,
                pool=cnn_pool,
                coord=cnn_coord,
                coord_scale=cnn_coord_scale,
            )
        elif encoder_type == "mlp":
            self.per_region = PerRegionTactileEncoder(region_sizes, hidden_dim, embed_dim)
        else:
            raise ValueError(f"unknown tactile encoder_type: {encoder_type!r}")
        self.aggregator = TactileSlotAggregator(embed_dim, num_tokens, num_heads)

    def select_and_normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """``[..., raw_dim]`` (0-255) -> ``[..., num_valid]`` in ``[0, 1]``."""
        dtype = self.aggregator.norm.weight.dtype
        valid = raw.index_select(-1, self.valid_idx).to(dtype)
        scales = self.channel_scales.to(device=valid.device, dtype=dtype)
        mask = self.channel_mask.to(device=valid.device, dtype=dtype)
        return ((valid - self.deadband).clamp_min(0) / scales).clamp_max(1) * mask

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


class TactileTemporalEncoder(nn.Module):
    """Fuse causal per-frame tactile slot tokens through a bottleneck transformer."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        history_length: int,
        num_layers: int = 1,
        num_heads: int = 8,
    ):
        super().__init__()
        if history_length < 2:
            raise ValueError("tactile_history_length must be at least 2")
        if num_layers < 1:
            raise ValueError("tactile_temporal_layers must be at least 1")
        if num_heads < 1:
            raise ValueError("tactile_temporal_heads must be at least 1")
        if hidden_dim % num_heads != 0:
            raise ValueError("tactile temporal hidden_dim must be divisible by num_heads")
        self.history_length = history_length
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        self.time_embedding = nn.Parameter(torch.empty(history_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError(f"expected tactile tokens [B,H,N,D], got {tuple(tokens.shape)}")
        batch, history, slots, width = tokens.shape
        if history != self.history_length:
            raise ValueError(f"expected tactile history {self.history_length}, got {history}")
        hidden = self.input_projection(tokens).permute(0, 2, 1, 3)
        hidden = hidden.reshape(batch * slots, history, -1)
        hidden = hidden + self.time_embedding[None].to(hidden.dtype)
        hidden = self.transformer(hidden)[:, -1]
        update = self.output_projection(hidden).reshape(batch, slots, width)
        return self.output_norm(tokens[:, -1] + update)


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
    target_has_direction = target.norm(dim=-1) > 1e-6
    direction = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    direction = direction * target_has_direction.to(direction.dtype)
    magnitude = F.smooth_l1_loss(
        pred.norm(dim=-1), target.norm(dim=-1), reduction="none"
    )  # [B, dream_horizon]
    return (direction + beta * magnitude).mean()


def latent_prediction_target(
    future: torch.Tensor, current: torch.Tensor, use_delta: bool
) -> torch.Tensor:
    """Return absolute future latents or ``future - current`` in one teacher space."""
    if not use_delta:
        return future
    if current.dim() == future.dim() - 1:
        current = current.unsqueeze(1)
    return future - current


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """In-place EMA: ``theta_ema <- decay * theta_ema + (1 - decay) * theta`` (HTD Eq. 4)."""
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    # Keep buffers (e.g. valid_idx) in sync.
    for ema_buf, buf in zip(teacher.buffers(), student.buffers()):
        ema_buf.copy_(buf)


def build_ema_teacher(student: nn.Module) -> nn.Module:
    """Clone ``student`` into a frozen EMA target encoder initialized from it.

    Works for any ``nn.Module`` (tactile encoder for touch-dreaming, state encoder
    for the state-JEPA branch); the EMA is driven externally by ``ema_update``.
    """
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    return teacher
