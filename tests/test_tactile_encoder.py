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

"""CPU-only shape/behavior tests for the tactile encoder + touch-dreaming heads."""

from gr00t.data.tactile_layout import get_region_grids, get_region_sizes, get_valid_idx
from gr00t.model.modules.tactile_encoder import (
    TactileDreamHead,
    TactileEncoder,
    TactileTemporalEncoder,
    build_ema_teacher,
    ema_update,
    latent_prediction_target,
    touch_dreaming_loss,
)
import torch


EMBED = 1536
N_TOKENS = 8
HIDDEN = 256  # small for a fast CPU test
RAW = 768
TAU = 4
TRUNK = 1024


def _make_encoder(valid_idx, region_sizes):
    return TactileEncoder(
        raw_dim=RAW,
        valid_idx=valid_idx,
        region_sizes=region_sizes,
        embed_dim=EMBED,
        num_tokens=N_TOKENS,
        hidden_dim=HIDDEN,
    )


def test_encoder_eight_region_shapes():
    enc = _make_encoder(get_valid_idx(), get_region_sizes())
    raw = torch.randint(0, 256, (3, RAW)).float()
    tokens = enc(raw)
    assert tokens.shape == (3, N_TOKENS, EMBED)
    # pooled latents over current + future frames
    raw_seq = torch.randint(0, 256, (3, TAU, RAW)).float()
    pooled = enc.encode_pooled(raw_seq)
    assert pooled.shape == (3, TAU, EMBED)
    pooled1 = enc.encode_pooled(raw)
    assert pooled1.shape == (3, EMBED)


def _make_cnn_encoder(valid_idx, region_sizes, region_grids):
    return TactileEncoder(
        raw_dim=RAW,
        valid_idx=valid_idx,
        region_sizes=region_sizes,
        embed_dim=EMBED,
        num_tokens=N_TOKENS,
        hidden_dim=HIDDEN,
        encoder_type="cnn",
        region_grids=region_grids,
        cnn_channels=8,  # small for a fast CPU test
        cnn_pool=(2, 2),
    )


def test_cnn_encoder_eight_region_shapes():
    # get_region_grids() includes the degenerate 1x4 shoulder strips, exercising
    # the pool-clamp path (pool (2,2) -> (1,2) for a 1-row region).
    enc = _make_cnn_encoder(get_valid_idx(), get_region_sizes(), get_region_grids())
    raw = torch.randint(0, 256, (3, RAW)).float()
    assert enc(raw).shape == (3, N_TOKENS, EMBED)
    raw_seq = torch.randint(0, 256, (3, TAU, RAW)).float()
    assert enc.encode_pooled(raw_seq).shape == (3, TAU, EMBED)


def test_cnn_encoder_backward_flows():
    enc = _make_cnn_encoder(get_valid_idx(), get_region_sizes(), get_region_grids())
    raw = torch.randint(0, 256, (2, RAW)).float()
    enc(raw).sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_cnn_encoder_coordconv_shapes_and_backward():
    # CoordConv adds 2 input channels; first conv must accept 3 in-channels and the
    # 1x4 shoulder strips (row coord -> 0) must not break.
    enc = TactileEncoder(
        raw_dim=RAW,
        valid_idx=get_valid_idx(),
        region_sizes=get_region_sizes(),
        embed_dim=EMBED,
        num_tokens=N_TOKENS,
        hidden_dim=HIDDEN,
        encoder_type="cnn",
        region_grids=get_region_grids(),
        cnn_channels=8,
        cnn_pool=(2, 2),
        cnn_coord=True,
    )
    assert enc.per_region.branches[0][0].in_channels == 3
    raw = torch.randint(0, 256, (2, RAW)).float()
    out = enc(raw)
    assert out.shape == (2, N_TOKENS, EMBED)
    out.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_cnn_encoder_rejects_grid_size_mismatch():
    import pytest

    with pytest.raises(AssertionError):
        _make_cnn_encoder(get_valid_idx(), get_region_sizes(), [(6, 8)])  # wrong #regions


def test_encoder_fallback_single_region():
    # No spec mapping: use all raw_dim channels as one region.
    enc = _make_encoder(list(range(RAW)), [RAW])
    raw = torch.randint(0, 256, (2, RAW)).float()
    assert enc(raw).shape == (2, N_TOKENS, EMBED)
    # normalization clamps to [0, 1]
    norm = enc.select_and_normalize(raw)
    assert norm.shape == (2, RAW)
    assert float(norm.min()) >= 0.0 and float(norm.max()) <= 1.0


def test_dream_head_and_loss():
    head = TactileDreamHead(in_dim=TRUNK, latent_dim=EMBED, dream_horizon=TAU, hidden_dim=HIDDEN)
    trunk = torch.randn(5, TRUNK)
    pred = head(trunk)
    assert pred.shape == (5, TAU, EMBED)

    target = torch.randn(5, TAU, EMBED)
    loss = touch_dreaming_loss(pred, target, beta=1.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # identical pred/target -> direction term 0, magnitude term 0
    zero = touch_dreaming_loss(target, target, beta=1.0)
    assert float(zero) < 1e-5
    zero_delta = torch.zeros_like(target)
    assert float(touch_dreaming_loss(zero_delta, zero_delta, beta=1.0)) == 0.0


def test_temporal_encoder_fuses_history_and_backpropagates():
    encoder = TactileTemporalEncoder(
        embed_dim=32,
        hidden_dim=16,
        history_length=4,
        num_layers=1,
        num_heads=4,
    )
    tokens = torch.randn(2, 4, 3, 32, requires_grad=True)
    output = encoder(tokens)
    assert output.shape == (2, 3, 32)
    output.sum().backward()
    assert tokens.grad is not None
    assert float(tokens.grad[:, :-1].abs().sum()) > 0


def test_latent_prediction_target_subtracts_current_latent():
    future = torch.tensor([[[2.0, 5.0], [4.0, 8.0]]])
    current = torch.tensor([[1.0, 3.0]])

    assert latent_prediction_target(future, current, False) is future
    assert torch.equal(
        latent_prediction_target(future, current, True),
        torch.tensor([[[1.0, 2.0], [3.0, 5.0]]]),
    )


def test_ema_teacher_tracks_student():
    student = _make_encoder(get_valid_idx(), get_region_sizes())
    teacher = build_ema_teacher(student)
    # teacher starts equal to student and is frozen
    assert all(not p.requires_grad for p in teacher.parameters())
    p_s = next(student.parameters())
    p_t = next(teacher.parameters())
    assert torch.allclose(p_s, p_t)

    # perturb student, EMA-update teacher -> teacher moves a little toward student
    with torch.no_grad():
        p_s.add_(1.0)
    before = p_t.clone()
    ema_update(teacher, student, decay=0.99)
    after = next(teacher.parameters())
    assert not torch.allclose(after, before)  # moved
    # moved by ~ (1 - decay) of the gap
    assert torch.allclose(after, before + 0.01 * (p_s - before), atol=1e-5)


def test_encoder_backward_flows():
    enc = _make_encoder(get_valid_idx(), get_region_sizes())
    raw = torch.randint(0, 256, (2, RAW)).float()
    tokens = enc(raw)
    tokens.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(float(g.abs().sum()) > 0 for g in grads)
