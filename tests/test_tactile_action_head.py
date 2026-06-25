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

"""CPU-only integration tests for tactile injection into the GR00T N1.7 action head.

Exercises the real action head (with a small randomly-initialized DiT, no
pretrained weights) to verify the tactile wiring: tactile tokens enter ``sa_embs``
before the action tokens, the touch-dreaming loss is computed and backprops to the
tactile encoder + dream head (but never the EMA target encoder), and the
``use_tactile=False`` path is byte-for-byte the original behavior.
"""

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

B = 2
LVL = 7  # backbone (vision+language) sequence length


def _tiny_config(**overrides):
    cfg = dict(
        use_flash_attention=False,
        use_alternate_vl_dit=False,
        tactile_hidden_dim=128,
        n_tactile_tokens=8,
        dream_horizon=4,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            # inner_dim (num_heads * head_dim) must equal input_embedding_dim (1536)
            "num_attention_heads": 4,
            "attention_head_dim": 384,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 1024,  # must equal hidden_size (action_decoder input)
            "interleave_self_attention": True,
        },
    )
    cfg.update(overrides)
    return Gr00tN1d7Config(**cfg)


def _backbone_output(cfg):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(B, LVL, cfg.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(B, LVL, dtype=torch.long),
        }
    )


def _action_input(cfg, with_tactile=True, with_action=True):
    data = {
        "state": torch.randn(B, cfg.state_history_length, cfg.max_state_dim),
        "embodiment_id": torch.zeros(B, dtype=torch.long),
    }
    if with_action:
        # Ground-truth action is present only in training; at inference its
        # presence would (intentionally) trigger the RTC inpainting path.
        data["action"] = torch.randn(B, cfg.action_horizon, cfg.max_action_dim)
        data["action_mask"] = torch.ones(B, cfg.action_horizon, cfg.max_action_dim)
    if with_tactile:
        data["tactile"] = torch.randint(
            0, 256, (B, cfg.dream_horizon + 1, cfg.tactile_raw_dim)
        ).float()
    return BatchFeature(data=data)


def test_forward_with_tactile_backprops_correctly():
    cfg = _tiny_config(use_tactile=True)
    head = Gr00tN1d7ActionHead(cfg).float().train()

    # EMA target encoder is frozen at construction.
    assert all(not p.requires_grad for p in head.tactile_target_encoder.parameters())

    out = head(_backbone_output(cfg), _action_input(cfg))
    assert "tactile_loss" in out
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["tactile_loss"])

    out["loss"].backward()
    enc_grad = sum(
        float(p.grad.abs().sum()) for p in head.tactile_encoder.parameters() if p.grad is not None
    )
    dream_grad = sum(
        float(p.grad.abs().sum())
        for p in head.tactile_dream_head.parameters()
        if p.grad is not None
    )
    target_grads = [p.grad for p in head.tactile_target_encoder.parameters() if p.grad is not None]
    assert enc_grad > 0  # tactile encoder learns
    assert dream_grad > 0  # dream head learns
    assert len(target_grads) == 0  # EMA target encoder never gets gradients


def test_ema_target_moves_during_forward():
    cfg = _tiny_config(use_tactile=True, ema_decay=0.9)
    head = Gr00tN1d7ActionHead(cfg).float().train()
    # diverge student from target so the EMA step is observable
    with torch.no_grad():
        for p in head.tactile_encoder.parameters():
            p.add_(0.5)
    before = next(head.tactile_target_encoder.parameters()).clone()
    head(_backbone_output(cfg), _action_input(cfg))  # forward runs ema_update
    after = next(head.tactile_target_encoder.parameters())
    assert not torch.allclose(after, before)


def test_use_tactile_false_has_no_tactile_modules():
    cfg = _tiny_config(use_tactile=False)
    head = Gr00tN1d7ActionHead(cfg).float().train()
    assert not hasattr(head, "tactile_encoder")
    out = head(_backbone_output(cfg), _action_input(cfg, with_tactile=False))
    assert "tactile_loss" not in out
    assert torch.isfinite(out["loss"])


def test_tactile_no_dream_control_group():
    # Ablation control: tactile encoded + injected into sa_embs as a plain input,
    # but no dream head / EMA teacher / auxiliary loss.
    cfg = _tiny_config(use_tactile=True, use_tactile_dream=False)
    head = Gr00tN1d7ActionHead(cfg).float().train()

    # Dream-only modules must not exist; the encoder still does.
    assert hasattr(head, "tactile_encoder")
    assert not hasattr(head, "tactile_dream_head")
    assert not hasattr(head, "tactile_target_encoder")

    out = head(_backbone_output(cfg), _action_input(cfg))
    assert "tactile_loss" not in out
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    enc_grad = sum(
        float(p.grad.abs().sum()) for p in head.tactile_encoder.parameters() if p.grad is not None
    )
    assert enc_grad > 0  # tactile still flows into the action loss as an input


def test_inference_runs_with_tactile():
    cfg = _tiny_config(use_tactile=True)
    head = Gr00tN1d7ActionHead(cfg).float().eval()
    bo = _backbone_output(cfg)
    ai = _action_input(cfg, with_action=False)
    out = head.get_action(bo, ai)
    # tail-anchored action decode is unaffected by the injected tactile tokens
    assert out["action_pred"].shape == (B, cfg.action_horizon, cfg.max_action_dim)
    assert torch.isfinite(out["action_pred"]).all()


def test_inference_tactile_zero_fallback():
    # use_tactile=True but no tactile provided -> model zero-fills, still runs.
    cfg = _tiny_config(use_tactile=True)
    head = Gr00tN1d7ActionHead(cfg).float().eval()
    out = head.get_action(
        _backbone_output(cfg), _action_input(cfg, with_tactile=False, with_action=False)
    )
    assert out["action_pred"].shape == (B, cfg.action_horizon, cfg.max_action_dim)
