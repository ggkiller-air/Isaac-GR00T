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

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.tactile_encoder import latent_prediction_target
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])

    @pytest.mark.parametrize("predictor_tactile_source", ["pre_dit", "post_dit"])
    def test_full_tactile_jepa_forward_and_backward(
        self, monkeypatch, predictor_tactile_source
    ):
        config = _small_config(
            input_embedding_dim=96,
            hidden_size=64,
            use_tactile=True,
            use_tactile_dream=True,
            use_tactile_temporal=True,
            tactile_history_length=4,
            tactile_hidden_dim=32,
            tactile_temporal_heads=4,
            dream_horizon=4,
            dream_state=True,
            dream_vision=True,
            vision_horizon=4,
            use_delta_targets=True,
            predictor_tactile_source=predictor_tactile_source,
            diffusion_model_cfg={
                "positional_embeddings": None,
                "num_layers": 2,
                "num_attention_heads": 2,
                "attention_head_dim": 48,
                "norm_type": "ada_norm",
                "dropout": 0.0,
                "final_dropout": False,
                "output_dim": 64,
                "interleave_self_attention": True,
            },
        )
        head = Gr00tN1d7ActionHead(config).float().train()
        captured = {}
        delta_calls = []

        def capture_temporal_output(_module, _inputs, output):
            captured["pre_dit"] = output.detach()

        def capture_dit_output(_module, _inputs, output):
            captured["post_dit"] = output[0].detach()

        def capture_predictor_input(name):
            def hook(_module, inputs):
                captured[name] = inputs[0].detach()

            return hook

        def record_delta_target(future, current, use_delta):
            target = latent_prediction_target(future, current, use_delta)
            current_for_subtraction = current
            if current.dim() == future.dim() - 1:
                current_for_subtraction = current.unsqueeze(1)
            assert use_delta is True
            torch.testing.assert_close(target, future - current_for_subtraction)
            delta_calls.append(target.detach())
            return target

        monkeypatch.setattr(
            "gr00t.model.gr00t_n1d7.gr00t_n1d7.latent_prediction_target",
            record_delta_target,
        )
        head.tactile_temporal_encoder.register_forward_hook(capture_temporal_output)
        head.model.register_forward_hook(capture_dit_output)
        head.tactile_dream_head.register_forward_pre_hook(
            capture_predictor_input("tactile_context")
        )
        head.state_dream_head.register_forward_pre_hook(
            capture_predictor_input("state_context")
        )
        head.vision_dream_head.register_forward_pre_hook(
            capture_predictor_input("vision_context")
        )
        action_input = _make_action_input(config)
        action_input["state"] = torch.randn(
            2,
            config.state_history_length + config.dream_horizon,
            config.max_state_dim,
        )
        action_input["tactile"] = torch.randint(
            0,
            256,
            (2, config.tactile_history_length + config.dream_horizon, config.tactile_raw_dim),
        ).float()
        action_input["vision_current"] = torch.randn(2, config.backbone_embedding_dim)
        action_input["vision_target"] = torch.randn(
            2, config.vision_horizon, config.backbone_embedding_dim
        )

        out = head(_make_backbone_output(config), action_input)

        assert {"tactile_loss", "state_jepa_loss", "vision_jepa_loss"} <= set(out)
        assert len(delta_calls) == 3
        if predictor_tactile_source == "pre_dit":
            expected_context = head.model.proj_out_2(captured["pre_dit"]).mean(dim=1)
        else:
            expected_context = captured["post_dit"][:, 1 : 1 + head.n_tactile_tokens].mean(
                dim=1
            )
        torch.testing.assert_close(captured["tactile_context"], expected_context)
        torch.testing.assert_close(captured["state_context"], expected_context)
        torch.testing.assert_close(captured["vision_context"], expected_context)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in head.tactile_temporal_encoder.parameters()
        )
        assert all(parameter.grad is None for parameter in head.tactile_target_encoder.parameters())
        assert all(parameter.grad is None for parameter in head.state_target_encoder.parameters())

    def test_predictor_source_ablation_does_not_change_modules(self):
        common = dict(
            use_tactile=True,
            use_tactile_dream=True,
            use_tactile_temporal=True,
            tactile_history_length=4,
            tactile_hidden_dim=32,
            tactile_temporal_heads=4,
            dream_horizon=4,
            dream_state=True,
            dream_vision=True,
            vision_horizon=4,
            use_delta_targets=True,
        )
        pre_dit = Gr00tN1d7ActionHead(
            _small_config(**common, predictor_tactile_source="pre_dit")
        )
        post_dit = Gr00tN1d7ActionHead(
            _small_config(**common, predictor_tactile_source="post_dit")
        )

        pre_shapes = {name: parameter.shape for name, parameter in pre_dit.named_parameters()}
        post_shapes = {name: parameter.shape for name, parameter in post_dit.named_parameters()}
        assert pre_shapes == post_shapes
        assert sum(parameter.numel() for parameter in pre_dit.parameters()) == sum(
            parameter.numel() for parameter in post_dit.parameters()
        )


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad
