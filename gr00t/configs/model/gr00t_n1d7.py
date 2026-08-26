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

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


def _default_tactile_valid_idx() -> list[int]:
    """Default tactile valid-channel map for unitree_g1_sonic (lazy to avoid import cycles)."""
    from gr00t.data.tactile_layout import get_valid_idx

    return get_valid_idx()


def _default_tactile_region_sizes() -> list[int]:
    """Default per-region channel counts for unitree_g1_sonic (lazy import)."""
    from gr00t.data.tactile_layout import get_region_sizes

    return get_region_sizes()


def _default_tactile_region_rows() -> list[int]:
    """Default per-region row counts for unitree_g1_sonic (lazy import).

    Stored flat (not as (rows, cols) tuples) because OmegaConf -- used by
    ``experiment.run`` -- rejects nested-tuple-typed dataclass fields.
    """
    from gr00t.data.tactile_layout import get_region_grids

    return [rows for rows, _ in get_region_grids()]


def _default_tactile_region_cols() -> list[int]:
    """Default per-region column counts for unitree_g1_sonic (lazy import)."""
    from gr00t.data.tactile_layout import get_region_grids

    return [cols for _, cols in get_region_grids()]


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # Tactile (skin-suit) modality + touch-dreaming (HTD graft, arXiv:2604.13015).
    # `use_tactile` gates the entire feature: when False the model is byte-for-byte
    # the original N1.7. The raw on-disk packet is `tactile_raw_dim` wide; the
    # encoder selects `tactile_valid_idx` (768->624 for unitree_g1_sonic, defaulted
    # from gr00t/data/tactile_layout.py) and splits them by `tactile_region_sizes`.
    # If `tactile_valid_idx` is None, the encoder falls back to using all
    # `tactile_raw_dim` channels as a single region (lets training start before a
    # spec mapping is available).
    use_tactile: bool = False
    tactile_raw_dim: int = 768
    tactile_valid_idx: list[int] | None = field(default_factory=_default_tactile_valid_idx)
    tactile_region_sizes: list[int] | None = field(default_factory=_default_tactile_region_sizes)
    # Optional dataset-derived preprocessing. Defaults reproduce the legacy
    # valid-select + /255 path exactly. Values are expanded per region inside
    # TactileEncoder and serialized with the checkpoint for deployment parity.
    tactile_deadband: float = 0.0
    tactile_region_scales: list[float] | None = None
    tactile_region_mask: list[float] | None = None
    # Optional bounded learned multiplier on tokens entering the action trunk.
    # None preserves the legacy ungated path; otherwise the sigmoid gate starts
    # at this value and remains in (0, 1).
    tactile_input_gate_init: float | None = None
    # Per-region encoder: "mlp" (default; flat per-region MLP) or "cnn" (per-region
    # 2D conv over each region's (rows, cols) grid -> adaptive pool -> MLP fuse).
    tactile_encoder_type: str = "mlp"
    # Per-region (rows, cols) grid as two flat int lists (used only when
    # tactile_encoder_type == "cnn"). Flat rather than (rows, cols) tuples because
    # OmegaConf.create in experiment.run rejects nested-tuple-typed fields.
    tactile_region_rows: list[int] | None = field(default_factory=_default_tactile_region_rows)
    tactile_region_cols: list[int] | None = field(default_factory=_default_tactile_region_cols)
    tactile_cnn_channels: int = 32  # conv channels per region (cnn encoder only)
    tactile_cnn_pool: list[int] = field(
        default_factory=lambda: [2, 2]
    )  # adaptive-pool (h, w) spatial size (cnn encoder only)
    # CoordConv: add normalized row/col position channels so the pooled CNN can
    # encode *where* in a region a contact lands (cnn encoder only).
    tactile_cnn_coord: bool = False
    # Scale on the [-1, 1] CoordConv channels. At 1.0 they dominate the ~0.04-std
    # value channel and ill-condition the conv (grad spikes, loss stalls). Measured
    # value std gives a principled range ~0.06-0.13; 0.1 is the default. No CLI flag
    # by design -- edit here to sweep.
    tactile_cnn_coord_scale: float = 0.1
    n_tactile_tokens: int = 8  # number of slot tokens injected into sa_embs
    tactile_hidden_dim: int = 512  # per-region / dream-head MLP hidden width
    use_tactile_temporal: bool = False
    tactile_history_length: int = 4
    tactile_temporal_layers: int = 1
    tactile_temporal_heads: int = 8
    dream_horizon: int = 4  # tau: future tactile frames predicted (must be <= delta_indices reach)
    ema_decay: float = 0.99  # EMA target-encoder decay (HTD Eq. 4)
    lambda_tactile: float = 0.5  # weight of touch-dreaming loss in total loss
    tactile_dream_beta: float = 1.0  # magnitude-term weight in touch-dreaming loss (anti-collapse)
    tune_tactile: bool = True  # train the tactile encoder / aggregator / dream head
    # Touch-dreaming master switch (only matters when use_tactile). True (default)
    # = full HTD graft (EMA teacher + dream head + L_tact aux loss). False = ablation
    # control: tactile is encoded and injected into sa_embs as a plain input only,
    # no dream head / EMA teacher / auxiliary loss is built or run.
    use_tactile_dream: bool = True
    use_delta_targets: bool = False
    # Predictor context: temporal tactile tokens before DiT, tactile positions
    # after DiT, all current observation modalities before DiT, or complete trunk.
    predictor_tactile_source: str = "post_dit"
    tactile_token_chunk_targets: bool = False
    # --- Tactile JEPA: predict future *other-modality* latents from the selected
    # pre- or post-DiT tactile context (shares the HTD dream machinery: EMA target encoder + a
    # TactileDreamHead-shaped predictor + touch_dreaming_loss). All branches are
    # training-only and gated on use_tactile + use_tactile_dream. Targets are absolute by default; the full
    # JEPA mode predicts future-minus-current latents for all three branches.
    # State branch: EMA(state_encoder) encodes the future state window into the
    # target; requires the dataset's state modality to load a future window
    # (delta_indices >= dream_horizon+1) and state_history_length == 1.
    dream_state: bool = False
    lambda_state: float = 0.5  # weight of the state-JEPA loss in total loss
    # Vision branch: predict the future *vision* latent from the same selected
    # tactile trunk. Unlike state/tactile there is no clean+trained encoder to EMA,
    # so the target is the FROZEN backbone vision tower (`backbone.model.visual`)
    # run once over the future frames -- a fixed, pretrained teacher (no EMA, no
    # collapse source). The target dim is backbone_embedding_dim (the vision
    # merger projects patches into the LLM hidden size). Requires the dataset's
    # video modality to load a future window (delta_indices >= vision_horizon+1,
    # widened conditionally at launch so non-vision runs pay no extra video IO).
    # Training-only; inference / action output are unchanged.
    dream_vision: bool = False
    lambda_vision: float = 0.5  # weight of the vision-JEPA loss in total loss
    vision_horizon: int = (
        4  # future frames predicted (target shape [B, vision_horizon, backbone_embedding_dim])
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
