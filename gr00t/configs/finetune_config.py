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

# Finetune config used for single node post-training.
from dataclasses import dataclass
from typing import Any, Literal


NamedTactileMode = Literal["notactile", "htd", "jepa"]


@dataclass(frozen=True)
class TactileModeSettings:
    """Resolved switches for one tactile experiment mode."""

    use_tactile: Literal["dream", "input", "notac"]
    dream_state: bool
    dream_vision: bool


NAMED_TACTILE_MODES: dict[NamedTactileMode, TactileModeSettings] = {
    "notactile": TactileModeSettings("notac", False, False),
    "htd": TactileModeSettings("dream", False, False),
    "jepa": TactileModeSettings("dream", True, True),
}


def resolve_tactile_mode(
    tactile_mode: NamedTactileMode | None,
    *,
    use_tactile: Literal["dream", "input", "notac"],
    dream_state: bool,
    dream_vision: bool,
) -> TactileModeSettings:
    """Resolve a fixed named mode, falling back to the legacy independent switches."""

    if tactile_mode is not None:
        return NAMED_TACTILE_MODES[tactile_mode]
    return TactileModeSettings(use_tactile, dream_state, dream_vision)


def configure_tactile_data_windows(
    modality_config: dict[str, Any],
    settings: TactileModeSettings,
    *,
    dream_horizon: int,
    vision_horizon: int,
) -> None:
    """Load future observations only for auxiliary targets enabled by ``settings``."""

    if settings.use_tactile == "notac":
        modality_config.pop("tactile", None)
    else:
        if "tactile" not in modality_config:
            raise ValueError("The selected tactile mode requires a tactile modality")
        tactile_horizon = dream_horizon + 1 if settings.use_tactile == "dream" else 1
        modality_config["tactile"].delta_indices = list(range(tactile_horizon))

    dream_enabled = settings.use_tactile == "dream"
    modality_config["state"].delta_indices = (
        list(range(dream_horizon + 1)) if dream_enabled and settings.dream_state else [0]
    )
    modality_config["video"].delta_indices = (
        list(range(vision_horizon + 1)) if dream_enabled and settings.dream_vision else [0]
    )


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to one dataset root, or an os.pathsep-separated list of dataset roots."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    tactile_mode: NamedTactileMode | None = None
    """Fixed experiment mode for comparable runs:
      - "notactile": no tactile input and no auxiliary future targets.
      - "htd": current tactile input plus future-tactile dreaming only.
      - "jepa": HTD plus future-state and future-stereo-vision JEPA targets.

    When omitted, the legacy ``use_tactile``, ``dream_state``, and ``dream_vision``
    switches below remain fully supported. When set, this mode is authoritative."""

    use_tactile: Literal["dream", "input", "notac"] = "notac"
    """Tactile (skin-suit) mode. Requires the embodiment to declare a `tactile`
    modality (e.g. unitree_g1_sonic) for "dream"/"input".
      - "notac" (default): tactile disabled entirely (byte-for-byte the original
        N1.7). Default so ordinary non-tactile fine-tunes are unaffected.
      - "dream": full HTD graft -- the encoder injects tactile tokens into the
        action-head sequence AND a touch-dreaming auxiliary loss (EMA target encoder
        + dream head) is trained.
      - "input": ablation control group -- tactile is encoded and injected as a plain
        input only; no dream head / EMA teacher / auxiliary loss."""

    tune_tactile: bool = True
    """If True (and use_tactile != "notac"), train the tactile encoder / aggregator /
    dream head. Set False to freeze them (e.g. to first warm up other modules)."""

    tactile_encoder_type: Literal["mlp", "cnn", "coord"] = "mlp"
    """Per-region tactile encoder (only when use_tactile != "notac").
      - "mlp" (default): flat per-region MLP.
      - "cnn": per-region 2D conv over each region's (rows, cols) sensel grid ->
        adaptive pool -> MLP fuse.
      - "coord": "cnn" plus CoordConv row/col position channels, so the pooled CNN
        can encode where in a region a contact lands.
    Switching requires retraining (new params); "mlp" keeps prior runs unchanged."""

    dream_state: bool = False
    """State-JEPA branch (only meaningful with use_tactile="dream"). When True, the
    post-DiT tactile trunk additionally predicts the future *state* latent against an
    EMA(state_encoder) target, adding lambda_state * L_state to the loss. Requires the
    dataset state modality to load a future window (delta_indices >= dream_horizon+1;
    already set for unitree_g1_sonic) and state_history_length == 1. Training-only;
    inference and action output format are unchanged."""

    lambda_state: float = 0.5
    """Weight of the state-JEPA loss in the total loss (used only when dream_state)."""

    dream_vision: bool = False
    """Vision-JEPA branch (only meaningful with use_tactile="dream"). When True, the
    post-DiT tactile trunk additionally predicts the future *vision* latent against a
    frozen target: the backbone vision tower (`backbone.model.visual`) run over the
    future frames (no EMA -- it is already a fixed pretrained teacher). Adds
    lambda_vision * L_vision to the loss. Widens the video modality's delta_indices to
    range(vision_horizon+1) at launch (only for vision runs, so other runs pay no extra
    video IO). Training-only; inference and action output format are unchanged."""

    lambda_vision: float = 0.5
    """Weight of the vision-JEPA loss in the total loss (used only when dream_vision)."""

    vision_horizon: int = 4
    """Number of future frames the vision-JEPA branch predicts (target shape
    [B, vision_horizon, backbone_embedding_dim]). Each costs one frozen-ViT image
    encode and one extra decoded video frame per camera; lower it if dataloader IO or
    VRAM is tight."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    eval_steps: int | None = None
    """Validation interval. Defaults to save_steps for SONIC fine-tuning."""

    eval_set_split_ratio: float = 0.05
    """Fraction of whole episodes held out for validation."""

    eval_batch_size: int = 2
    """Per-device validation batch size."""

    eval_batches: int = 8
    """Maximum number of validation batches across the fixed held-out subset."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""
