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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import copy
import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import (
    FinetuneConfig,
    configure_tactile_data_windows,
    resolve_tactile_mode,
)
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # DataConfig defaults to the global modality registry. Copy it before narrowing
    # observation windows so one launch cannot mutate another config in-process.
    config.data.modality_configs = copy.deepcopy(config.data.modality_configs)

    tactile_settings = resolve_tactile_mode(
        ft_config.tactile_mode,
        use_tactile=ft_config.use_tactile,
        dream_state=ft_config.dream_state,
        dream_vision=ft_config.dream_vision,
        use_tactile_temporal=ft_config.use_tactile_temporal,
        use_delta_targets=ft_config.use_delta_targets,
    )
    if ft_config.tactile_mode is not None:
        print(f"[launch_finetune] Using fixed tactile mode: {ft_config.tactile_mode}")

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    # Translate the user-facing 3-way tactile enums into the model's internal bool
    # combination. "notac"->off, "input"->encoder-only (control), "dream"->full HTD.
    _tactile_mode = {"notac": (False, False), "input": (True, False), "dream": (True, True)}
    config.model.use_tactile, config.model.use_tactile_dream = _tactile_mode[
        tactile_settings.use_tactile
    ]
    config.model.tune_tactile = ft_config.tune_tactile
    config.model.use_tactile_temporal = tactile_settings.use_tactile_temporal
    config.model.tactile_history_length = ft_config.tactile_history_length
    config.model.use_delta_targets = tactile_settings.use_delta_targets
    config.model.predictor_tactile_source = ft_config.predictor_tactile_source
    config.model.tactile_token_chunk_targets = ft_config.tactile_token_chunk_targets
    if config.model.tactile_token_chunk_targets and tactile_settings.use_tactile != "dream":
        raise ValueError("tactile token chunk targets require use_tactile='dream'")
    if config.model.tactile_token_chunk_targets and tactile_settings.use_delta_targets:
        raise ValueError("token chunk ablation uses absolute targets, not delta targets")
    # State-JEPA branch rides on the touch-dreaming trunk, so it only applies in
    # "dream" mode. Silently ignore (with a warning) otherwise to avoid a no-op build.
    config.model.dream_state = (
        tactile_settings.dream_state and tactile_settings.use_tactile == "dream"
    )
    config.model.lambda_state = ft_config.lambda_state
    if tactile_settings.dream_state and tactile_settings.use_tactile != "dream":
        print(
            "[launch_finetune] WARNING: dream_state=True ignored because use_tactile != 'dream' "
            "(state-JEPA needs the touch-dreaming trunk)."
        )
    # Vision-JEPA also rides the post-DiT tactile trunk -> only in "dream" mode.
    config.model.dream_vision = (
        tactile_settings.dream_vision and tactile_settings.use_tactile == "dream"
    )
    if config.model.dream_vision and ft_config.tune_visual:
        raise ValueError("vision-JEPA requires tune_visual=False for a frozen vision teacher")
    config.model.lambda_vision = ft_config.lambda_vision
    config.model.vision_horizon = ft_config.vision_horizon
    if tactile_settings.dream_vision and tactile_settings.use_tactile != "dream":
        print(
            "[launch_finetune] WARNING: dream_vision=True ignored because use_tactile != 'dream' "
            "(vision-JEPA needs the touch-dreaming trunk)."
        )
    # Keep current observations on every policy path. Future windows are loaded only
    # for the enabled auxiliary teachers and are never used as action conditioning.
    configure_tactile_data_windows(
        config.data.modality_configs[embodiment_tag],
        tactile_settings,
        dream_horizon=config.model.dream_horizon,
        vision_horizon=config.model.vision_horizon,
        tactile_history_length=config.model.tactile_history_length,
    )
    if config.model.use_tactile_temporal:
        config.data.allow_padding = True
    # "coord" == cnn encoder with CoordConv channels enabled.
    _tactile_enc = {"mlp": ("mlp", False), "cnn": ("cnn", False), "coord": ("cnn", True)}
    config.model.tactile_encoder_type, config.model.tactile_cnn_coord = _tactile_enc[
        ft_config.tactile_encoder_type
    ]
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = os.environ.get("GR00T_BACKBONE_NAME", "nvidia/Cosmos-Reason2-2B")
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.eval_strategy = "steps"
    config.training.eval_steps = ft_config.eval_steps or ft_config.save_steps
    config.training.eval_set_split_ratio = ft_config.eval_set_split_ratio
    config.training.eval_batch_size = ft_config.eval_batch_size
    config.training.eval_batches = ft_config.eval_batches
    config.training.save_best_eval_metric_name = "eval_action_mse"
    config.training.save_best_eval_metric_greater_is_better = False
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
