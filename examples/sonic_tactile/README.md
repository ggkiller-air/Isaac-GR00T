# Tactile (skin-suit) finetuning for `unitree_g1_sonic`

Grafts a tactile modality + touch-dreaming (HTD, arXiv:2604.13015) onto the
GR00T N1.7 action head, **without** replacing the pretrained VLM + DiT. Gated by
`use_tactile`, so the base model is unchanged when it's off.

## Quick start

```bash
export HF_TOKEN=...    # base model nvidia/GR00T-N1.7-3B (~6GB, not gated) auto-downloads
# full run (watch loss in your own tmux):
CUDA_VISIBLE_DEVICES=1 bash examples/sonic_tactile/finetune_sonic_tactile.sh

# quick smoke (does it start end-to-end?):
CUDA_VISIBLE_DEVICES=1 MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=2 \
  OUTPUT_DIR=/tmp/sonic_tactile_smoke bash examples/sonic_tactile/finetune_sonic_tactile.sh
```

W&B logs `tactile_loss` alongside `loss`. Touch-dreaming is what makes tactile
*meaningful* (per the paper, tactile-as-plain-input alone is unreliable); watch
that `tactile_loss` decreases and does not collapse (the magnitude term guards it).

## ⚠️ One decision you must confirm: which camera(s)

`carry-bucket-stereo` ships two camera streams (`ego_view_left` / `ego_view_right`)
but the pretrained SONIC backbone is **mono**. `modality_config.py` therefore
defaults to a **single camera (`ego_view_left`)**. To use both cameras (stereo),
edit the VIDEO block in `modality_config.py` (one-line swap). See the comments there.

## What was changed (for review)

Data layer:
- `gr00t/data/tactile_layout.py` (new) — the spec-sheet 256→112 valid-channel map,
  6 regions `[48,40,8,4,8,4]`, validated against the dataset.
- `gr00t/data/types.py` — `VLAStepData.tactile` field.
- `gr00t/data/dataset/lerobot_episode_loader.py` — `ALLOWED_MODALITIES += "tactile"`;
  load `observation.tactile_raw` verbatim (bypasses slicing).
- `gr00t/data/dataset/sharded_single_step_dataset.py` — window tactile by
  `delta_indices` (current + future frames) and carry it in `VLAStepData`.

Model layer:
- `gr00t/model/modules/tactile_encoder.py` (new) — per-region encoder + slot
  aggregator, EMA target encoder, dream head, touch-dreaming loss.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — inject tactile tokens into `sa_embs`
  (before action tokens; tail-anchored action decode unaffected) and add the
  touch-dreaming loss; `use_tactile` / `tune_tactile` gating.
- `gr00t/configs/model/gr00t_n1d7.py` — tactile config fields (defaults from the
  layout map; `valid_idx=None` → fallback to all 256 as one region).

Launch / data-config:
- `gr00t/model/gr00t_n1d7/setup.py` — pass `use_tactile`/`tune_tactile` into the
  checkpoint rebuild; allow the new tactile params to be missing from a
  tactile-free base checkpoint (initialized fresh, like `mask_token`).
- `gr00t/configs/finetune_config.py` + `gr00t/experiment/launch_finetune.py` —
  `--use_tactile` / `--tune_tactile` CLI flags.
- `examples/sonic_tactile/modality_config.py` — opt-in tactile modality + camera
  selection (the registered `unitree_g1_sonic` config is left unchanged, so
  tactile-free SONIC datasets keep working).

Tests (CPU): `tests/test_tactile_data_path.py`, `tests/test_tactile_encoder.py`,
`tests/test_tactile_action_head.py`.

## Hyperparameters (model config, defaults)

`n_tactile_tokens=8`, `dream_horizon=4` (≈80ms @ 50fps), `ema_decay=0.99`,
`lambda_tactile=0.5`, `tactile_dream_beta=1.0`. Override via the model config if needed.
