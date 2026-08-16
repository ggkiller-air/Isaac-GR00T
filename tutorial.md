# Isaac-GR00T SONIC tactile training

This branch provides three comparable modes from one codebase. Future observations are
training-only targets. JEPA conditions on a rolling four-frame tactile history; HTD conditions
on the current tactile frame only.

| `--tactile-mode` | Tactile condition | Future targets | Target representation |
|---|---|---|---|
| `notactile` | none | none | none |
| `htd` | current frame | tactile | absolute latent |
| `jepa` (UniVLaT/JEPA) | four-frame history | tactile, state, stereo | future minus current latent |

The old `--use-tactile {notac,input,dream}`, `--dream-state`, and `--dream-vision`
flags remain supported when `--tactile-mode` is omitted.

The JEPA training window is `[-3,-2,-1,0,1,2,3,4]`. Only `[-3..0]` enters the policy;
`[1..4]` is teacher-only. At episode and deployment starts, the first observed tactile frame is
repeated to fill missing history. `Gr00tPolicy.reset()` clears this rolling history.

HTD is short for *Humanoid Transformer with Touch Dreaming* (arXiv:2604.13015). In this
port, HTD mode names the current-tactile fusion and future-tactile latent objective; it is
not a claim that GR00T reproduces the paper's complete policy and controller system.

## Environment

```bash
cd /home/wzh/Projects/Uni_VLaT/Isaac-GR00T
uv sync --python 3.10 --all-extras
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

On Ubuntu 20.04/A800, the locked prebuilt `flash-attn` wheel needs newer GLIBC. Build it
once locally, then keep using `uv run --no-sync` so `uv` does not reinstall that wheel:

```bash
CUDA_VISIBLE_DEVICES='' CUDA_HOME=/usr/local/cuda-12.8 FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin" \
uv pip install --python .venv/bin/python --reinstall-package flash-attn --no-deps \
  --no-binary flash-attn --no-build-isolation 'flash-attn==2.7.4.post1'
```

## Full training

These are the complete 20k-step runs used for the three comparable experiments. The tested
four-A800 setting uses global batch 32, four data workers per rank, W&B logging, and keeps
only the latest checkpoint.

```bash
cd /home/wzh/Projects/Uni_VLaT/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export WANDB_MODE=online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export GR00T_BASE_MODEL_PATH=/home/wzh/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495
export GR00T_BACKBONE_NAME=/home/wzh/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

COMMON_ARGS=(
  --base-model-path "$GR00T_BASE_MODEL_PATH"
  --dataset-path /home/wzh/Projects/Uni_VLaT/data/desk_sweep
  --embodiment-tag UNITREE_G1_SONIC
  --modality-config-path gr00t/configs/data/embodiment_configs.py
  --num-gpus "$NUM_GPUS"
  --global-batch-size 64
  --dataloader-num-workers 8
  --max-steps 50000
  --save-steps 5000
  --save-total-limit 5
  --use-wandb
  --wandb-project univlat
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
  --tactile-encoder-type mlp
)

uv run --no-sync torchrun --nproc_per_node="$NUM_GPUS" --master_port=29501 \
  gr00t/experiment/launch_finetune.py "${COMMON_ARGS[@]}" \
  --tactile-mode notactile --output-dir outputs/sonic_notactile

uv run --no-sync torchrun --nproc_per_node="$NUM_GPUS" --master_port=29502 \
  gr00t/experiment/launch_finetune.py "${COMMON_ARGS[@]}" \
  --tactile-mode htd --output-dir outputs/sonic_htd

uv run --no-sync torchrun --nproc_per_node="$NUM_GPUS" --master_port=29503 \
  gr00t/experiment/launch_finetune.py "${COMMON_ARGS[@]}" \
  --tactile-mode jepa --output-dir outputs/sonic_jepa
```

Checkpoints are emitted at steps 10000 and 20000. Because `--save-total-limit 1` is set,
only `outputs/sonic_<mode>/checkpoint-20000` remains after training.

## Native GR00T deployment

Start the GR00T ZMQ policy server:

```bash
cd /home/wzh/Projects/Uni_VLaT/Isaac-GR00T
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/sonic_jepa/checkpoint-20000 \
  --embodiment-tag UNITREE_G1_SONIC --device cuda:0 --port 5550
```

Then start the unchanged SONIC workflow from the shared controller repository:

```bash
cd /home/wzh/Projects/Uni_VLaT/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 --policy-port 5550 \
  --camera-host 192.168.123.164 --tactile-zmq-host 192.168.123.164 \
  --prompt "carry the bucket"
```

For a `notactile` checkpoint, omit `--tactile-zmq-host` and add `--no-use-tactile`.

## Shared `sonic_vla_v1` boundary

Every backend must accept `state: float32[46]`, `ego_view_left/right: uint8[H,W,3]`,
`prompt: str`, and tactile `uint8[768]` only when its metadata says it is required. It must
return finite `actions: float32[40,78]` with
`motion_token[0:64] | left_hand[64:71] | right_hand[71:78]`. SONIC decodes the 64-D motion
token and controls the G1; the VLA does not directly output whole-body joint commands.
