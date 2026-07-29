# Isaac-GR00T SONIC tactile training

This branch provides three comparable modes from one codebase. Future observations are
training-only targets; inference always conditions on the current state, current stereo pair,
prompt, and (for HTD/JEPA) current tactile packet.

| `--tactile-mode` | Current tactile | Future tactile | Future state | Future stereo |
|---|---:|---:|---:|---:|
| `notactile` | no | no | no | no |
| `htd` | yes | yes | no | no |
| `jepa` (UniVLaT/JEPA) | yes | yes | yes | yes |

The old `--use-tactile {notac,input,dream}`, `--dream-state`, and `--dream-vision`
flags remain supported when `--tactile-mode` is omitted.

HTD is short for *Humanoid Transformer with Touch Dreaming* (arXiv:2604.13015). In this
port, HTD mode names the current-tactile fusion and future-tactile latent objective; it is
not a claim that GR00T reproduces the paper's complete policy and controller system.

## Environment

```bash
cd /root/Projects/Isaac-GR00T
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

## Two-GPU smoke runs

Check `nvidia-smi` first and use only idle GPUs. These commands run two optimizer steps; they
verify the real data/model path but are not training experiments.

```bash
cd /root/Projects/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=2,3
export NUM_GPUS=2

COMMON_ARGS=(
  --base-model-path nvidia/GR00T-N1.7-3B
  --dataset-path /root/Projects/data/carry-bucket-stereo
  --embodiment-tag UNITREE_G1_SONIC
  --modality-config-path gr00t/configs/data/embodiment_configs.py
  --num-gpus "$NUM_GPUS"
  --global-batch-size 2
  --dataloader-num-workers 2
  --max-steps 2
  --save-steps 2
  --save-total-limit 1
  --tactile-encoder-type mlp
)

uv run --no-sync torchrun --nproc_per_node=2 --master_port=29501 \
  gr00t/experiment/launch_finetune.py "${COMMON_ARGS[@]}" \
  --tactile-mode notactile --output-dir outputs/sonic_notactile_smoke

uv run --no-sync torchrun --nproc_per_node=2 --master_port=29502 \
  gr00t/experiment/launch_finetune.py "${COMMON_ARGS[@]}" \
  --tactile-mode htd --output-dir outputs/sonic_htd_smoke

uv run --no-sync torchrun --nproc_per_node=2 --master_port=29503 \
  gr00t/experiment/launch_finetune.py "${COMMON_ARGS[@]}" \
  --tactile-mode jepa --output-dir outputs/sonic_jepa_smoke
```

Checkpoints are written to `outputs/sonic_<mode>_smoke/checkpoint-2`. Increase
`--max-steps`, batch size, save interval, and output path for a real run; add `--use-wandb`
only when desired.

## Native GR00T deployment

Start the GR00T ZMQ policy server:

```bash
cd /root/Projects/Isaac-GR00T
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/sonic_jepa_smoke/checkpoint-2 \
  --embodiment-tag UNITREE_G1_SONIC --device cuda:0 --port 5550
```

Then start the unchanged SONIC workflow from the shared controller repository:

```bash
cd /root/Projects/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 --policy-port 5550 \
  --camera-host 192.168.123.164 --tactile-zmq-host 192.168.123.164 \
  --prompt "carry the bucket"
```

For a `notactile` checkpoint, omit `--tactile-zmq-host` and add `--no-use-tactile`.

## Shared `sonic_vla_v1` boundary

Every backend must accept `state: float32[46]`, `ego_view_left/right: uint8[H,W,3]`,
`prompt: str`, and tactile `uint8[256]` only when its metadata says it is required. It must
return finite `actions: float32[40,78]` with
`motion_token[0:64] | left_hand[64:71] | right_hand[71:78]`. SONIC decodes the 64-D motion
token and controls the G1; the VLA does not directly output whole-body joint commands.
