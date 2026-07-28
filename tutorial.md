# 1.Training

## 1.1 modify configs
- 配置所需模态 line70
[gr00t/configs/data/embodiment_configs.py](gr00t/configs/data/embodiment_configs.py)

- 配置tactile encoder 包括通道数归一化数值大小CNN通道数
  你可以搜索cnn / tactile
[gr00t/configs/model/gr00t_n1d7.py](gr00t/configs/model/gr00t_n1d7.py)

## 1.2 Ubuntu 20.04 / A800 flash-attn

This host uses GLIBC 2.31, while the prebuilt `flash-attn` wheel pinned by
`uv.lock` requires GLIBC 2.32. Build the same package version locally for SM80:

```bash
CUDA_VISIBLE_DEVICES='' \
CUDA_HOME=/usr/local/cuda-12.8 \
FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=8 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin" \
uv pip install --python .venv/bin/python \
    --reinstall-package flash-attn \
    --no-deps \
    --no-binary flash-attn \
    --no-build-isolation \
    'flash-attn==2.7.4.post1'
```

On this host, use `uv run --no-sync` after the local build. A plain `uv run`
reinstalls the incompatible prebuilt wheel from `uv.lock` before launching.

## 1.3 launch training
```bash
tmux new -s dream_state

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

uv run --no-sync torchrun --nproc_per_node=4 --master_port=29501 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path data/carry-bucket-stereo \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus $NUM_GPUS \
    --output-dir outputs/dream_full \
    --save-total-limit 1 \
    --save-steps 10000 \
    --max-steps 20000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    --use-tactile dream \
    --dream-state \
    --dream-vision \
    --tactile-encoder-type mlp
```
--use-tactile input \  #dream/input/notac
--tactile-encoder-type coord  #mlp/cnn/coord
