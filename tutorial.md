# 1.Training

## 1.1 modify configs
- 配置所需模态 line70
[gr00t/configs/data/embodiment_configs.py](gr00t/configs/data/embodiment_configs.py)

- 配置tactile encoder 包括通道数归一化数值大小CNN通道数
  你可以搜索cnn / tactile
[gr00t/configs/model/gr00t_n1d7.py](gr00t/configs/model/gr00t_n1d7.py)

## 1.2 launch training
```bash
tmux new -s dream_state

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

uv run torchrun --nproc_per_node=4 --master_port=29501 \
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