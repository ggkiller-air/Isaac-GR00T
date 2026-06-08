# 1.Training

## 1.1 modify configs

gr00t/configs/data/embodiment_configs.py #line70

## 1.2 launch training
```bash
tmux new -s tactile_ft

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run torchrun --nproc_per_node=4 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path data/carry-bucket-stereo \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus $NUM_GPUS \
    --output-dir outputs/tactile_no \
    --save-total-limit 1 \
    --save-steps 10000 \
    --max-steps 20000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    --use-tactile \
    --tactile-encoder-type cnn \
    --tactile-cnn-coord
```

# 2. tactile flow graph

> 符号：B=batch，τ=`dream_horizon`(默认4)，τ+1=`delta_indices` 取的帧数(当前1 + 未来τ)，
> N=`n_tactile_tokens`(默认8)，H=action_horizon，
> **D_emb=`input_embedding_dim`**(token 宽，进 sa_embs / DiT 序列；图中 1536)，
> **D_hid=`hidden_size`**(DiT 主干隐藏宽 / DiT 输出特征宽)，512=`tactile_hidden_dim`(MLP 中间宽)。

```mermaid
flowchart TD
    %% ---------- 数据层 ----------
    P["parquet · observation.tactile_raw<br/>uint8[256]"]
    V["VLAStepData.tactile['tactile_raw']<br/>float32[τ+1, 256]<br/>(未切有效位 · 未归一化)"]
    BT["batch['tactile']<br/>float32[B, τ+1, 256]"]
    P -->|"loader 整列搬运<br/>delta_indices[0..τ] 取窗口 + vstack"| V
    V -->|"processor 透传原始256(不切)<br/>collator np.stack"| BT

    %% ---------- 当前帧分支 ----------
    subgraph CURB["当前帧分支 — 进 DiT 上下文"]
        direction TB
        CUR["cur = tactile[:, 0]<br/>[B, 256]"]
        SEL["select_and_normalize<br/>256→112 · /255<br/>[B, 112]"]
        ENC["PerRegionTactileEncoder<br/>split[48,40,8,4,8,4]<br/>每区 SmallMLP(size→512→D_emb)<br/>→ 6 区 token [B, 6, D_emb]"]
        AGG["TactileSlotAggregator<br/>8 可学 query cross-attn<br/>[B, N=8, D_emb]"]
        CUR --> SEL --> ENC --> AGG
    end

    %% ---------- 未来帧分支 ----------
    subgraph FUTB["未来帧分支 — 造监督目标 (no grad)"]
        direction TB
        FUT["future = tactile[:, 1:1+τ]<br/>[B, τ, 256]"]
        EMA["EMA teacher.encode_pooled<br/>同一 encoder + 对 8 slot 取均值<br/>→ z* target_latent [B, τ, D_emb] (detach)"]
        FUT --> EMA
    end

    BT -->|"取第 0 帧"| CUR
    BT -->|"取未来 τ 帧"| FUT

    %% ---------- DiT 主干 ----------
    SA["sa_embs = cat(state[B,1], tactile[B,8], action[B,H])<br/>[B, 1+8+H, D_emb]"]
    DIT["DiT (flow-matching)<br/>条件 = VLM 视觉/语言特征<br/>model_output [B, 1+8+H, D_hid]"]
    AGG --> SA --> DIT

    ACT["动作段 pred[:, -H:]<br/>→ action_decoder"]
    TR["触觉段 output[:, 1:1+N].mean<br/>= tactile_trunk [B, D_hid]"]
    DR["TactileDreamHead<br/>D_hid→512→τ·D_emb<br/>→ ẑ dream_pred [B, τ, D_emb]"]
    DIT --> ACT
    DIT --> TR --> DR

    %% ---------- 损失 ----------
    LT["L_tact = mean_k[ 1 − cos(ẑ_k, z*_k) + β·smoothL1(‖ẑ_k‖ − ‖z*_k‖) ]"]
    LTOT["L_total = L_action + λ_tactile · L_tact"]
    DR --> LT
    EMA --> LT
    ACT -->|"L_action (flow-matching)"| LTOT
    LT --> LTOT

    classDef loss fill:#ffe3e3,stroke:#e03131,color:#000;
    classDef teacher fill:#e7f0ff,stroke:#1c7ed6,stroke-dasharray:4 3,color:#000;
    class LT,LTOT loss
    class EMA teacher
```
