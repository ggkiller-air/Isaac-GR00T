# Tactile Fusion / JEPA 迁移说明

本文记录 `Isaac-GR00T` 的 tactile fusion、HTD touch-dreaming、state-JEPA 和
vision-JEPA 实验逻辑，作为向 `openpi`、`starVLA`、`DiT4DiT` 迁移时的实现契约。

审阅基线：`JEPA` 分支 `cdb9460`。本文描述的是当前实现，而不是对 HTD 论文的完整复现声明。

## 1. 功能边界

用户侧有三个 tactile 模式：

| 模式 | 触觉条件 | 未来预测 | Target | 推理额外开销 |
| --- | --- | --- | --- | --- |
| `notac` | 无 | 无 | 无 | 无 |
| `input` | 当前帧 | 无 | 无 | tactile encoder |
| `dream` / HTD | 当前帧 | tactile | absolute latent | tactile encoder |
| named `jepa` | 4 帧历史 | tactile/state/vision | `z(t+k)-z(t)` | tactile + temporal encoder |

内部开关映射为：

```text
notac -> use_tactile=False, use_tactile_dream=False
input -> use_tactile=True,  use_tactile_dream=False
dream -> use_tactile=True,  use_tactile_dream=True
```

`dream_state` 和 `dream_vision` 只在 `dream` 模式生效。named `jepa` 同时启用 temporal encoder、
三个 predictor 和 delta target。所有 predictor/teacher 只用于训练；部署保留 tactile encoder、
temporal encoder 和 action policy。

## 2. 数据契约

### 2.1 Tactile

数据由 `vest`、`left_arm`、`right_arm` 三路 `uint8[256]` 组成，并按该顺序拼成 `uint8[768]`。
数据管线保持原始数值和通道顺序，
只负责构造时间窗；有效通道选择和归一化必须在模型 encoder 内完成，以确保 train/infer 一致。

当前 `unitree_g1_sonic` 布局为 624 个有效通道：

| 区域 | 网格 | 通道数 |
| --- | --- | ---: |
| `front_chest` | 6 x 8 | 48 |
| `back` | 5 x 8 | 40 |
| `left_arm` | 2 x 4 | 8 |
| `left_shoulder` | 1 x 4 | 4 |
| `right_arm` | 2 x 4 | 8 |
| `right_shoulder` | 1 x 4 | 4 |
| `left_arm_device` | 16 x 16 | 256 |
| `right_arm_device` | 16 x 16 | 256 |

通道映射的唯一来源是 `gr00t/data/tactile_layout.py`。其中 spec 索引是 1-based，
`get_valid_idx()` 输出供 tensor 使用的 0-based 索引。迁移时不要手工重排这 624 个值。

HTD 训练窗口：

```text
tactile_raw: [B, 1 + tau, 256]
index 0:     当前帧，进入 action policy
index 1:    第一个未来 target
...
index tau:  最后一个未来 target
```

当前 `tau = dream_horizon = 4`，因此 modality config 使用 `delta_indices=range(5)`。
`input` 模式只加载和消费 index 0。

named `jepa` 的训练窗口为：

```text
tactile_raw: [B, history + tau, 768]
delta index:  [-3, -2, -1, 0, 1, 2, 3, 4]
condition:    前 4 帧（过去到当前）
target:       后 4 帧（仅 teacher 使用）
```

episode 左边界用首帧重复 padding。future target 不进入 temporal encoder，因此没有未来信息泄漏。

### 2.2 State

`dream_state=True` 时，state modality 同样加载 `range(1 + tau)`：

```text
state:          [B, state_history_length + tau, max_state_dim]
state_current:  state[:, :state_history_length]
state_future:   state[:, state_history_length:state_history_length + tau]
```

当前实现要求 `state_history_length == 1`。state 在进入 JEPA target encoder 前已经走过与主路径
相同的归一化和 padding。`CategorySpecificMLP` 保留时间维，因此 `[B, tau, D]` 会逐帧编码成
`[B, tau, D_emb]`。

### 2.3 Vision

`dream_vision=True` 时，launch 层把每个 camera 的 video `delta_indices` 改为
`range(1 + vision_horizon)`。processor 必须严格拆分：

```text
frame 0:     进入 VLM prompt，作为 policy condition
frame 1..vh: 仅作为 JEPA target，不能进入 VLM prompt
```

当前数据有两个相机。单样本 future image 顺序为 `frame outer, view inner`：

```text
t+1:left, t+1:right, t+2:left, t+2:right, ...
```

collator 再按 sample-major 拼接。vision tower 输出按每张图的 merged patch 数切分并 mean-pool，
最后 reshape 为 `[B, vision_horizon, num_views, D_vis]`，对 view 维取均值。迁移时必须同时保留
这个堆叠顺序和 reshape 顺序，否则不会报 shape 错误，但 target 时间语义会错位。

## 3. Tactile Encoder

`TactileEncoder` 的固定顺序是：

```text
raw[..., 768]
  -> select 624 valid channels
  -> cast to encoder dtype and divide by 255 once
  -> six independent region encoders
  -> [B, 8, input_embedding_dim]
  -> learnable-slot cross attention
  -> [B, n_tactile_tokens, input_embedding_dim]
```

默认参数：

```text
input_embedding_dim = 1536
n_tactile_tokens = 8
tactile_hidden_dim = 512
slot attention heads = 8
```

Region encoder 有三种用户选项：

- `mlp`：每个区域独立两层 MLP，是当前训练命令的基线。
- `cnn`：每个区域按自身网格 reshape，`3x3 conv -> adaptive pool -> linear`。
- `coord`：在 CNN 输入追加缩放后的 row/column CoordConv 通道。

slot attention 显式使用 PyTorch math SDPA。原因是 `1536 / 8 = 192` 的 head dimension 在当前
bf16 FlashAttention backward 中出现过 NaN；该 attention 只有 8 个 query 和 6 个 key，使用
math backend 的代价很小。

`encode_pooled()` 对 slot 维取均值，用于 teacher latent：

```text
[B, tau, 768] -> [B, tau, 8, 1536] -> [B, tau, 1536]
```

named `jepa` 对每个历史帧先独立运行 `TactileEncoder`，再按 slot 通过一层 bottleneck
Transformer 融合时间维，最后只输出当前时刻的 8 个 slot token。默认历史长度 4、hidden width
512、8 heads。Transformer 只看到过去到当前，不使用 future target。

## 4. Policy Fusion

融合发生在 action DiT 的 token 序列，不进入 VLM prompt：

```text
state token(s) | tactile slot tokens | noised action tokens
      1        |          8          |         40
```

训练和推理使用相同的 tactile/temporal encoder。在线推理由 `Gr00tPolicy` 维护滚动历史，首帧重复
填充，`reset()` 清空。tactile token 放在 action token 之前，action decoder 始终取输出序列最后
`action_horizon` 个 token，因此增加 tactile token 不改变 action 输出的索引和形状。

迁移到其他 policy 时，融合位置应满足：

1. tactile 能参与 action trunk 的上下文建模；
2. 不改变预训练 action token 的相对顺序；
3. action decode 使用显式 mask 或 tail slice，不能依赖融合前的绝对起始下标；
4. `use_tactile=False` 时不创建 tactile 模块，主模型计算图保持 baseline；
5. `use_tactile=True` 但部署观测缺失时，可 zero-fill 保持 shape，但这只是容错，不代表无性能损失。

## 5. JEPA Trunk 与 Target

DiT 输出宽度是 `hidden_size=1024`。当前实现从 post-DiT 的 tactile token 位置取共享 trunk：

```text
tactile_trunk = model_output[:, 1:1+n_tactile_tokens].mean(dim=1)
shape = [B, 1024]
```

这里的起点 `1` 假设 state token 数固定为 1。迁移时不要照抄常数，应由实际 token layout 计算
tactile slice 或显式携带 token mask。

三个 predictor 都是两层 MLP，一次输出整个 horizon：

| 分支 | Predictor 输出 | Target encoder | Target shape |
| --- | --- | --- | --- |
| tactile | future tactile latent | EMA(`tactile_encoder`) | `[B, tau, 1536]` |
| state | future state latent | EMA(`state_encoder`) | `[B, tau, 1536]` |
| vision | future visual latent | frozen vision tower | `[B, vh, 2048]` |

target 路径必须位于 `no_grad` 中并在 loss 前 detach。predictor 和 shared tactile trunk 接受梯度，
EMA/frozen teacher 不接受梯度。

named `jepa` 的三个 target 均在同一 teacher space 内改为 `z(t+k)-z(t)`；HTD 仍使用 absolute
future tactile latent。这样 `absolute z representation` 可作为独立 ablation。

## 6. Loss

每个 JEPA 分支复用同一个方向加模长损失：

```text
L_jepa = mean(1 - cosine(pred, target)
              + beta * smooth_l1(norm(pred), norm(target)))
```

delta target 可能严格或近似为 0；此时 cosine 没有定义。实现会屏蔽 target norm `<=1e-6` 的方向项，
但保留 magnitude 项，继续约束预测也接近 0。

模长项用于降低纯 cosine 表征塌缩风险。当前 `beta = tactile_dream_beta = 1.0`。

总损失：

```text
L_total = L_action
        + lambda_tactile * L_tactile
        + lambda_state   * L_state    # optional
        + lambda_vision  * L_vision   # optional
```

三个 lambda 当前默认都是 `0.5`。W&B 应分别记录 `action_loss`、`tactile_loss`、
`state_jepa_loss`、`vision_jepa_loss` 和 total loss，避免只看 total 时无法判断辅助任务是否塌缩。

## 7. EMA 与 Checkpoint 生命周期

正确生命周期如下：

1. 构造 online encoder；
2. deepcopy 为 teacher，立即 `requires_grad_(False)` 和 `eval()`；
3. 加载 base checkpoint 到 online model；
4. checkpoint 加载完成后，再执行 `teacher.load_state_dict(online.state_dict())`；
5. optimizer step 后更新 `teacher = decay * teacher + (1-decay) * online`；
6. 保存 checkpoint 时同时保存 online、teacher 和 predictor。

第 4 步不能省略。当前 GR00T 的 `state_encoder` 来自 base checkpoint，但 teacher 在权重加载前已经
deepcopy；若不重新同步，第一次 JEPA target 会来自随机 state encoder。当前 `setup.py` 已在所有
rank 上同步 state/tactile teacher。

加载 tactile-free base checkpoint 时，只允许下列新模块缺失：tactile encoder、target encoder、
dream predictor。其他 missing key、任何 unexpected key 或 mismatched key 都应失败，而不是宽泛忽略。

## 8. 配置与持久化契约

以下字段必须从 CLI/experiment config 一直传到 model constructor 和 checkpoint config：

```text
use_tactile, use_tactile_dream, tune_tactile
tactile_raw_dim, tactile_valid_idx, tactile_region_sizes
tactile_encoder_type, tactile_region_rows, tactile_region_cols
tactile_cnn_channels, tactile_cnn_pool, tactile_cnn_coord, tactile_cnn_coord_scale
n_tactile_tokens, tactile_hidden_dim
use_tactile_temporal, tactile_history_length
tactile_temporal_layers, tactile_temporal_heads
use_delta_targets
dream_horizon, ema_decay, lambda_tactile, tactile_dream_beta
dream_state, lambda_state
dream_vision, lambda_vision, vision_horizon
```

processor checkpoint 还必须保存 `dream_vision` 和 `vision_horizon`，因为它们决定推理/训练时如何拆分
video window。modality config 和 tactile layout 也必须随 checkpoint 固化，不能依赖目标仓库的新默认值。

## 9. 迁移实现顺序

建议每个仓库按以下顺序开发，每一步都有独立可验证的行为：

1. 接通原始 tactile 数据列和当前/未来窗口，先验证时间对齐与 episode boundary。
2. 移植 layout、一次性 `/255`、per-region encoder 和 slot aggregator。
3. 只实现 `input` 模式，在 policy trunk 注入当前 tactile token，并验证 action shape/train/infer。
4. 加 `notac` 回归测试，确认 baseline 没有新模块和新 loss。
5. 加 tactile EMA teacher、predictor 和 `L_tactile`。
6. 加 state future window、state EMA teacher 和 `L_state`。
7. 加 vision current/future split、frozen vision target 和 `L_vision`。
8. 接 optimizer-step EMA、checkpoint save/load、分布式和日志。
9. 最后做真实数据单 batch forward/backward，再做短训练 smoke test。

不要先移植所有分支再一次性调试；`input`、tactile dream、state、vision 四层开关本身就是必要的
ablation 和故障隔离面。

## 10. 必须具备的测试

### 数据与 layout

- 768 -> 624 的索引唯一、范围正确、region size/grid 一致。
- tactile/state/video 的 index 0 和未来 index 1..horizon 时间对齐。
- episode 尾部 padding 不跨 episode。
- 双相机 future image 顺序和 reshape 顺序一致。

### Encoder

- MLP/CNN/CoordConv 的输出 shape 和 backward。
- raw 只归一化一次；0 和 255 分别映射到 0 和 1。
- teacher 初始等于 student，EMA 更新方向正确，teacher 永远无梯度。

### Policy 集成

- `notac` 不创建 tactile 模块，不产生辅助 loss。
- `input` 有 action-loss 到 tactile encoder 的梯度，但没有 teacher/predictor。
- `dream` 的 tactile encoder、predictor、shared trunk 有梯度，teacher 无梯度。
- state/vision 分支可单独开启，也可同时开启。
- 缺 future window 时训练 fail fast，不能静默跳过辅助 loss。
- eval/get_action 不要求 future target，不更新 EMA，action shape 不变。
- tactile-free base checkpoint 只报告预期的新模块 missing keys，load 后 teacher 与 online 完全一致。

### 数值与短训练

- 每个 component loss finite，至少数十 step 内不是常数或 NaN。
- predictor/encoder 梯度 norm finite 且非零。
- 每个 optimizer step 只更新一次 EMA。
- DDP 各 rank 的开关、horizon、layout 和初始 teacher 一致。

## 11. 本次审阅已验证

- 当前 `tests/test_tactile_data_path.py` 和 `tests/test_tactile_encoder.py`：`12 passed`。
- 从历史 commit 内存执行 action-head tactile/state/vision 集成测试：`13 passed`。
- 实际 `carry-bucket-stereo` 单样本 CPU 数据链路：
  - `state [1,5,132]`
  - `tactile [1,5,256]`
  - `action [1,40,132]`
  - 当前图 2 张，未来图 `4 x 2 = 8` 张
- 完整 GR00T base checkpoint 在 CPU 成功增量构造四个 tactile/JEPA 分支：
  - 只有 78 个预期新参数 missing
  - 0 unexpected，0 mismatched
- pipeline checkpoint load 后，state/tactile teacher 与 online encoder 完全相等且全部冻结。
- vision future 顺序是 sample -> frame -> view，和 `[B,vh,V,D]` reshape 一致。

## 12. 当前实现的风险与迁移决策

以下不是本次环境 smoke test 的阻塞项，但迁移时必须显式处理：

1. **EMA 更新频率**：当前 GR00T 在每次 training forward 内更新 EMA。只有
   `gradient_accumulation_steps == 1` 时才等价于每 optimizer step 一次。其他仓库应把 EMA 放到
   optimizer-step hook；否则有效 decay 会改变。
2. **Vision teacher 冻结条件**：`no_grad` 只阻断 target 梯度，不会阻止 optimizer 改同一个 vision
   tower。启动器在 vision-JEPA 下要求 `tune_visual=False`；若未来需要 visual finetune，必须另建
   frozen/EMA teacher。
3. **State target augmentation**：processor 的 `state_dropout_prob` 会在 state future target 编码前
   把整个 state window 置零，而 action head 还会独立 dropout 当前 state feature。做干净 JEPA 对照时，
   建议 target 使用未增强 state，或先把 `state_dropout_prob` 设为 0。
4. **当前 EMA 位置存在 silent skip**：三个 JEPA loss 都嵌套在“tactile future window 有效”的分支内。
   若 tactile future 缺失，state/vision loss 也会静默消失。迁移实现应在训练启动或首 batch fail fast。
5. **参数透传不完整**：当前 `setup.py::from_pretrained` 已透传主要开关和 lambda，但没有透传所有
   layout、token 数、hidden width、CNN 参数；默认实验没问题，做 sweep 会静默回落到默认值。迁移时
   采用本文件第 8 节的完整字段表。
6. **测试文件缺口**：最新版已删除 `tests/test_tactile_action_head.py`；本次只能从历史 commit 执行。
   各目标仓库必须保留永久的 action-head 集成测试。
7. **“baseline 不变”的精确定义**：`use_tactile=False` 保持模型模块和 action 计算路径不变，但当前
   SONIC modality config 仍可能加载 tactile/future state，数据 I/O 不是严格 baseline。正式 ablation
   应分别记录模型行为和数据吞吐。
8. **数据类型开销**：loader 当前把 `uint8` tactile window 转成 float32，再由 encoder cast/除 255。
   语义正确但内存放大 4 倍。目标仓库可保持 uint8 到 device 前，但归一化位置和结果必须一致。

## 13. 数据驱动的非目标

`carry-bucket-stereo` 中有效 tactile 通道约 85.6% 为 0，帧间自相关约 0.997。基于当前数据：

- 不建议仅通过堆叠更多原始过去帧增加 temporal encoder；信息高度冗余。
- 当前可复现 baseline 必须先保留线性 `/255`。
- `sqrt`/`log` 等稀疏重尾输入缩放可作为独立对照，只改
  `TactileEncoder.select_and_normalize()`，并从头训练；不要和跨仓库迁移同时引入。
