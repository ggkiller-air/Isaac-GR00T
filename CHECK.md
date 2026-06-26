# CHECK.md — HTD / JEPA dream 分支代码审查清单

分支:`JEPA`
相关 commit:`2747918 dream state`、`498553c no verify state and vision dream`

## 改动涉及的文件
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — 主 forward + dream 三个分支(tactile / state / vision)
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` — 未来帧/状态的数据切分 + collator
- `gr00t/model/gr00t_n1d7/setup.py` — checkpoint 加载 + 新模块豁免 / EMA 同步
- `gr00t/model/modules/tactile_encoder.py` — `touch_dreaming_loss` / `ema_update` / `build_ema_teacher`
- `gr00t/configs/model/gr00t_n1d7.py`、`gr00t/configs/finetune_config.py`、`gr00t/experiment/launch_finetune.py`、`gr00t/experiment/trainer.py` — CLI / config 透传
- `tests/test_tactile_action_head.py` — 测试

---

## 1. State-JEPA:future 窗口切分与 encoder 调用
**文件:** `gr00t_n1d7.py:365-379`(切分)、`485-503`(loss)

- [ ] `state_current = state_seq[:, :shl].reshape(B, 1, -1)` 喂给 `state_encoder` 是 `[B,1,shl*D]`;而 `state_future = state_seq[:, shl:shl+tau]` 是 `[B, tau, D]`。**验证 `state_encoder` / `state_target_encoder` 对 `[B,tau,D]` 多帧输入的处理,是否和单帧 `[B,1,D]` 语义一致**(逐帧编码,输出 `[B,tau,D_emb]`)。最容易出 shape/语义 bug 的点。
- [ ] `dream_state` 强制 `state_history_length==1`(assert 在 `gr00t_n1d7.py:177`)。确认数据集 state modality 的 `delta_indices` 被加宽到至少 `range(dream_horizon+1)`,否则 `state_future is None` 会直接抛错(`490`)。**config 侧依赖,去 modality config / embodiment_configs 确认。**
- [ ] 时间对齐:`pred_state[:, i]` 是否对应未来第 `i` 帧 = `state_future[:, i]`。确认 dream_head 输出的 horizon 顺序和 future 切片顺序一致。

## 2. Vision-JEPA:未来帧 collate 顺序 vs reshape 顺序(高风险)
**文件:** `gr00t_n1d7.py:843-866`(`_compute_vision_target`)、`processing_gr00t_n1d7.py`(`future_images` 分支)

- [ ] collator 注释说 "frame outer, view inner",而 `_compute_vision_target` 用 `pooled.view(B, vh, num_views, D).mean(dim=2)` 还原。**两边堆叠顺序必须严格一致**,否则 vision target 把不同视角/帧混在一起——不报错但语义全错。重点核对 `processing` 里 `future_pils` 双重循环顺序(frame 外、view 内)与 `view(B, vh, num_views)` 是否匹配。
- [ ] `counts = (fthw.prod(dim=-1) // (merge**2))` 假设每张图 token 数可变。验证 `embeds.split(counts)` 切分正确、`num_imgs % (B*vh)==0`(`863`)在变分辨率下成立。
- [ ] `vision_horizon` 默认回退到 `dream_horizon`(`gr00t_n1d7.py:198`),但 vision 分支断言 `vision_target.shape[1] == vh`。确认 config 里 vision_horizon 和未来帧切片数 `full[1:1+vision_horizon]` 一致。

## 3. EMA target encoder 初始化时序
**文件:** `setup.py:131-157`(豁免名单 + 重新同步)、`gr00t_n1d7.py:162/181`(`build_ema_teacher`)

- [ ] `state_target_encoder = deepcopy(state_encoder)` 发生在 `from_pretrained` **之前**,而 `state_encoder` 在 base ckpt 里会被覆盖 → setup.py 加了加载后 `load_state_dict` 重新同步。**验证这个同步确实在所有 rank 上、在 EMA 第一次 update 之前执行**,且 `model.action_head` 路径名正确。
- [ ] 反过来确认 `tactile_target_encoder`:`tactile_encoder` 是新模块(随机),它和它的 deepcopy 本就一致,重新同步对它是 no-op——确认没有反而引入问题。

## 4. Collapse / loss 数值
**文件:** `tactile_encoder.py:283-300`(`touch_dreaming_loss`)

- [ ] loss = `(1 - cos) + beta * smoothL1(norm)`。target 全程 `detach()`(`gr00t_n1d7.py:478/497/525`)。确认没有梯度从 target 漏回 EMA encoder。
- [ ] magnitude 项靠 `beta`(默认 1.0)抗 collapse。验证 `tactile_dream_beta` 真的透传到这里(CLI → config → forward)。若 beta=0,state/vision JEPA 容易塌成常数。

## 5. EMA 更新的开关与频率
**文件:** `gr00t_n1d7.py:476/494`(`ema_update` 调用)、`set_trainable_parameters:225-251`

- [ ] `ema_update` 只在 forward 的 dream 分支里调用——确认它**只在 training、每个 step 调一次**,不会在 grad accumulation 的每个 micro-step 都更新(会让 EMA 实际衰减率偏离 `ema_decay`)。
- [ ] 若 `tune_projector=False`,`state_encoder` 被冻结(`228`),则在线 encoder 不变、EMA target 恒等于它。确认这是预期(state-JEPA 退化成固定 teacher),还是应该让 state_encoder 可训。

## 6. 推理路径不受污染
**文件:** `gr00t_n1d7.py:886`(只在有 `future_pixel_values` 时算 vision target)、`get_action`

- [ ] 确认 `get_action` / 推理 batch 不带 `future_*`,三个 dream 分支全部跳过,动作输出与无 tactile-dream 时一致(数值回归)。
- [ ] `forward:886` 用 `"future_pixel_values" in backbone_inputs` 守护,核对 state 分支也有等价守护(`state_seq.shape[1] >= shl+tau`,`372`)。

## 7. Config / CLI 透传一致性
**文件:** `finetune_config.py`、`configs/model/gr00t_n1d7.py`、`launch_finetune.py`

- [ ] 核对 `--use-tactile / --dream-state / --dream-vision / --tactile-encoder-type / lambda_* / ema_decay / dream_horizon / vision_horizon / tactile_dream_beta` 从 CLI → finetune_config → model config → 模型 `getattr` 默认值全程名字一致、默认值不冲突(模型里大量 `getattr(config, x, default)`,容易出现"CLI 设了但 config 字段名拼错导致永远走默认"的静默 bug)。
- [ ] `save_pretrained` 是否把 `dream_*` 新字段持久化(processing 里有 `dream_vision`),确保 reload checkpoint 推理时配置不丢。

## 8. 测试覆盖
**文件:** `tests/test_tactile_action_head.py`

- [ ] 确认新测试真的覆盖 state/vision dream 分支的 shape 和 loss,而不仅是 tactile;以及是否有"推理路径无 future 输入"的回归测试。
- [ ] 直接跑:`uv run pytest tests/test_tactile_action_head.py -v`

---

## 优先级
1. **第 2 项**(vision collate 顺序)、**第 1 项**(state 多帧 encoder 语义)—— 最可能藏静默语义 bug,先查。
2. 第 3、5 项 —— 数值正确性。
3. 第 6、7 项 —— 回归安全。
