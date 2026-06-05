# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot
skills. This repo contains the model, training/fine-tuning pipeline, evaluation harness, and
multi-platform deployment tooling (ONNX/TensorRT export for dGPU, Jetson Orin/Thor, DGX Spark).

- **Package manager:** [uv](https://docs.astral.sh/uv/); build backend is setuptools (`pyproject.toml`)
- **Python:** 3.10 on dGPU/Orin, 3.12 on Thor/DGX Spark — the wired `pyproject.toml` pins `requires-python == 3.10.*`; per-platform variants live under `scripts/deployment/`
- **Submodules:** required. Clone with `--recurse-submodules` (or `git submodule update --init --recursive`). `git-lfs` is required to fetch parquet files in `demo_data/`.

## Quick-start commands

```bash
# Install (dGPU/x86_64, default). GPU deps (flash-attn, TensorRT) are in the default install.
uv sync --python 3.10
uv sync --all-extras          # also installs dev tools (ruff, pytest, pre-commit, ...)

# Other platforms: run the platform installer, then activate
bash scripts/deployment/{orin,thor,spark}/install_deps.sh
source scripts/activate_{orin,thor,spark}.sh   # exports platform library paths (dGPU needs none)

# Lint / format (ruff via pre-commit; run before committing)
pre-commit run --all-files

# Tests — markers: gpu, multigpu (default selection is CPU-safe)
python -m pytest tests/ -m "not gpu" -v --timeout=300   # CPU
python -m pytest tests/ -m gpu -v --timeout=300         # GPU
python -m pytest tests/path/to/test_x.py::test_name -v  # single test

# Build / lockfile
uv build
uv lock --locked
```

Code style: `ruff format` (double quotes, 4-space indent, line-length 100) + `ruff check` with
rules E/F/I, `E501` ignored. Config in `pyproject.toml` under `[tool.ruff]`.

## Architecture — the big picture

The central abstraction is the **embodiment tag**: a single `EmbodimentTag` enum value
(`gr00t/data/embodiment_tags.py`) selects everything that is robot-specific — which cameras/state/
action keys exist, how raw data is normalized, and which model head is used. Tags are grouped into
`PRETRAIN_TAGS`, `POSTTRAIN_TAGS`, and `FINETUNE_ONLY_TAGS` (e.g. `NEW_EMBODIMENT` for custom robots).
Almost every entry point takes `--embodiment-tag`.

Data flows through these layers (read together to understand any change to I/O):

1. **GR00T LeRobot format on disk** → loaded by `gr00t/data/dataset/` (`factory.py`,
   `lerobot_episode_loader.py`, and the sharded datasets for large mixtures).
2. **ModalityConfig** (`gr00t/data/types.py`) declares the `video`/`state`/`action`/`language`
   keys and their `delta_indices` (time windows) per embodiment. `DataConfig`
   (`gr00t/configs/data/`) binds modality configs + transforms for a dataset.
3. **State/action processing** (`gr00t/data/state_action/`) handles action chunking, pose math, and
   the relative-EEF action space that N1.7 shares across human and robot embodiments.
4. **Processor / transforms** — the N1.7 processor (`gr00t/model/gr00t_n1d7/`, exported as a HF
   `ProcessorMixin`, see `BaseProcessor` in `gr00t/data/interfaces.py`) turns observations into model
   inputs and decodes action outputs. Modality configs are carried inside the processor and keyed by
   embodiment tag value.
5. **Model** — VLM backbone (Cosmos-Reason2-2B / Qwen3-VL in N1.7) + action head, assembled under
   `gr00t/model/` (`base/`, `gr00t_n1d7/`, `modules/`). Config classes are paired with pipeline
   classes via `MODEL_REGISTRY` in `gr00t/model/registry.py` (`register_model(cfg_cls, pipeline_cls)`).

`Gr00tPolicy` (`gr00t/policy/gr00t_policy.py`) ties it together at inference: given an embodiment tag
it loads the model + processor via `AutoModel`/`AutoProcessor`, validates incoming observations
against the embodiment's modality config, and returns actions. `server_client.py` exposes it over
ZMQ for robot deployment; `replay_policy.py` replays recorded actions.

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <p> --dataset-path <p> --embodiment-tag <tag> --output-dir <d>` (wraps `gr00t/experiment/launch_finetune.py`; pretraining via `launch_train.py`, both using `trainer.py`)
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <p> --embodiment-tag <tag>`
- **Open-loop eval:** `gr00t/eval/open_loop_eval.py`; sim/real harnesses under `gr00t/eval/sim/` and `gr00t/eval/real_robot/`
- **Deploy export:** `scripts/deployment/export_onnx_n1d7.py`, `build_trt_pipeline.py`, `benchmark_inference.py` (see `scripts/deployment/README.md`)

## Layout (non-obvious dirs)

```
gr00t/
  configs/        configs: data/ (DataConfig, embodiment_configs), model/, training/, deepspeed/
  data/           dataset loaders, embodiment_tags, modality types, state_action, collators, stats
  model/          architecture: base/, gr00t_n1d7/, modules/, registry.py
  policy/         Gr00tPolicy, server/client, replay
  experiment/     training entry points + trainer + dist_utils
  eval/           run_gr00t_server, open_loop_eval, sim/, real_robot/
  deployment/     runtime modes
  utils/          determinism, video, initial_actions
examples/         per-embodiment configs + READMEs (DROID, LIBERO, SO100, SimplerEnv, robocasa, ...)
scripts/          deployment installers (dgpu/orin/thor/spark), lerobot_conversion, eval, utilities
getting_started/  user guides + GR00T_inference.ipynb (data_config, finetune_new_embodiment, ...)
demo_data/        sample datasets (git-lfs) used by tests/fixtures
```

## Notes

- CI: internal GitLab CI (`.gitlab-ci.yml` + `ci/`, not in the public repo) plus public GitHub Actions (`.github/workflows/`). Pytest uses `junit_duration_report = "total"` so fixture setup counts toward reported test time.
- CUDA 13.x (Thor/Spark/GB300): PyTorch 2.7 pins Triton 3.3.1, which rejects CUDA 13+. Run `uv run bash scripts/patch_triton_cuda13.sh`. `torch.compile` is unsupported on GB300 (sm_103) — use eager or TensorRT.
- `AGENTS.md` is a symlink to this file.

# Attention
回答问题时避免过分的夸赞。请记住，你的回答不一定是对的，我的判断也不一定是对的。对待所有问题都要反复推敲，优先保证准确性，必要时你可以主动向我索要补充信息或证据，回答时保持结构化输出，条理清晰。