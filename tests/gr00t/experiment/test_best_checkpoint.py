import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.experiment.utils import BestMetricCheckpointCallback


def test_existing_best_is_only_replaced_by_strictly_lower_metric(tmp_path):
    best_dir = tmp_path / "best_model"
    best_dir.mkdir()
    (best_dir / "metrics.json").write_text(
        json.dumps({"step": 10, "eval_action_mse": 0.2}), encoding="utf-8"
    )
    callback = BestMetricCheckpointCallback(
        "eval_action_mse", greater_is_better=False
    )
    model = MagicMock()
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(is_world_process_zero=True, global_step=20)

    callback.on_evaluate(
        args, state, None, {"eval_action_mse": 0.3}, model=model
    )
    model.save_pretrained.assert_not_called()

    callback.on_evaluate(
        args, state, None, {"eval_action_mse": 0.1}, model=model
    )
    model.save_pretrained.assert_called_once_with(best_dir)
    assert json.loads((best_dir / "metrics.json").read_text())["eval_action_mse"] == 0.1


def test_first_training_step_saves_step_zero_before_forward(monkeypatch):
    trainer = object.__new__(Gr00tTrainer)
    trainer._initial_checkpoint_saved = False
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace()
    trainer.control = SimpleNamespace()
    trainer.callback_handler = SimpleNamespace(
        on_save=lambda *_args: trainer.control
    )
    events = []
    trainer._save_checkpoint = lambda _model, trial: events.append(("save", trial))
    monkeypatch.setattr(
        "transformers.Trainer.training_step",
        lambda _self, _model, _inputs, num_items_in_batch=None: events.append("forward")
        or torch.tensor(1.0),
    )

    result = trainer.training_step(object(), {})

    assert result.item() == 1.0
    assert events == [("save", None), "forward"]
    assert trainer._initial_checkpoint_saved
