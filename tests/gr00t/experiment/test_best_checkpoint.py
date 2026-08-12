import json
from types import SimpleNamespace
from unittest.mock import MagicMock

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
