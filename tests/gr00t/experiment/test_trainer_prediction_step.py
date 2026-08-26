from types import SimpleNamespace

from gr00t.experiment.trainer import Gr00tTrainer
import torch
from transformers.feature_extraction_utils import BatchFeature


class _Model:
    def __init__(self):
        self.action_head = SimpleNamespace(config=SimpleNamespace(state_history_length=1))
        self.inference_inputs = None

    def get_action(self, inputs):
        self.inference_inputs = inputs
        return {"action_pred": torch.ones(1, 2, 3)}


def test_prediction_step_reads_collator_inputs_and_removes_training_targets():
    trainer = object.__new__(Gr00tTrainer)
    trainer._prepare_inputs = lambda inputs: inputs
    model = _Model()
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.args = SimpleNamespace(seed=42)
    batch = BatchFeature(
        data={
            "inputs": {
                "action": torch.zeros(1, 2, 3),
                "action_mask": torch.ones(1, 2, 3),
                "state": torch.zeros(1, 2, 4),
                "tactile": torch.zeros(1, 4, 8),
                "input_ids": torch.ones(1, 2, dtype=torch.long),
            }
        }
    )

    loss, logits, labels = trainer.prediction_step(model, batch, False)

    assert loss.item() == 1.0
    assert logits is None
    assert labels is None
    assert "action" not in model.inference_inputs
    assert "action_mask" not in model.inference_inputs
    assert model.inference_inputs["state"].shape == (1, 1, 4)
    assert model.inference_inputs["tactile"].shape == (1, 1, 8)
