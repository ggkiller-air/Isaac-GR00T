from gr00t.experiment.experiment import wandb_run_metadata


def test_wandb_metadata_uses_environment_overrides(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_NAME", "task / isaac_groot / seed42 / stamp")
    monkeypatch.setenv("WANDB_RUN_GROUP", "task")
    monkeypatch.setenv("WANDB_JOB_TYPE", "train-isaac_groot")
    monkeypatch.setenv("WANDB_TAGS", "task:task, backbone:isaac_groot,stage:anchor")

    assert wandb_run_metadata("local-run", "train") == {
        "entity": "team",
        "name": "task / isaac_groot / seed42 / stamp",
        "group": "task",
        "job_type": "train-isaac_groot",
        "tags": ["task:task", "backbone:isaac_groot", "stage:anchor"],
    }


def test_wandb_metadata_preserves_existing_defaults(monkeypatch):
    for name in (
        "WANDB_ENTITY",
        "WANDB_NAME",
        "WANDB_RUN_GROUP",
        "WANDB_JOB_TYPE",
        "WANDB_TAGS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert wandb_run_metadata("local-run", "train") == {
        "entity": None,
        "name": "Isaac-GR00T / local-run",
        "group": "sonic-htd-model-comparison",
        "job_type": "comparison-training",
        "tags": ["train"],
    }
