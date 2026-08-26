from gr00t.configs.finetune_config import FinetuneConfig


def test_default_training_schedule_has_no_validation():
    config = FinetuneConfig(
        base_model_path="unused",
        dataset_path="unused",
        embodiment_tag="UNITREE_G1_SONIC",
    )

    assert config.max_steps == 100000
    assert config.save_steps == 10000
    assert config.save_total_limit == 5
    assert config.eval_strategy == "no"
