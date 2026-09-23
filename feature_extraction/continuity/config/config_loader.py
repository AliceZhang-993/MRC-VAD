from pathlib import Path

import yaml


def load_config(exp_name: str) -> dict:
    config_path = Path(__file__).parent / "experiments.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    return configs["experiments"][exp_name]
