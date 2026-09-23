import argparse
import importlib
import os
import sys
import time
from pathlib import Path

# --gpus must be read before torch is imported so CUDA_VISIBLE_DEVICES applies.
if __name__ == "__main__":
    _early = argparse.ArgumentParser(add_help=False)
    _early.add_argument('--gpus', type=str, default='0')
    _early_args, _ = _early.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.gpus

import torch

_project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(_project_root))

from config import config_loader
from utils.compatibility import pt_compat
from utils.distributed import dist_manager


def run_experiment(exp_name, mode):
    dist_manager.setup()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = config_loader.load_config(exp_name)

    cap = pt_compat.device_capability
    config["data"]["batch_size"] = config["hardware"][cap["type"]]["batch_size"]
    config["data"]["val_batch_size"] = config["hardware"][cap["type"]]["val_batch_size"]

    if dist_manager.is_master:
        print(time.strftime('------------%Y-%m-%d %H:%M:%S------Here start vadnet!---------',
                            time.localtime(time.time())))
        print(f"hardware:{torch.cuda.get_device_name(0)},"
              f"dataset:{config['data']['dataset_name']},"
              f"train_batch_size:{config['data']['batch_size']},"
              f"val_batch_size:{config['data']['val_batch_size']}")

    try:
        test_module = importlib.import_module(f"{config['module_path']}.evaluate")
        train_module = importlib.import_module(f"{config['module_path']}.train")

        if mode == "train":
            dist_manager.is_training = True
            trainer = train_module.TrainingEngine(config, config["train"]["resumepoint_path"] or None)
            trainer.training_loop()
        elif mode == "test":
            dist_manager.is_training = False
            split_setting = config["test"].get("infer_split", "testing")
            splits_to_run = ["training", "testing"] if split_setting == "both" else [split_setting]

            for split in splits_to_run:
                config["test"]["infer_split"] = split
                if dist_manager.is_master:
                    print(f"\n[INFO] Running inference on '{split}' split...")

                inferencer = test_module.InferenceEngine(config, logger=None)

                checkpoints = []
                if dist_manager.is_master:
                    checkpoint_path = config["test"]["checkpoint_path"]
                    if not os.path.isabs(checkpoint_path):
                        checkpoint_path = os.path.join(str(_project_root), checkpoint_path)

                    if os.path.isdir(checkpoint_path):
                        dataset_name = config["data"]["dataset_name"]
                        per_dataset = os.path.join(checkpoint_path, f"{dataset_name}.pth")
                        if os.path.exists(per_dataset):
                            checkpoints = [per_dataset]
                        else:
                            checkpoints = sorted(
                                [os.path.join(checkpoint_path, f)
                                 for f in os.listdir(checkpoint_path) if f.endswith(".pth")])
                    else:
                        checkpoints = [checkpoint_path]
                    print(f"[INFO] [{split}] Found {len(checkpoints)} checkpoints in {checkpoint_path}")

                checkpoints = dist_manager.broadcast_object(
                    checkpoints if dist_manager.is_master else None, src=0)

                for ckpt in checkpoints:
                    if dist_manager.is_master:
                        print(f"[Testing][{split}] Checkpoint: {Path(ckpt).name}")
                    result = inferencer.validate(ckpt)
                    if result is None or result[0] is None:
                        if dist_manager.is_master:
                            print("[Main] Feature extraction complete.")
    except ModuleNotFoundError as e:
        print(f"experiment {exp_name} not exists: {e}")
    finally:
        dist_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='exp3', help='experiment name')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    run_experiment(exp_name=args.exp, mode=args.mode)
