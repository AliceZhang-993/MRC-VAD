import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile

from torch.utils.tensorboard import SummaryWriter

from utils.distributed import dist_manager


class TrainingLogger:
    """TensorBoard scalar logger for the training loop."""

    def __init__(self, config, mode):
        self.config = config
        self.writer = None
        if dist_manager.is_master:
            now = datetime.now()
            timestamp = "_".join(list(map(lambda x: str(x).zfill(2), [now.year, now.month, now.day, now.hour, now.minute, now.second])))
            if mode == "train":
                log_dir = f'tensorboard_log/{config["expname"]}_{config["data"]["dataset_name"]}_bs{config["data"]["batch_size"]}/'
            else:
                log_dir = f'tensorboard_log/{config["expname"]}_{config["data"]["dataset_name"]}_bs{config["data"]["val_batch_size"]}_test_logs/'
            log_dir += f"{timestamp}"

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

            config_path = Path(__file__).parent.parent / "config/experiments.yaml"
            copyfile(config_path, f"{log_dir}/experiments.yaml")

    def log_train(self, loss, step):
        if dist_manager.is_master and self.writer is not None:
            self.writer.add_scalar("Loss/Train", loss, step)

    def log_scalar(self, tag, value, global_step):
        if dist_manager.is_master and self.writer is not None:
            self.writer.add_scalar(tag, value, global_step)

    def close(self):
        if dist_manager.is_master and self.writer is not None:
            self.writer.close()
