import torch


class DistManager:
    """Single-process device and sampler manager."""

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_training = True

    @property
    def device(self):
        return self._device

    @property
    def is_initialized(self):
        return False

    @property
    def world_size(self):
        return 1

    @property
    def rank(self):
        return 0

    @property
    def is_master(self):
        return True

    def setup(self, backend="nccl"):
        pass

    def cleanup(self):
        pass

    def data_parallel(self, model):
        return model

    def sync_batch_norm(self, model):
        return model

    def scaled_lr(self, base_lr):
        return base_lr

    def broadcast_object(self, data, src=0):
        return data

    def get_sampler(self, dataset, shuffle=True, drop_last=True):
        if shuffle:
            return torch.utils.data.RandomSampler(dataset)
        return torch.utils.data.SequentialSampler(dataset)


dist_manager = DistManager()
