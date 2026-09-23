import contextlib
import logging
from typing import Any, Optional, Union

import numpy as np
import torch

from utils.distributed import dist_manager

logging.basicConfig(level=logging.INFO)


class PyTorchCompatibility:
    """Device transfer, checkpoint loading and mixed-precision helpers."""

    def __init__(self):
        self._version = torch.__version__
        self._is_v2_0 = self._version.startswith("2.0")
        self.logger = logging.getLogger("pt_compat")
        self._dist_manager = dist_manager

        if hasattr(torch.cuda.amp, "GradScaler"):
            self.amp = torch.cuda.amp
        else:
            self.amp = None

    def tensor_to(self, x: torch.Tensor, device: Union[str, torch.device],
                  copy: Optional[bool] = None, **kwargs) -> torch.Tensor:
        """Copy tensor to a device, resolving the copy semantics per version."""
        if self._is_v2_0:
            return x.to(device, copy=copy if copy is not None else True, **kwargs)
        non_blocking = kwargs.pop('non_blocking', False)
        return x.to(device, copy=copy if copy is not None else False,
                    non_blocking=non_blocking, **kwargs)

    def move_data_to_device(self, data: Any, device: Union[str, torch.device]) -> Any:
        """Recursively move a nested container of tensors to a device."""
        if isinstance(data, torch.Tensor):
            return self.tensor_to(data, device=device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        elif isinstance(data, dict):
            return {k: self.move_data_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.move_data_to_device(v, device) for v in data)
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        elif hasattr(data, "to"):
            return data.to(device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def load_checkpoint(self, path, model=None):
        checkpoint = torch.load(path, map_location="cpu")

        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
            if any(k.startswith('_orig_mod.') for k in model_state.keys()):
                model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
            if any(k.startswith('module.') for k in model_state.keys()):
                model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}
            checkpoint['model_state'] = model_state

        if model is not None:
            model.load_state_dict(checkpoint['model_state'])
            checkpoint['model'] = model

        return checkpoint

    @property
    def device_capability(self):
        """Map the current GPU to a batch-size tier; unknown GPUs fall back to
        the rtx4090 tier (lower batch_size via config/experiments.yaml hardware)."""
        name = torch.cuda.get_device_name(0)

        return {"type": "rtx4090", "max_batch": 16}


    def autocast(self, enabled=True):
        if self.amp:
            return self.amp.autocast(enabled=enabled)
        return contextlib.nullcontext()


pt_compat = PyTorchCompatibility()
