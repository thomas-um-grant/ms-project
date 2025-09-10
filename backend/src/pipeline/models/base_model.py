from abc import ABC, abstractmethod

import torch

from utils.device import DeviceConfig


class BaseModel(ABC):
    """Abstract base class for all models (embedding and generation). Defines common properties and an abstract method for model loading."""

    model_name: str
    device_config: DeviceConfig

    @property
    def device(self) -> str:
        """Get the device string for tensor operations."""
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype from device config."""
        return self.device_config.dtype

    @abstractmethod
    def _load_model(self) -> None:
        """Abstract method to load the model. Each concrete model will implement its specific loading logic here."""
