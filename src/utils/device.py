import logging
from dataclasses import dataclass
from enum import Enum

import torch


class DeviceType(Enum):
    """Supported device types for model execution."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DeviceMap(Enum):
    """Device mapping strategies for model loading."""

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """
    Configuration for device-specific settings.

    Args:
        device: The device type to use for computation
        dtype: The torch data type for model weights and computations
        device_map: The device mapping strategy for model loading

    """

    device: DeviceType
    dtype: torch.dtype
    device_map: DeviceMap

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.device, DeviceType):
            msg = f"device must be a DeviceType, got {type(self.device)}"
            raise TypeError(msg)
        if not isinstance(self.dtype, torch.dtype):
            msg = f"dtype must be a torch.dtype, got {type(self.dtype)}"
            raise TypeError(msg)
        if not isinstance(self.device_map, DeviceMap):
            msg = f"device_map must be a DeviceMap, got {type(self.device_map)}"
            raise TypeError(msg)

    @property
    def device_str(self) -> str:
        """Get device as string for torch operations."""
        return self.device.value

    @property
    def device_map_str(self) -> str:
        """Get device_map as string for model loading."""
        return self.device_map.value

    @classmethod
    def auto_detect(cls, preferred_device: str | None = None) -> "DeviceConfig":
        """
        Auto-detect the best device configuration for this system.

        Args:
            preferred_device: Optional preferred device string ("cuda", "mps", "cpu")

        Returns:
            DeviceConfig: Optimized configuration for the detected or preferred device

        """
        if preferred_device:
            return cls._get_config_for_device(preferred_device)

        # Auto-detection logic
        if torch.cuda.is_available():
            return cls._get_config_for_device("cuda")
        if torch.backends.mps.is_available():
            return cls._get_config_for_device("mps")
        return cls._get_config_for_device("cpu")

    @classmethod
    def _get_config_for_device(cls, device: str) -> "DeviceConfig":
        """
        Get optimal configuration for a specific device.

        Args:
            device: Device string ("cuda", "mps", "cpu")

        Returns:
            DeviceConfig: Optimized configuration for the specified device

        Raises:
            ValueError: If device string is not supported

        """
        if device == "cuda":
            return cls(
                device=DeviceType.CUDA,
                dtype=torch.float16,
                device_map=DeviceMap.AUTO,
            )
        if device == "mps":
            # Test MPS stability
            try:
                test_tensor = torch.randn(10, 10, dtype=torch.float16, device="mps")
                _ = torch.mm(test_tensor, test_tensor.T)
                return cls(
                    device=DeviceType.MPS,
                    dtype=torch.float16,
                    device_map=DeviceMap.MPS,
                )
            except RuntimeError:
                # Fallback to CPU if MPS has issues
                logger.warning("MPS instability detected, falling back to CPU")
                return cls._get_config_for_device("cpu")
        if device == "cpu":
            return cls(
                device=DeviceType.CPU,
                dtype=torch.float32,
                device_map=DeviceMap.CPU,
            )

        supported_devices = ", ".join([dt.value for dt in DeviceType])
        msg = f"Unsupported device: {device}. Supported devices: {supported_devices}"
        raise ValueError(msg)
