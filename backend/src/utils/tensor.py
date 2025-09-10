import torch


def safe_tensor_convert(
    tensor: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Safely convert tensor with NaN handling and dtype conversion."""
    # Convert to target dtype if needed
    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)

    # Move to CPU for storage
    tensor_cpu = tensor.cpu()

    # Handle NaN values
    if torch.isnan(tensor_cpu).any():
        print("Warning: NaN values detected, replacing with zeros")
        tensor_cpu = torch.nan_to_num(tensor_cpu, nan=0.0)

    return tensor_cpu
