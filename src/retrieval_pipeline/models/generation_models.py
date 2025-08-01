from typing import Any

from retrieval_pipeline.models.base_generator import BaseGenerationModel


class ColQwen2Model(BaseGenerationModel):
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        # Initialize the model here (e.g., load from Hugging Face or local path)

    async def generate(self, query, context: list[Any] | None = None):
        """Generate an answer based on the input query and context."""
        # Implement the generation logic using the model


def setup_generation_model(
    model_name: str,
    device: str = "auto",
) -> BaseGenerationModel:
    """Setup the generation model based on the model name and device."""
    if model_name == "colqwen2":
        return ColQwen2Model(model_name, device)

    error_msg = f"Unsupported generation model: {model_name}"
    raise ValueError(error_msg)
