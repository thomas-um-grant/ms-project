from src.prompts.dataset_generation_pompts import (
    TEMPLATES as DATASET_GENERATION_PROMPTS,
)
from src.prompts.retrieval_pipeline_pompts import TEMPLATES as RETRIEVAL_PROMPTS


def load_prompt_template(name: str, prompt_type: str, **kwargs) -> str:
    """
    Load a prompt template by name and format it with dynamic values.

    Args:
        name (str): The key for the prompt template.
        prompt_type (str): The type of prompt, either 'dataset_generation' or 'pipeline'.
        **kwargs: Dynamic values to fill into the template.

    Returns:
        str: The formatted prompt.

    Raises:
        KeyError: If the template name does not exist or a placeholder is missing.

    """
    try:
        if prompt_type == "dataset_generation":
            template = DATASET_GENERATION_PROMPTS[name]
        elif prompt_type == "pipeline":
            template = RETRIEVAL_PROMPTS[name]
        else:
            msg = f"Unknown prompt type: {prompt_type}, expected 'dataset_generation' or 'pipeline'."
            raise ValueError(msg)
    except KeyError as err:
        msg = f"Prompt template '{name}' not found."
        raise KeyError(msg) from err
    try:
        return template.format(**kwargs)
    except KeyError as e:
        msg = f"Missing value for placeholder: {e}"
        raise KeyError(msg) from e
