import asyncio
from typing import Any

from PIL import Image

from src.utils.device import cleanup_memory, log_memory_usage


class ImageProcessor:
    """Handles image processing operations including description generation."""

    def __init__(
        self,
        generation_model: Any,
        image_description_prompt: str = "Describe this image in great details, it will be used as the description metadata for retrieval. Return the description only in plain text.",
        fallback_description: str = "Image description unavailable",
        batch_completion_template: str = "✓ Batch {batch_num}/{total_batches}: {successful}/{total} successful",
    ) -> None:
        """
        Initialize the ImageProcessor.

        Args:
            generation_model: The model used for generating image descriptions
            image_description_prompt: Prompt to use for image description generation
            fallback_description: Description to use when generation fails
            batch_completion_template: Template for batch completion logging

        """
        self.generation_model = generation_model
        self.image_description_prompt = image_description_prompt
        self.fallback_description = fallback_description
        self.batch_completion_template = batch_completion_template

    async def extract_descriptions_batch(
        self,
        images: list[Image.Image],
        corpus_ids: list[str],
        batch_size: int = 4,
        max_retries: int = 3,
        sleep_between_images: float = 0.1,
        sleep_between_batches: float = 0.05,
        cleanup_every_n_batches: int = 3,
    ) -> list[str]:
        """
        Extract descriptions for multiple images.

        Args:
            images: List of PIL images to process
            corpus_ids: List of corresponding corpus IDs for logging
            batch_size: Number of images to process in each batch
            max_retries: Number of retry attempts per image
            sleep_between_images: Sleep time between individual images
            sleep_between_batches: Sleep time between batches
            cleanup_every_n_batches: How often to cleanup memory

        Returns:
            List of descriptions (or fallback description for failures)

        """
        descriptions = []
        total_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_ids = corpus_ids[i : i + batch_size]

            batch_num = i // batch_size + 1
            log_memory_usage(f"Batch {batch_num}/{total_batches} start")

            # Process each image in the batch with retry logic
            batch_descriptions = []
            for j, img in enumerate(batch_images):
                description = await self._generate_description_with_retries(
                    img,
                    batch_ids[j],
                    max_retries,
                )
                batch_descriptions.append(description)

                # Minimal delay between individual images
                if j < len(batch_images) - 1:
                    await asyncio.sleep(sleep_between_images)

            descriptions.extend(batch_descriptions)

            # Show batch completion status
            successful = sum(
                1 for desc in batch_descriptions if desc != self.fallback_description
            )
            print(
                self.batch_completion_template.format(
                    batch_num=batch_num,
                    total_batches=total_batches,
                    successful=successful,
                    total=len(batch_descriptions),
                ),
            )

            if batch_num % cleanup_every_n_batches == 0:  # Cleanup every N batches
                cleanup_memory()
                log_memory_usage(f"Cleanup after batch {batch_num}")
                await asyncio.sleep(sleep_between_images)
            else:
                await asyncio.sleep(sleep_between_batches)

        return descriptions

    async def _generate_description_with_retries(
        self,
        image: Image.Image,
        image_id: str,
        max_retries: int,
        retry_delay_seconds: int = 1,
    ) -> str:
        """
        Generate description for a single image with retry logic.

        Args:
            image: PIL image to describe
            image_id: ID for logging purposes
            max_retries: Number of retry attempts
            retry_delay_seconds: Base delay between retries in seconds

        Returns:
            Description string or fallback description if all attempts fail

        """
        for attempt in range(max_retries):
            try:
                description = await self.generation_model.generate(
                    self.image_description_prompt,
                    context=[{"type": "image", "image": image}],
                )

            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                print(f"{image_id} attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # backoff: 1s, 2s, 3s
                    delay = attempt + retry_delay_seconds
                    print(f"Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)

            else:
                return description

        # All attempts failed
        print(f"All {max_retries} attempts failed for {image_id}")
        return self.fallback_description
