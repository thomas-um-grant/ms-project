from PIL import Image

from retrieval_pipeline.models.embedding_models import ColQwen2Model
from retrieval_pipeline.utils import score_max_sim


class ColPaliAdapter:
    def __init__(self):
        self.model_name = "vidore/colqwen2-v1.0"

        self.embedding_model = ColQwen2Model()

        self.images = None
        self.image_embeddings = None
        self.query_embeddings = None

    async def index(self, images: list):
        self.images = [_scale_image(image) for image in images]
        self.image_embeddings = self.embedding_model.embed_images(self.images)

    async def retrieve(self, texts: list[str], top_k: int = 3):
        self.query_embeddings = self.embedding_model.embed_texts(texts)

        scores = score_max_sim(
            self.query_embeddings,
            self.image_embeddings,
        )

        retrieved_image_indexes = sorted(scores, key=scores.get, reverse=True)[:top_k]
        retrieved_images = [self.images[i] for i in retrieved_image_indexes]

        return retrieved_images


def _scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """Scale an image to a new height while maintaining the aspect ratio."""
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image
