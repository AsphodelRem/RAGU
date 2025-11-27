from ragu.embedder.base_embedder import BaseEmbedder
from ragu.common.batch_generator import BatchGenerator

import numpy as np


class STEmbedder(BaseEmbedder):
    """
    Embedder that uses Sentence Transformers to compute text embeddings.
    """

    def __init__(self, model_name_or_path: str, dim: int=None, *args, **kwargs):
        """
        Initializes the STEmbedder with a specified model.

        :param model_name_or_path: Path or name of the Sentence Transformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "RAGU needs SentenceTransformer to use this class. Please install it using `pip install sentence-transformers`."
            )
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path, **kwargs)
        self.dim = self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: str | list[str], batch_size: int=16):
        """
        Computes embeddings for a string or a list of strings.

        :param texts: Input text(s) to embed.
        :param batch_size: Batch size.
        :return: Embeddings for the input text(s).
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_generator = BatchGenerator(texts, batch_size=batch_size)
        embeddings_list = [
            await self.model.encode(batch, show_progress_bar=False)
            for batch in batch_generator.get_batches()
        ]

        return np.concatenate(embeddings_list)
