from ragu.embedder.base_embedder import BaseEmbedder


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

    async def embed(self, texts: str | list[str]):
        """
        Computes embeddings for a string or a list of strings.

        :param texts: Input text(s) to embed.
        :return: Embeddings for the input text(s).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False)
