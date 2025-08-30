from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the embedder.
        """
        self.dim = None

    @abstractmethod
    def embed(self, texts: str | list[str]):
        """
        Computes embeddings for a list of text inputs.
        """
        ...

    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)


class STEmbedder(BaseEmbedder):
    """
    Embedder that uses Sentence Transformers to compute text embeddings.
    """

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        """
        Initializes the STEmbedder with a specified model.

        :param model_name_or_path: Path or name of the Sentence Transformer model.
        """
        super().__init__(*args, **kwargs)
        self.model = SentenceTransformer(model_name_or_path, **kwargs)
        self.dim = self.model.get_sentence_embedding_dimension()

        if self.dim is None:
            self.dim = len(self.model.encode(["asphodel"]))

    def embed(self, texts: str | list[str]):
        """
        Computes embeddings for a string or a list of strings.

        :param texts: Input text(s) to embed.
        :return: Embeddings for the input text(s).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False)
