import numpy as np

from typing import List
from razdel import sentenize
from tqdm import tqdm

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.utils.ragu_utils import compute_mdhash_id


class SimpleChunker(BaseChunker):
    """
    A simple chunker that splits text into fixed-size overlapping chunks.
    """

    def __init__(self, max_chunk_size: int, overlap: int = 0) -> None:
        """
        Initializes the simple chunker.

        :param max_chunk_size: Maximum chunk size in characters.
        :param overlap: Number of overlapping characters between consecutive chunks.
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split(self, documents: str | List[str]) -> List[Chunk]:
        """
        Splits documents into fixed-size overlapping chunks.

        :param documents: List of input documents.
        :return: List of text chunks.
        """

        if isinstance(documents, str):
            documents = [documents]

        all_chunks: list[Chunk] = []
        for doc_idx, document in enumerate(tqdm(documents, desc="Splitting documents")):
            sentences = [chunk.text for chunk in sentenize(document)]
            current_chunk = ""
            chunks: list[str] = []

            # TODO: add doc_id from Document class, do not calculate it here
            doc_id = compute_mdhash_id(document)

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    if chunks and self.overlap > 0:
                        current_chunk = (
                            chunks[-1][-self.overlap:] + " " + sentence + " "
                        )
                    else:
                        current_chunk = sentence + " "

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            for i, text in enumerate(chunks):
                all_chunks.append(
                    Chunk(content=text, chunk_order_idx=i, doc_id=doc_id)
                )

        return all_chunks


class SemanticTextChunker(BaseChunker):
    """
    A semantic chunker that splits text into coherent chunks
    based on sentence boundaries and semantic similarity.
    """

    def __init__(
        self,
        model_name: str,
        max_chunk_size: int,
        device: str = "cuda:0",
    ) -> None:
        """
        Initializes the semantic chunker.

        :param model_name: Name of the sentence-transformer model.
        :param max_chunk_size: Maximum number of tokens per chunk.
        :param device: Device for model execution.
        """
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Please install 'sentence_transformers' for SemanticTextChunker:\n"
                "    pip install sentence_transformers"
            )

        self.model = SentenceTransformer(model_name).to(device)
        self.model.eval()
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _sentenize(text: str) -> List[str]:
        """
        Splits text into sentences.
        """
        return [chunk.text for chunk in sentenize(text)]

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Computes embeddings for a list of sentences.
        """
        embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        return embeddings.cpu().numpy()

    @staticmethod
    def _compute_similarities(embeddings: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarities between consecutive embeddings.
        """
        if len(embeddings) < 2:
            return np.array([])
        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        return sims

    def _merge_sentences(self, sentences: List[str], similarities: np.ndarray) -> List[str]:
        """
        Recursively merges semantically similar sentences
        until chunk length fits within max_chunk_size.
        """
        if len(sentences) < 2:
            return sentences

        n_tokens = len(self.model.tokenize([" ".join(sentences)])["input_ids"])
        if n_tokens <= self.max_chunk_size:
            return [" ".join(sentences)]

        min_idx = int(np.argmin(similarities))
        return (
            self._merge_sentences(sentences[: min_idx + 1], similarities[:min_idx])
            + self._merge_sentences(sentences[min_idx + 1:], similarities[min_idx + 1:])
        )

    def split(self, documents: str | List[str]) -> List[Chunk]:
        """
        Splits input documents into semantically coherent chunks.

        :param documents: A single document or a list of documents.
        :return: List of `Chunk` objects with content and IDs.
        """
        if isinstance(documents, str):
            documents = [documents]

        all_chunks: list[Chunk] = []

        for doc_idx, document in enumerate(tqdm(documents, desc="Splitting documents semantically")):
            sentences = self._sentenize(document)
            if not sentences:
                continue

            embeddings = self._embed(sentences)
            similarities = self._compute_similarities(embeddings)
            merged_chunks = self._merge_sentences(sentences, similarities)

            # TODO: add doc_id from Document class, do not calculate it here
            doc_id = compute_mdhash_id(document)
            for i, text in enumerate(merged_chunks):
                all_chunks.append(
                    Chunk(content=text, chunk_order_idx=i, doc_id=doc_id)
                )

        return all_chunks


class SmartSemanticChunker(BaseChunker):
    """
    A smart semantic chunker using a reranker-based algorithm
    (e.g., BAAI/bge-reranker-v2-m3) for improved coherence.
    """

    def __init__(
        self,
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
        newline_as_separator: bool = False,
        device: str = "cuda:0",
        max_chunk_length: int = 250,
        minibatch_size: int = 8,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the smart semantic chunker.

        :param reranker_name: Model name used for semantic reranking.
        :param newline_as_separator: Whether to split text by newlines.
        :param device: Device for computation.
        :param max_chunk_length: Maximum number of tokens per chunk.
        :param minibatch_size: Batch size for embedding computation.
        :param verbose: Whether to display detailed progress logs.
        """
        super().__init__()
        try:
            from ragu.chunker.smart_chunker import SmartChunker
        except ImportError:
            raise ImportError(
                "The 'smart_chunker' module is required. "
                "Install it with: pip install smart_chunker"
            )

        self.chunker = SmartChunker(
            reranker_name=reranker_name,
            newline_as_separator=newline_as_separator,
            device=device,
            max_chunk_length=max_chunk_length,
            minibatch_size=minibatch_size,
            verbose=verbose,
        )

    def split(self, documents: str | List[str]) -> List[Chunk]:
        """
        Splits documents using the SmartChunker model.

        :param documents: A single document or a list of documents.
        :return: List of `Chunk` objects with per-document indexing.
        """
        if isinstance(documents, str):
            documents = [documents]

        all_chunks: list[Chunk] = []

        for doc_idx, document in enumerate(tqdm(documents, desc="Splitting documents")):
            # TODO: add doc_id from Document class, do not calculate it here
            doc_id = compute_mdhash_id(document)
            parts = self.chunker.split_into_chunks(source_text=document)

            for i, text in enumerate(parts):
                all_chunks.append(Chunk(content=text, chunk_order_idx=i, doc_id=doc_id))

        return all_chunks
