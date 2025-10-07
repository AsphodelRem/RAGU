from typing import List

import numpy as np
import pandas as pd
from razdel import sentenize
from tqdm import tqdm

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.utils.ragu_utils import compute_mdhash_id


class SimpleChunker(BaseChunker):
    """
    A simple chunker that splits text into fixed-size overlapping chunks.
    """

    def __init__(self, max_chunk_size: int, overlap: int=0) -> None:
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

        chunks: list[str] = []
        for document in documents:
            sentences = [chunk.text for chunk in sentenize(document)]
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    if chunks and self.overlap > 0:
                        current_chunk = chunks[-1][-self.overlap:] + " " + sentence + " "
                    else:
                        current_chunk = sentence + " "

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        return [Chunk(content=text, chunk_order_idx=0, doc_id="") for text in chunks]


class SemanticTextChunker(BaseChunker):
    """
    A semantic chunker that splits text based on sentence boundaries and semantic similarity.
    """

    def __init__(self, model_name: str, max_chunk_size: int, device='cuda:0',) -> None:
        """
        Initializes the semantic chunker.

        :param model_name: Name of the sentence transformer model.
        :param max_chunk_size: Maximum chunk size in tokens.
        """
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence_transformers' packages are required for the semantic chunker. "
                "Please install it with 'pip install sentence_transformers'."
            )
        self.model = SentenceTransformer(model_name).to(device)
        self.model.eval()
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def split_text_by_chunks(text: str) -> List[str]:
        """
        Splits text into sentence-based chunks using Razdel.

        :param text: Input text.
        :return: List of sentence-based chunks.
        """
        return [chunk.text for chunk in sentenize(text)]

    def calculate_document_embedding(self, document: str) -> np.ndarray:
        """
        Computes the embedding of a given document using the sentence transformer model.

        :param document: Input document.
        :return: Document embedding as a NumPy array.
        """
        embeddings = self.model.encode(document, convert_to_tensor=True, show_progress_bar=False)
        return embeddings.cpu().numpy()

    def compute_similarities(self, chunks: List[str]) -> np.ndarray:
        """
        Computes pairwise cosine similarities between consecutive chunks.

        :param chunks: List of sentence-based chunks.
        :return: Similarity scores as a NumPy array.
        """
        if len(chunks) < 2:
            return np.array([])

        embeddings = np.vstack(
            [self.calculate_document_embedding(chunk) for chunk in chunks]
        )
        similarities = np.array([
            np.dot(embeddings[i], embeddings[i + 1])
            for i in range(len(chunks) - 1)
        ])
        return similarities

    def join_chunks_by_semantics(self, sentences: List[str], similarities: np.ndarray) -> List[str]:
        """
        Merges chunks based on semantic similarity, ensuring chunk sizes do not exceed the limit.

        :param sentences: List of sentence-based chunks.
        :param similarities: Pairwise similarity scores between chunks.
        :return: List of semantically merged text chunks.
        """
        if len(sentences) < 2:
            return sentences

        n_tokens = len(self.model.tokenize([" ".join(sentences)])["input_ids"])
        if n_tokens <= self.max_chunk_size:
            return [" ".join(sentences)]

        min_similarity_idx = np.argmin(similarities)
        return (
                self.join_chunks_by_semantics(
                    sentences[: min_similarity_idx + 1],
                    similarities[:min_similarity_idx]
                )
                + self.join_chunks_by_semantics(
            sentences[min_similarity_idx + 1:],
            similarities[min_similarity_idx + 1:]
            )
        )

    def split(self, documents: str | List[str]) -> pd.DataFrame:
        """
        Splits text into semantically coherent chunks.

        :param documents: List of input documents.
        :return: List of semantically split text chunks.
        """
        if isinstance(documents, str):
            documents = [documents]

        all_chunks = []
        for document in tqdm(documents, desc="Index creation: splitting documents"):
            sentences = self.split_text_by_chunks(document)
            similarities = self.compute_similarities(sentences)
            chunks = self.join_chunks_by_semantics(sentences, similarities)
            all_chunks.extend(chunks)

        chunk_id = [compute_mdhash_id(chunk) for chunk in all_chunks]

        return pd.DataFrame({"chunk": all_chunks, "chunk_id": chunk_id})


class SmartSemanticChunker(BaseChunker):
    def __init__(
            self,
            reranker_name='BAAI/bge-reranker-v2-m3',
            newline_as_separator=False,
            device='cuda:0',
            max_chunk_length=250,
            minibatch_size=8,
            verbose=False
    ):
        super().__init__()
        try:
            from ragu.chunker.smart_chunker import SmartChunker
        except ImportError:
            raise ImportError(
                "The 'smart_chunker' package is required for the smart semantic chunker. "
                "Please install it with 'pip install smart_chunker'."
            )

        self.chunker = SmartChunker(
            reranker_name=reranker_name,
            newline_as_separator=newline_as_separator,
            device=device,
            max_chunk_length=max_chunk_length,
            minibatch_size=minibatch_size,
            verbose=verbose
        )

    def split(self, documents: str | List[str]) -> List[Chunk]:
        if isinstance(documents, str):
            documents = [documents]

        chunks: list[str] = []
        for document in tqdm(documents, desc="Index creation: splitting documents"):
            chunks.extend(self.chunker.split_into_chunks(source_text=document))

        return [Chunk(content=text, chunk_order_idx=0, doc_id="") for text in chunks]