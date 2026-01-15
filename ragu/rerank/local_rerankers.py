from typing import List, Tuple

from ragu.common.batch_generator import BatchGenerator
from ragu.rerank.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Reranker that uses Sentence Transformers CrossEncoder to compute relevance scores.
    """

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        """
        Initializes the CrossEncoderReranker with a specified model.

        :param model_name_or_path: Path or name of the CrossEncoder model.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "RAGU needs SentenceTransformer to use this class. "
                "Please install local version of RAGU using `pip install graph_ragu[local]`"
                "or install sentence-transformers manually."
            )
        self.model = CrossEncoder(model_name_or_path, **kwargs)

    async def rerank(
            self,
            x: str,
            others: List[str],
            batch_size: int = 16,
            top_k: int | None = None
    ) -> List[Tuple[int, float]]:
        """
        Reranks documents based on relevance to the query.

        :param x: Query text.
        :param others: List of documents to rerank.
        :param batch_size: Batch size for inference.
        :param top_k: Number of top results to return. If None, returns all.
        :return: List of (index, score) tuples sorted by relevance descending.
        """
        if not others:
            return []

        pairs = [(x, doc) for doc in others]
        batch_generator = BatchGenerator(pairs, batch_size=batch_size)

        scores_list = []
        for batch in batch_generator.get_batches():
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            scores_list.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores))

        indexed_scores = [(i, score) for i, score in enumerate(scores_list)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores
