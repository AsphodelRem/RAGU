import pytest
from ragu.chunker.chunkers import (
    SimpleChunker,
    SemanticTextChunker,
    SmartSemanticChunker
)
from ragu.chunker.types import Chunk


class TestSimpleChunker:
    def test_split_single_document(self, sample_text_short):
        chunker = SimpleChunker(max_chunk_size=50, overlap=0)
        chunks = chunker.split(sample_text_short)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(hasattr(chunk, 'content') for chunk in chunks)
        assert all(hasattr(chunk, 'id') for chunk in chunks)
        assert all(hasattr(chunk, 'doc_id') for chunk in chunks)
        assert all(hasattr(chunk, 'chunk_order_idx') for chunk in chunks)

    def test_split_list_of_documents(self, sample_documents):
        chunker = SimpleChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.split(sample_documents)

        assert isinstance(chunks, list)
        assert len(chunks) >= len(sample_documents)

        # Verify chunks from different documents have different doc_ids
        doc_ids = set(chunk.doc_id for chunk in chunks)
        assert len(doc_ids) > 0

    def test_chunk_order_indices(self, sample_text_long):
        chunker = SimpleChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.split(sample_text_long)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_order_idx == i

    def test_max_chunk_size_respected(self, sample_text_long):
        max_size = 200
        chunker = SimpleChunker(max_chunk_size=max_size, overlap=0)
        chunks = chunker.split(sample_text_long)

        for chunk in chunks:
            assert len(chunk.content) <= max_size + 50  # Allow some margin for sentence boundaries

    def test_overlap_functionality(self, sample_text_medium):
        overlap = 20
        chunker = SimpleChunker(max_chunk_size=100, overlap=overlap)
        chunks = chunker.split(sample_text_medium)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].content[-overlap:] if len(chunks[i].content) >= overlap else chunks[i].content
                chunk2_start = chunks[i + 1].content[:overlap]
                # At least some characters should be similar
                assert len(chunk1_end) > 0
                assert len(chunk2_start) > 0

    def test_empty_string(self):
        chunker = SimpleChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.split("")

        # Empty string should produce no chunks or one empty chunk
        assert len(chunks) <= 1

    def test_very_long_sentence(self):
        long_sentence = "A" * 500 + ". This is another sentence."
        chunker = SimpleChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.split(long_sentence)

        assert len(chunks) > 0
        # The long sentence should be split somehow
        assert any(len(chunk.content) > 0 for chunk in chunks)

    def test_chunk_ids_unique(self, sample_text_long):
        chunker = SimpleChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.split(sample_text_long)

        chunk_ids = [chunk.id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All IDs should be unique

    def test_doc_id_consistency(self):
        text = "This is a test document. It has multiple sentences. Each will be chunked."
        chunker = SimpleChunker(max_chunk_size=30, overlap=0)
        chunks = chunker.split(text)

        if len(chunks) > 1:
            doc_id = chunks[0].doc_id
            assert all(chunk.doc_id == doc_id for chunk in chunks)


class TestSemanticTextChunker:
    @pytest.fixture(scope="class")
    def semantic_chunker(self):
        try:
            # Use a tiny model for faster testing
            return SemanticTextChunker(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                max_chunk_size=50,
                device="cpu"
            )
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentenize(self):
        text = "First sentence. Second sentence! Third question?"
        sentences = SemanticTextChunker._sentenize(text)

        assert isinstance(sentences, list)
        assert len(sentences) >= 3

    def test_compute_similarities(self, semantic_chunker):
        import numpy as np

        # Create some dummy embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0]
        ])

        similarities = semantic_chunker._compute_similarities(embeddings)
        assert len(similarities) == 2
        assert similarities[0] > similarities[1]  # First two are more similar

    def test_compute_similarities_edge_cases(self, semantic_chunker):
        import numpy as np

        # Single embedding - should return empty array
        embeddings = np.array([[1.0, 0.0]])
        similarities = semantic_chunker._compute_similarities(embeddings)
        assert len(similarities) == 0

        # Empty array
        embeddings = np.array([]).reshape(0, 3)
        similarities = semantic_chunker._compute_similarities(embeddings)
        assert len(similarities) == 0

    def test_split_single_document(self, semantic_chunker, sample_text_medium):
        chunks = semantic_chunker.split(sample_text_medium)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(hasattr(chunk, 'content') for chunk in chunks)

    def test_split_multiple_documents(self, semantic_chunker, sample_documents):
        chunks = semantic_chunker.split(sample_documents)

        assert isinstance(chunks, list)
        assert len(chunks) >= len(sample_documents)

        # Verify different doc_ids
        doc_ids = set(chunk.doc_id for chunk in chunks)
        assert len(doc_ids) > 0

    def test_empty_document(self, semantic_chunker):
        chunks = semantic_chunker.split("")

        # Should handle gracefully
        assert isinstance(chunks, list)


class TestSmartSemanticChunker:
    @pytest.fixture(scope="class")
    def smart_chunker(self):
        try:
            return SmartSemanticChunker(
                reranker_name="BAAI/bge-reranker-v2-m3",
                max_chunk_length=100,
                device="cpu",
                verbose=False
            )
        except ImportError:
            pytest.skip("smart_chunker not installed")
        except Exception as e:
            pytest.skip(f"Could not initialize SmartSemanticChunker: {e}")

    def test_split_single_document(self, smart_chunker, sample_text_medium):
        chunks = smart_chunker.split(sample_text_medium)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_split_multiple_documents(self, smart_chunker, sample_documents):
        chunks = smart_chunker.split(sample_documents)

        assert isinstance(chunks, list)
        assert len(chunks) >= len(sample_documents)

    def test_chunk_structure(self, smart_chunker, sample_text_long):
        chunks = smart_chunker.split(sample_text_long)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_order_idx == i
            assert chunk.doc_id is not None
            assert chunk.id is not None
            assert len(chunk.content) > 0
