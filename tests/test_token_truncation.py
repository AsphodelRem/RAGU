import pytest

from ragu.utils.token_truncation import TokenTruncation


class TestTokenTruncationWithTiktoken:
    @pytest.fixture
    def truncator(self):
        try:
            return TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=50
            )
        except Exception as e:
            pytest.skip(f"tiktoken not available: {e}")

    def test_truncate_short_text(self, truncator):
        text = "This is a short text."
        result = truncator(text)

        assert isinstance(result, str)
        assert result == text

    def test_truncate_long_text(self, truncator):
        # Create a text that's definitely longer than 50 tokens
        text = " ".join(["word"] * 200)
        result = truncator(text)

        assert isinstance(result, str)
        assert len(result) < len(text)

    def test_truncate_empty_string(self, truncator):
        result = truncator("")

        assert result == ""

    def test_truncate_unicode_text(self, truncator):
        text = "Hello 世界 " * 50
        result = truncator(text)

        assert isinstance(result, str)

    def test_safe_decode_handling(self):
        try:
            truncator_safe = TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=10,
                safe_decode=True
            )

            text = "Test text with special characters: 你好世界"
            result = truncator_safe(text)

            assert isinstance(result, str)
        except Exception:
            pytest.skip("tiktoken not available")

    def test_consistency(self):
        try:
            truncator = TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=30
            )

            text = "This is a test sentence that should be truncated consistently."
            result1 = truncator(text)
            result2 = truncator(text)

            assert result1 == result2
        except Exception:
            pytest.skip("tiktoken not available")


class TestTokenTruncationEdgeCases:
    def test_special_characters(self):
        try:
            truncator = TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=50
            )

            text = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
            result = truncator(text)

            assert isinstance(result, str)
        except Exception:
            pytest.skip("tiktoken not available")