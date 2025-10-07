from ragu.common import logger


class TokenTruncation:
    """
    A universal text truncator class that limits input text
    to a maximum number of tokens using either `tiktoken` or HuggingFace's `AutoTokenizer`.

    :param model_id: The model name or identifier (e.g., "gpt-4o", "bert-base-uncased")
    :param tokenizer_type: Tokenizer type - either "tiktoken" or "local"
    :param max_tokens: Maximum number of tokens to retain
    :param safe_decode: Whether to use safe UTF-8 decoding for truncated output
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        tokenizer_type: str = "tiktoken",
        max_tokens: int = 30000,
        safe_decode: bool = True,
    ):
        self.model_id = model_id
        self.tokenizer_type = tokenizer_type
        self.max_tokens = max_tokens
        self.safe_decode = safe_decode

        if tokenizer_type == "tiktoken":
            try:
                import tiktoken
                self.encoder = tiktoken.encoding_for_model(model_id)
            except Exception as e:
                raise ValueError(f"[tiktoken] Failed to initialize tokenizer for '{model_id}': {e}")
        elif tokenizer_type == "local":
            try:
                from transformers import AutoTokenizer
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception as e:
                raise ValueError(f"[transformers] Failed to load tokenizer '{model_id}': {e}")
        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

    def _truncate_with_tiktoken(self, text: str) -> tuple[str, int, int]:
        """
        Truncate text using `tiktoken` tokenizer.

        :param text: Input text
        :return: (Truncated text, original token length, truncated token length)
        """
        tokens = self.encoder.encode(text)
        original_len = len(tokens)
        truncated_tokens = tokens[:self.max_tokens]
        decoded = self.encoder.decode(truncated_tokens)
        if self.safe_decode:
            decoded = decoded.encode("utf-8", errors="replace").decode("utf-8")
        return decoded, original_len, len(truncated_tokens)

    def _truncate_with_local_tokenizer(self, text: str) -> tuple[str, int, int]:
        """
        Truncate text using HuggingFace tokenizer.

        :param text: Input text
        :return: (Truncated text, original token length, truncated token length)
        """
        tokens = self.local_tokenizer.encode(text, add_special_tokens=False)
        original_len = len(tokens)
        truncated_tokens = tokens[:self.max_tokens]
        decoded = self.local_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return decoded, original_len, len(truncated_tokens)

    def __call__(self, text: str, return_stats: bool = False) -> str:
        """
        Truncate input text to `max_tokens` using selected tokenizer.

        :param text: Input text to truncate
        :param return_stats: If True, returns additional stats: original and truncated token counts
        :return: Truncated string or a tuple with statistics
        """
        if not text:
            return ""

        if self.tokenizer_type == "tiktoken":
            result = self._truncate_with_tiktoken(text)
        elif self.tokenizer_type == "local":
            result = self._truncate_with_local_tokenizer(text)
        else:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")

        truncated_text, original_len, truncated_len = result
        if return_stats:
            logger.info("Before truncation length: %s, after truncation length: %s", original_len, truncated_len)

        return truncated_text
