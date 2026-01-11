import asyncio
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel

from ragu.common.global_parameters import DEFAULT_FILENAMES, Settings
from ragu.common.logger import logger
from ragu.utils.ragu_utils import compute_mdhash_id


# TODO: Remove
@dataclass(frozen=True, slots=True)
class CacheResult:
    response: Any
    cache_id: str
    source: Literal["cache", "live"]

# TODO: Remove
class CachedLLMCall:
    """
    Wrap an LLM callable with a simple disk cache.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any] | Any],
        cache_path: str | Path | None = None,
        *,
        key_builder: Callable[[str], str] | None = None,
        flush_every_n_writes: int = 50,
        response_model: Type[BaseModel] | None = None,
    ) -> None:
        self.fn = fn
        self.key_builder = key_builder
        self.flush_every_n_writes = max(1, flush_every_n_writes)
        self.response_model = response_model

        self._mem_cache: Dict[str, Dict[str, Any]] = {}
        self._pending_disk_writes = 0

        if cache_path is None:
            Settings.init_storage_folder()
            default_name = DEFAULT_FILENAMES["llm_cache_file_name"]
            cache_path = Path(Settings.storage_folder) / default_name

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path
        self._load_json_cache(cache_path)

    def peek(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        cache_id: str | None = None,
    ) -> CacheResult | None:
        """
        Return a cached value if present without recording a miss or waiting on limiters.
        """
        cache_id = cache_id or self._build_cache_id(prompt, system_prompt)
        cached = self._mem_cache.get(cache_id)
        if cached is None:
            return None
        response = cached.get("response") if isinstance(cached, dict) else cached

        # Reconstruct BaseModel if response_model is specified
        if self.response_model is not None and isinstance(response, dict):
            try:
                response = self.response_model.model_validate(response)
            except Exception as e:
                logger.warning(f"Failed to reconstruct BaseModel from cache: {e}")
                return None

        return CacheResult(response=response, cache_id=cache_id, source="cache")

    async def __call__(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        cache_id: str | None = None,
        fn_kwargs: Dict[str, Any] | None = None,
    ) -> CacheResult:
        combined_prompt = self._combine_prompts(system_prompt, prompt)
        cache_id = cache_id or self._build_cache_id(prompt, system_prompt)
        fn_kwargs = fn_kwargs or {}

        cached = self._mem_cache.get(cache_id)
        if cached is not None:
            response = cached.get("response") if isinstance(cached, dict) else cached
            return CacheResult(response=response, cache_id=cache_id, source="cache")

        result = self.fn(prompt, **fn_kwargs)
        if hasattr(result, "__await__"):
            result = await result

        payload = {
            "input": combined_prompt,
            "response": self._extract_content(result),
            "tokens": self._extract_tokens(result),
            "model": self._extract_model(result),
            "time": self._extract_time(result),
        }

        self._queue_json_write(cache_id, payload)

        return CacheResult(response=result, cache_id=cache_id, source="live")

    def close(self) -> None:
        """
        Flush any buffered cache writes to disk.
        """
        self._flush_json_cache()

    def _load_json_cache(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-16") as f:
                cache = json.load(f)
                if not isinstance(cache, dict):
                    return

                self._mem_cache = cache

        except Exception as e:
            logger.warning(f"Failed to load JSONL cache at {path}: {e}")

    def _queue_json_write(self, cache_id: str, record: Dict[str, Any]) -> None:
        self._mem_cache[cache_id] = record
        self._pending_disk_writes += 1
        self._maybe_flush_to_disk()

    def _flush_json_cache(self) -> None:
        if not self._mem_cache:
            self._pending_disk_writes = 0
            return
        try:
            tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-16") as f:
                json.dump(self._mem_cache, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self._cache_path)
        except Exception as e:
            logger.warning(f"Failed to write cache record to {self._cache_path}: {e}")
            return
        self._pending_disk_writes = 0

    def _maybe_flush_to_disk(self) -> None:
        if self._pending_disk_writes < self.flush_every_n_writes:
            return
        self._flush_json_cache()

    @staticmethod
    def _to_serializable(resp: Any) -> Any:
        """
        Convert a response to something JSON-serializable. Prefers model_dump when available.
        """
        if resp is None:
            return None

        if hasattr(resp, "model_dump") and callable(resp.model_dump):
            try:
                return resp.model_dump()
            except Exception:
                pass

        if isinstance(resp, (str, int, float, bool, list, dict)):
            try:
                json.dumps(resp)
                return resp
            except TypeError:
                pass

        return repr(resp)

    @staticmethod
    def _extract_tokens(resp: Any) -> Optional[Dict[str, int]]:
        """
        Try to extract token usage information if present.
        """
        usage = None
        if resp is None:
            return None
        if isinstance(resp, dict):
            usage = resp.get("usage")
        else:
            usage = getattr(resp, "usage", None)

        if usage is None:
            return None

        try:
            if isinstance(usage, dict):
                return {
                    k: int(v)
                    for k, v in usage.items()
                    if isinstance(v, (int, float))
                }
            total = getattr(usage, "total_tokens", None)
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            result: Dict[str, int] = {}
            if total is not None:
                result["total_tokens"] = int(total)
            if prompt is not None:
                result["prompt_tokens"] = int(prompt)
            if completion is not None:
                result["completion_tokens"] = int(completion)
            return result or None
        except Exception:
            return None

    @staticmethod
    def _extract_model(resp: Any) -> Optional[str]:
        """
        Try to extract model name from response metadata.
        """
        if resp is None:
            return None
        model = getattr(resp, "model", None)
        if model:
            return str(model)
        if isinstance(resp, dict):
            model = resp.get("model")
            if model:
                return str(model)
        return None

    @staticmethod
    def _extract_time(resp: Any) -> str:
        """
        Extract a timestamp from response metadata when available, else current UTC ISO.
        """
        created = None
        if isinstance(resp, dict):
            created = resp.get("created")
        else:
            created = getattr(resp, "created", None)

        if created is not None:
            try:
                # OpenAI returns seconds since epoch
                ts = float(created)
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            except Exception:
                pass

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _extract_content(resp: Any) -> str:
        """
        Extract only the text content field from an OpenAI-compatible response.
        """
        if resp is None:
            return ""

        if isinstance(resp, str):
            return resp.strip()

        if isinstance(resp, dict):
            try:
                choices = resp.get("choices") or []
                first = choices[0] if choices else {}
                message = first.get("message") or {}
                content = message.get("content") or ""
                return str(content).strip()
            except Exception:
                return ""

        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass

        return ""

    @staticmethod
    def _combine_prompts(system_prompt: str | None, user_prompt: str) -> str:
        """
        Build a combined prompt string to be used as cache key/input.
        """
        if system_prompt:
            return f"[system]: {system_prompt}\n[user]: {user_prompt}"
        return user_prompt

    def _build_cache_id(self, prompt: str, system_prompt: str | None) -> str:
        combined_prompt = self._combine_prompts(system_prompt, prompt)
        cache_key = self.key_builder(combined_prompt) if self.key_builder else combined_prompt
        return compute_mdhash_id(cache_key, prefix="cache-")


@dataclass(frozen=True, slots=True)
class PendingRequest:
    """
    Represents a request pending generation (not found in cache).
    """
    index: int
    prompt: str
    cache_key: str


def make_llm_cache_key(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    schema: Optional[Type[BaseModel]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a deterministic cache key from LLM request parameters.

    :param prompt: The user prompt.
    :param system_prompt: Optional system prompt.
    :param model_name: Model name used for generation.
    :param schema: Optional Pydantic schema class.
    :param kwargs: Additional API parameters.
    :return: A unique cache key string.
    """
    key_parts = []

    if system_prompt:
        key_parts.append(f"[system]: {system_prompt}")
    key_parts.append(f"[user]: {prompt}")

    if model_name:
        key_parts.append(f"[model]: {model_name}")

    if schema is not None:
        schema_name = schema.__name__
        field_names = sorted(schema.model_fields.keys())
        key_parts.append(f"[schema]: {schema_name}({','.join(field_names)})")

    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
        key_parts.append(f"[kwargs]: {kwargs_str}")

    combined = "\n".join(key_parts)
    return compute_mdhash_id(combined, prefix="llm-cache-")


class TextCache:
    """
    Key-value cache for LLM responses with automatic BaseModel serialization.

    Supports both string and Pydantic BaseModel responses. When a schema is provided,
    automatically serializes/deserializes BaseModel instances.
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        flush_every_n_writes: int = 10,
    ) -> None:
        """
        Initialize async LLM cache.

        :param cache_path: Path to cache file. Defaults to storage_folder/llm_cache.json
        :param flush_every_n_writes: Write to disk after N cache updates.
        """
        self.flush_every_n_writes = max(1, flush_every_n_writes)

        self._mem_cache: Dict[str, Any] = {}
        self._pending_disk_writes = 0
        self._write_lock = asyncio.Lock()

        if cache_path is None:
            Settings.init_storage_folder()
            default_name = DEFAULT_FILENAMES["llm_cache_file_name"]
            cache_path = Path(Settings.storage_folder) / default_name

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        """
        Load cache from disk into memory.
        """
        if not self._cache_path.exists():
            return

        try:
            with self._cache_path.open("r", encoding="utf-16") as f:
                cache = json.load(f)
                if isinstance(cache, dict):
                    self._mem_cache = cache
        except Exception as e:
            logger.warning(f"Failed to load cache from {self._cache_path}: {e}")

    async def flush_cache(self) -> None:
        """
        Flush cache to disk.
        """
        if not self._mem_cache:
            self._pending_disk_writes = 0
            return

        async with self._write_lock:
            try:
                tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_cache_file,
                    tmp_path,
                )

                tmp_path.replace(self._cache_path)
                self._pending_disk_writes = 0
            except Exception as e:
                logger.warning(f"Failed to flush cache to {self._cache_path}: {e}")

    def _write_cache_file(self, path: Path) -> None:
        """
        Write cache to file.
        """
        with path.open("w", encoding="utf-16") as f:
            json.dump(self._mem_cache, f, ensure_ascii=False, indent=2)

    async def get(
        self,
        key: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Optional[Any]:
        """
        Retrieve a value from cache.

        :param key: Cache key.
        :param schema: Optional Pydantic schema to reconstruct BaseModel.
        :return: Cached value (string or BaseModel), or None if not found.
        """
        cached = self._mem_cache.get(key)
        if cached is None:
            return None

        # Extract response from payload if it's a structured cache entry
        if isinstance(cached, dict) and "response" in cached:
            response = cached["response"]
        else:
            # Backward compatibility: old cache entries without payload structure
            response = cached

        # If schema provided and response is a dict, reconstruct BaseModel
        if schema is not None and isinstance(response, dict):
            try:
                return schema.model_validate(response)
            except Exception as e:
                logger.warning(f"Failed to reconstruct {schema.__name__} from cache: {e}")
                return None

        return response

    async def set(
        self,
        key: str,
        value: Any,
        *,
        input_instruction: Optional[str] = None,
    ) -> None:
        """
        Store a value in cache.

        :param key: Cache key.
        :param value: Value to cache (string or BaseModel).
        :param input_instruction: Optional input/prompt that produced this value.
        """
        # Serialize BaseModel to dict for storage
        if isinstance(value, BaseModel):
            cached_value = value.model_dump()
        else:
            cached_value = value

        payload: Dict[str, Any] = {"response": cached_value}
        if input_instruction is not None:
            payload["input"] = input_instruction
        payload["time"] = datetime.now(timezone.utc).isoformat()

        self._mem_cache[key] = payload
        self._pending_disk_writes += 1

        if self._pending_disk_writes >= self.flush_every_n_writes:
            await self.flush_cache()

    async def close(self) -> None:
        """
        Flush any pending writes and close cache.
        """
        if self._pending_disk_writes > 0:
            await self.flush_cache()


class EmbeddingCache:
    """
    Async cache specifically for embeddings (lists of floats).

    Uses pickle serialization for efficient storage of numeric data.
    Separate from LLM cache to allow independent management and different storage strategies.
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        flush_every_n_writes: int = 50,
    ) -> None:
        """
        Initialize async embedding cache.

        :param cache_path: Path to cache file. Defaults to storage_folder/embedding_cache.pkl
        :param flush_every_n_writes: Write to disk after N cache updates.
        """
        self.flush_every_n_writes = max(1, flush_every_n_writes)

        self._mem_cache: Dict[str, List[float]] = {}
        self._pending_disk_writes = 0
        self._write_lock = asyncio.Lock()

        if cache_path is None:
            Settings.init_storage_folder()
            default_name = DEFAULT_FILENAMES["embedding_cache_file_name"]
            cache_path = Path(Settings.storage_folder) / default_name

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        """
        Load cache from disk into memory.
        """
        if not self._cache_path.exists():
            return

        try:
            with self._cache_path.open("rb") as f:
                cache = pickle.load(f)
                if isinstance(cache, dict):
                    self._mem_cache = cache
        except Exception as e:
            logger.warning(f"Failed to load embedding cache from {self._cache_path}: {e}")

    async def _flush_cache(self) -> None:
        """
        Flush cache to disk.
        """
        if not self._mem_cache:
            self._pending_disk_writes = 0
            return

        async with self._write_lock:
            try:
                tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_cache_file,
                    tmp_path,
                )

                tmp_path.replace(self._cache_path)
                self._pending_disk_writes = 0
            except Exception as e:
                logger.warning(f"Failed to flush embedding cache to {self._cache_path}: {e}")

    def _write_cache_file(self, path: Path) -> None:
        """Write cache to file (runs in thread pool)."""
        with path.open("wb") as f:
            pickle.dump(self._mem_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def get(self, key: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from cache.

        :param key: Cache key.
        :return: Cached embedding (list of floats), or None if not found.
        """
        return self._mem_cache.get(key)

    async def set(self, key: str, embedding: List[float]) -> None:
        """
        Store an embedding in cache.

        :param key: Cache key.
        :param embedding: Embedding vector to cache (list of floats).
        """
        self._mem_cache[key] = embedding
        self._pending_disk_writes += 1

        # Flush to disk if threshold reached
        if self._pending_disk_writes >= self.flush_every_n_writes:
            await self._flush_cache()

    async def close(self) -> None:
        """Flush any pending writes and close cache."""
        if self._pending_disk_writes > 0:
            await self._flush_cache()
