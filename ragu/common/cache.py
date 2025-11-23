import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from ragu.common.global_parameters import DEFAULT_FILENAMES, Settings
from ragu.common.logger import logger
from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(frozen=True, slots=True)
class CacheResult:
    response: Any
    cache_id: str
    source: Literal["cache", "live"]


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
    ) -> None:
        self.fn = fn
        self.key_builder = key_builder
        self.flush_every_n_writes = max(1, flush_every_n_writes)

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
