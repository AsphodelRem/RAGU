import asyncio
from hashlib import md5
from pathlib import Path
from typing import Callable, Any
from typing import List

from aiolimiter import AsyncLimiter


class AsyncRunner:
    def __init__(
        self,
        semaphore: asyncio.Semaphore,
        rps_limiter: AsyncLimiter,
        rpm_limiter: AsyncLimiter,
        progress_bar,
    ):
        self.semaphore = semaphore
        self.rps_limiter = rps_limiter
        self.rpm_limiter = rpm_limiter
        self.progress_bar = progress_bar

    async def make_request(self, func: Callable[..., Any], **kwargs):
        async with self.semaphore:
            async with self.rps_limiter:
                async with self.rpm_limiter:
                    try:
                        return await func(**kwargs)
                    finally:
                        self.progress_bar.update(1)


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError()
        return current_loop

    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def read_text_from_files(directory: str, file_extensions=None) -> List[str]:
    texts = []
    directory = Path(directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and (file_extensions is None or file_path.suffix in file_extensions):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Cannot read file {file_path}: {e}")

    return texts

