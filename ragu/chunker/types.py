from dataclasses import dataclass, field

from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Chunk:
    id: str=field(init=False)
    content: str
    chunk_order_idx: int
    doc_id: str
    num_tokens: int=None

    def __post_init__(self):
        self.id = compute_mdhash_id(content=self.content, prefix="chunk-")

