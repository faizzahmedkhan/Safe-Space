"""Training dataset over the frame cache, shared by train.py and the notebook.

Each item is one random ready 16-frame window of one cached video (a new
window every epoch), optionally horizontally flipped, preprocessed with
inference_core.preprocess_batch (/255 only, (C, T, H, W)), i.e. the same
tensor live mode builds.  No video decoding or frame_prep happens here.

Randomness comes from the `random` module, so seed it once in the main
process and pass seed_worker as the DataLoader's worker_init_fn (plus a
seeded torch.Generator) for reproducible, decorrelated workers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from frame_cache import CachedVideo, FrameCache
from inference_core import preprocess_batch


@dataclass(frozen=True)
class CacheItem:
    key: str  # FrameCache key
    label: int  # 0 = Normal, 1 = Dangerous
    path: str  # source video (for reports)
    cls: str = ""  # raw class name (Normal / Violence / Weaponized / safe / unsafe)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: torch already gives each worker
    base_seed + worker_id (base_seed drawn from the loader's generator, so it
    changes every epoch but is reproducible); propagate it to `random` and
    numpy, which the dataset uses."""
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)


class CachedClipDataset(Dataset):
    def __init__(
        self,
        cache: FrameCache,
        items: Sequence[CacheItem],
        windows_per_video: int = 1,
        hflip: bool = False,
    ):
        self.cache = cache
        self.items = list(items)
        self.windows_per_video = max(1, int(windows_per_video))
        self.hflip = bool(hflip)

    # Attributes the notebook's diagnostic cells read.
    @property
    def video_paths(self) -> List[str]:
        return [it.path for it in self.items]

    @property
    def labels(self) -> List[int]:
        return [it.label for it in self.items]

    @property
    def classes(self) -> List[str]:
        return [it.cls for it in self.items]

    def __len__(self) -> int:
        return len(self.items) * self.windows_per_video

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx // self.windows_per_video]
        video = self.cache.load(item.key)
        clip = video.window(video.random_window(random)).frames  # (T, H, W, C)
        if self.hflip and random.random() < 0.5:
            clip = clip[:, :, ::-1, :]  # same flip for every frame of the clip
        x = preprocess_batch(clip[None])[0]
        return x, torch.tensor(item.label, dtype=torch.long)


def iter_eval_videos(
    cache: FrameCache, items: Sequence[CacheItem]
) -> Iterator[Tuple[str, int, str, CachedVideo]]:
    """Lazy (path, label, class, CachedVideo) stream for clip_eval.evaluate_model."""
    for it in items:
        yield it.path, it.label, it.cls, cache.load(it.key)
