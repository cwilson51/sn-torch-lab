"""Write a synthetic ImageNet-like dataset of JPEG files to disk.

Author: Collin Wilson at 2026-11-16, https://github.com/cwilson51

``torchvision.datasets.FakeData`` makes each image in memory, so a training loop that
uses it never opens a file or decodes a JPEG. This module writes an ``ImageFolder``
tree of real JPEG files instead, sized and compressed like ImageNet, so each sample
the DataLoader loads has to be opened, read and decoded from disk.

Run ``python -m torch_trace_lab.synthetic_data ROOT N_IMAGES`` to create one by hand.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_VERSION = 1
MARKER = ".synthetic-imagenet.json"
NUM_CLASSES = 100
JPEG_QUALITY = 90

# Common ImageNet train-set shapes (width, height). Most images are about 500 px on
# the long side; a few large ones make decode time vary as much as it does in practice.
COMMON_SHAPES = [(500, 375), (375, 500), (500, 333), (333, 500), (500, 500), (640, 480)]
LARGE_FRACTION = 0.1


def default_data_dir() -> Path:
    """Node-local ``$SLURM_TMPDIR`` inside a Slurm job, otherwise the user cache dir."""
    tmp = os.environ.get("SLURM_TMPDIR")
    if tmp:
        return Path(tmp) / "synthetic-imagenet"
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "torch-trace-lab" / "synthetic-imagenet"


def _image_shape(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < LARGE_FRACTION:
        width = int(rng.integers(800, 1600))
        return width, int(width * rng.choice([0.75, 0.667, 1.333]))
    return COMMON_SHAPES[rng.integers(len(COMMON_SHAPES))]


def _make_jpeg(path: Path, seed: int) -> None:
    """Write one JPEG with smooth structure plus fine noise.

    Pure noise would compress far worse than a photo; this mix gives file sizes of
    about 100-150 KB on average, close to ImageNet (about 110 KB).
    """
    rng = np.random.default_rng(seed)
    width, height = _image_shape(rng)
    coarse = rng.integers(0, 256, (height // 16 + 1, width // 16 + 1, 3), dtype=np.uint8)
    base = np.asarray(Image.fromarray(coarse).resize((width, height), Image.BICUBIC))
    noise = rng.normal(0, 14, (height, width, 3))
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, "JPEG", quality=JPEG_QUALITY)


def _image_paths(root: Path, n_images: int) -> list[Path]:
    return [root / f"class_{i % NUM_CLASSES:03d}" / f"img_{i:06d}.JPEG" for i in range(n_images)]


def _read_marker(root: Path) -> dict | None:
    try:
        return json.loads((root / MARKER).read_text())
    except (OSError, ValueError):
        return None


def ensure_dataset(root: Path, n_images: int, workers: int | None = None) -> Path:
    """Create the dataset under ``root`` unless one with at least ``n_images`` exists.

    Returns ``root``. Images are written in parallel threads (PIL releases the GIL
    while encoding).
    """
    root = Path(root)
    marker = _read_marker(root)
    if marker and marker.get("version") == FORMAT_VERSION and marker.get("n_images", 0) >= n_images:
        return root

    paths = _image_paths(root, n_images)
    # Only folders that get images: ImageFolder rejects empty class folders.
    for folder in {p.parent for p in paths}:
        folder.mkdir(parents=True, exist_ok=True)
    workers = workers or min(os.cpu_count() or 1, 8)
    print(f"Writing {n_images} synthetic JPEGs to {root} ...", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_make_jpeg, paths, range(n_images)))

    (root / MARKER).write_text(json.dumps({"version": FORMAT_VERSION, "n_images": n_images}))
    return root


def evict_from_page_cache(root: Path) -> bool:
    """Ask the kernel to drop the dataset's files from the page cache.

    Without this, files that were just written or read by an earlier run are served
    from RAM, which hides the disk reads a first epoch would do. Needs
    ``os.posix_fadvise`` (Linux); returns False where it is unavailable.
    """
    if not hasattr(os, "posix_fadvise"):
        return False
    for path in Path(root).rglob("*.JPEG"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return True


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or not argv[1].isdigit():
        print("usage: python -m torch_trace_lab.synthetic_data ROOT N_IMAGES", file=sys.stderr)
        return 2
    ensure_dataset(Path(argv[0]), int(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
