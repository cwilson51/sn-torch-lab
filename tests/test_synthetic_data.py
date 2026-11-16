from PIL import Image
from torchvision.datasets import ImageFolder

from torch_trace_lab import synthetic_data as sd


def test_ensure_dataset_writes_imagefolder_of_jpegs(tmp_path):
    root = sd.ensure_dataset(tmp_path / "data", n_images=12, workers=2)

    ds = ImageFolder(root)
    assert len(ds) == 12
    assert len(ds.classes) == 12  # one image per class until NUM_CLASSES is reached
    img = Image.open(ds.samples[0][0])
    assert img.format == "JPEG"
    assert min(img.size) >= 333


def test_ensure_dataset_reuses_existing(tmp_path, monkeypatch):
    root = sd.ensure_dataset(tmp_path, n_images=6, workers=1)
    monkeypatch.setattr(sd, "_make_jpeg", lambda *a: (_ for _ in ()).throw(AssertionError))
    assert sd.ensure_dataset(root, n_images=4) == root


def test_ensure_dataset_grows_when_too_small(tmp_path):
    sd.ensure_dataset(tmp_path, n_images=3, workers=1)
    sd.ensure_dataset(tmp_path, n_images=8, workers=1)
    assert len(ImageFolder(tmp_path)) == 8


def test_default_data_dir_prefers_slurm_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    assert sd.default_data_dir() == tmp_path / "synthetic-imagenet"


def test_default_data_dir_outside_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert sd.default_data_dir() == tmp_path / "torch-trace-lab" / "synthetic-imagenet"
