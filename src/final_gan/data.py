from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RecursiveImageDataset(Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.root}. "
                "Place LFW images there or pass --data-root."
            )
        self.files = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise FileNotFoundError(f"No images found under {self.root}.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def build_transforms(image_size: int = 64) -> transforms.Compose:
    resize_size = max(image_size, int(round(image_size * 1.1)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_dataset(config: dict) -> Dataset:
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    transform = build_transforms(int(data_cfg.get("image_size", 64)))
    dataset_name = data_cfg.get("dataset", "recursive_images")

    if dataset_name == "lfw_people":
        return datasets.LFWPeople(root=str(root), split="train", download=False, transform=transform)
    if dataset_name == "image_folder":
        return datasets.ImageFolder(root=str(root), transform=transform)
    if dataset_name == "recursive_images":
        return RecursiveImageDataset(root, transform=transform)

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Use recursive_images, image_folder, or lfw_people."
    )


def build_dataloader(config: dict, shuffle: bool = True) -> DataLoader:
    dataset = build_dataset(config)
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 2))
    persistent_workers = bool(data_cfg.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    pin_memory = torch.cuda.is_available()
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": shuffle,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
