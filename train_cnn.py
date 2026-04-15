from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGET_CLASSES = ["no_damage", "minor_crack", "major_crack"]


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    epochs: int = 8
    batch_size: int = 16
    lr: float = 1e-3
    num_workers: int = 2
    device: str = "cpu"
    output: Path = Path("model.pth")
    image_size: int = 224


class RemapTargetsDataset:
    """Wraps ImageFolder to enforce TARGET_CLASSES index order.

    ImageFolder assigns indices alphabetically by folder name, which may not match
    our inference label order. This wrapper remaps targets to TARGET_CLASSES.
    """

    def __init__(self, base, class_to_idx_map: dict[str, int]):
        self.base = base
        self.class_to_idx_map = class_to_idx_map

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        class_name = self.base.classes[int(y)]
        return x, int(self.class_to_idx_map[class_name])


def _accuracy(logits, y) -> float:
    import torch

    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


def _iter_batches(loader) -> Iterable:
    for batch in loader:
        yield batch


def main() -> int:
    ap = argparse.ArgumentParser(description="Train MobileNetV2 for screen damage (3 classes).")
    ap.add_argument("--data-dir", default="data", help="Dataset root containing train/ and val/ folders.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=os.getenv("TORCH_DEVICE", "cpu"), help="cpu or cuda")
    ap.add_argument("--output", default="model.pth", help="Output checkpoint path.")
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=Path(args.data_dir),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=float(args.lr),
        num_workers=max(0, int(args.num_workers)),
        device=str(args.device),
        output=Path(args.output),
        image_size=max(64, int(args.image_size)),
    )

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import MobileNet_V2_Weights
    import torchvision.models as models

    train_dir = cfg.data_dir / "train"
    val_dir = cfg.data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected dataset structure:\n"
            f"  {cfg.data_dir}/train/<class>/...\n"
            f"  {cfg.data_dir}/val/<class>/...\n"
            f"Missing: {train_dir if not train_dir.exists() else val_dir}"
        )

    # Fast, decent augmentations for cracked-screen photos.
    train_tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_train = datasets.ImageFolder(str(train_dir), transform=train_tfm)
    base_val = datasets.ImageFolder(str(val_dir), transform=val_tfm)

    # Ensure all required classes exist.
    missing = [c for c in TARGET_CLASSES if c not in base_train.class_to_idx or c not in base_val.class_to_idx]
    if missing:
        raise SystemExit(f"Missing class folders in train/val: {missing}. Expected {TARGET_CLASSES}")

    class_to_idx = {name: i for i, name in enumerate(TARGET_CLASSES)}
    train_ds = RemapTargetsDataset(base_train, class_to_idx)
    val_ds = RemapTargetsDataset(base_val, class_to_idx)

    device = torch.device("cuda" if cfg.device.strip().lower() == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"device={device} epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr}")
    print(f"train_images={len(train_ds)} val_images={len(val_ds)} classes={TARGET_CLASSES}")
    print(f"imagefolder_train_classes={base_train.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    net = models.mobilenet_v2(weights=weights)
    net.classifier[1] = nn.Linear(net.last_channel, len(TARGET_CLASSES))

    # Freeze backbone for fast MVP iteration; train head only.
    for p in net.features.parameters():
        p.requires_grad = False

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.classifier.parameters(), lr=cfg.lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.perf_counter()
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        for x, y in _iter_batches(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = int(y.shape[0])
            n_train += bs
            train_loss += float(loss.item()) * bs
            train_acc += _accuracy(logits, y) * bs

        net.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in _iter_batches(val_loader):
                x = x.to(device)
                y = y.to(device)
                logits = net(x)
                loss = criterion(logits, y)
                bs = int(y.shape[0])
                n_val += bs
                val_loss += float(loss.item()) * bs
                val_acc += _accuracy(logits, y) * bs

        train_loss = train_loss / max(1, n_train)
        train_acc = train_acc / max(1, n_train)
        val_loss = val_loss / max(1, n_val)
        val_acc = val_acc / max(1, n_val)

        dt = time.perf_counter() - t_epoch
        print(
            f"epoch {epoch:02d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"secs={dt:.1f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    if best_state is None:
        raise SystemExit("Training produced no checkpoint state.")

    ckpt = {
        "arch": "mobilenet_v2",
        "image_size": cfg.image_size,
        "idx_to_class": list(TARGET_CLASSES),
        "class_to_idx": dict(class_to_idx),
        "state_dict": best_state,
        "best_val_acc": float(best_val_acc),
        "meta": {"notes": "features frozen; head trained"},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(cfg.output))

    print(f"saved={cfg.output} best_val_acc={best_val_acc:.4f}")
    print("checkpoint_meta=" + json.dumps({k: ckpt[k] for k in ("arch", "image_size", "idx_to_class", "best_val_acc")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

