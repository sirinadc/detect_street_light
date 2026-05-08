"""
EfficientNet-B0 + Bbox-Aware Classifier Eğitimi

Mantık:
    Modele 4 kanallı input verilir:
    - Kanal 0-2: RGB crop (224x224)
    - Kanal 3: Bbox mask (bbox içi=1, dışı=0, Gaussian smooth)

    Mask sayesinde model "ana direği" ayırt eder, civar lambaları
    bağlam olarak kullanır.

    Pretrained ImageNet ağırlıkları kullanılır (transfer learning).
    İlk conv layer 4 kanal olacak şekilde adapte edilir.

    Class weight ile dengesizlik (problem ~%15) çözülür.

Çıktı:
    runs/
    └── exp_001/
        ├── best_model.pt       (en iyi val F1'li checkpoint)
        ├── last_model.pt       (son epoch)
        ├── train_log.csv       (epoch başına metrikler)
        └── config.json         (kullanılan parametreler)

Kullanım:
    python train_classifier.py
    python train_classifier.py --epochs 30 --batch-size 16
    python train_classifier.py --no-mask  # bbox mask olmadan baseline

Bağımlılıklar:
    pip install torch torchvision pandas pillow scikit-learn tqdm
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ============================================================
# Dataset
# ============================================================

class PoleClassifierDataset(Dataset):
    """Crop + (opsiyonel) bbox mask -> label dataset."""

    def __init__(self, csv_path: Path, image_size: int = 224,
                 use_mask: bool = True, augment: bool = False,
                 mask_blur_sigma: float = 5.0):
        """
        Args:
            csv_path: train.csv / val.csv / test.csv
            image_size: modele verilecek boyut (224 = EfficientNet-B0 default)
            use_mask: 4. kanal mask kullan (False ise sadece RGB)
            augment: training sırasında augmentation uygula
            mask_blur_sigma: bbox mask'ini blur'la (yumuşak geçiş için)
        """
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.use_mask = use_mask
        self.augment = augment
        self.mask_blur_sigma = mask_blur_sigma

        # Label mapping
        self.label_map = {"ok": 0, "problem": 1}

        # Augmentation transforms (RGB için)
        if augment:
            self.color_aug = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        else:
            self.color_aug = None

        # ImageNet normalize
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Crop oku
        crop = Image.open(row["crop_path"]).convert("RGB")
        crop_w, crop_h = crop.size

        # Bbox koordinatları (crop içinde)
        bx1 = int(row["bbox_in_crop_x1"])
        by1 = int(row["bbox_in_crop_y1"])
        bx2 = int(row["bbox_in_crop_x2"])
        by2 = int(row["bbox_in_crop_y2"])

        # Augmentation: horizontal flip (mask ile birlikte)
        do_flip = self.augment and np.random.random() < 0.5
        if do_flip:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            # Bbox koordinatları da flipper
            bx1, bx2 = crop_w - bx2, crop_w - bx1

        # Color augmentation (sadece RGB'ye)
        if self.color_aug is not None:
            crop = self.color_aug(crop)

        # 224x224'e resize
        crop_resized = crop.resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        crop_array = np.array(crop_resized, dtype=np.float32) / 255.0  # HWC

        # RGB tensor (CHW)
        rgb_tensor = torch.from_numpy(crop_array).permute(2, 0, 1)
        rgb_tensor = self.normalize(rgb_tensor)

        if self.use_mask:
            # Mask oluştur (orijinal crop boyutunda, sonra resize)
            mask = np.zeros((crop_h, crop_w), dtype=np.float32)
            # Sınırları kontrol et
            bx1c = max(0, min(crop_w, bx1))
            by1c = max(0, min(crop_h, by1))
            bx2c = max(0, min(crop_w, bx2))
            by2c = max(0, min(crop_h, by2))
            mask[by1c:by2c, bx1c:bx2c] = 1.0

            # Yumuşak geçiş için Gaussian blur
            if self.mask_blur_sigma > 0:
                from scipy.ndimage import gaussian_filter
                mask = gaussian_filter(mask, sigma=self.mask_blur_sigma)

            # Resize to image_size
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_img.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            mask_tensor = torch.from_numpy(
                np.array(mask_resized, dtype=np.float32) / 255.0
            ).unsqueeze(0)  # 1xHxW

            # 4 kanallı tensor
            input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)
        else:
            input_tensor = rgb_tensor

        label = self.label_map[row["label"]]
        return input_tensor, label


# ============================================================
# Model
# ============================================================

def create_model(num_classes: int = 2, use_mask: bool = True,
                 pretrained: bool = True):
    """EfficientNet-B0, opsiyonel olarak 4 kanallı input için adapte edilmiş."""
    if pretrained:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    else:
        model = models.efficientnet_b0(weights=None)

    if use_mask:
        # İlk conv layer'ı 3 -> 4 kanala genişlet
        old_conv = model.features[0][0]  # Conv2d(3, 32, kernel=3, stride=2, padding=1)
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # Pretrained ağırlıkları ilk 3 kanala kopyala
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # 4. kanal: RGB ortalaması ile başlat (orta değerli init)
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias
        model.features[0][0] = new_conv

    # Son katmanı 2 sınıfa indirgeyelim
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# Eğitim
# ============================================================

def evaluate_model(model, loader, device, threshold: float = 0.5):
    """Validation/test setinde değerlendirme yap."""
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    n_batches = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)[:, 1]  # problem olasılığı
            preds = (probs >= threshold).long()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": (all_preds == all_labels).mean(),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    if len(np.unique(all_labels)) >= 2:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets
    train_ds = PoleClassifierDataset(
        args.splits_dir / "train.csv",
        image_size=args.image_size,
        use_mask=not args.no_mask,
        augment=True,
        mask_blur_sigma=args.mask_blur,
    )
    val_ds = PoleClassifierDataset(
        args.splits_dir / "val.csv",
        image_size=args.image_size,
        use_mask=not args.no_mask,
        augment=False,
        mask_blur_sigma=args.mask_blur,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = create_model(
        num_classes=2,
        use_mask=not args.no_mask,
        pretrained=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametreleri: {n_params/1e6:.2f}M")
    print(f"Mode: {'4-channel (RGB+mask)' if not args.no_mask else '3-channel (RGB only)'}")

    # Class weight (problem class daha az olduğu için ağırlıklandır)
    train_df = pd.read_csv(args.splits_dir / "train.csv")
    n_ok = (train_df["label"] == "ok").sum()
    n_prob = (train_df["label"] == "problem").sum()
    pos_weight = n_ok / max(n_prob, 1)
    print(f"Class weight: ok=1.0, problem={pos_weight:.2f}")
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Output klasörü
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Eğitim loop
    log_rows = []
    best_val_f1 = -1
    best_epoch = -1

    print()
    print("=" * 60)
    print("EĞİTİM BAŞLIYOR")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total
        scheduler.step()

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"({epoch_time:.0f}s) | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val: loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.3f} "
            f"f1={val_metrics['f1']:.3f} "
            f"recall={val_metrics['recall']:.3f} "
            f"prec={val_metrics['precision']:.3f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": scheduler.get_last_lr()[0],
        })
        pd.DataFrame(log_rows).to_csv(
            args.output_dir / "train_log.csv", index=False
        )

        # Best model checkpoint (val F1'e göre)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "use_mask": not args.no_mask,
                    "image_size": args.image_size,
                },
                args.output_dir / "best_model.pt",
            )
            print(f"  -> Yeni best (val F1={best_val_f1:.4f})")

    # Son model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "use_mask": not args.no_mask,
            "image_size": args.image_size,
        },
        args.output_dir / "last_model.pt",
    )

    print()
    print("=" * 60)
    print(f"EĞİTİM BİTTİ — Best epoch: {best_epoch} (val F1={best_val_f1:.4f})")
    print(f"Checkpoint'ler: {args.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("./splits"),
        help="train/val/test.csv'lerin olduğu klasör (prepare_dataset.py çıktısı)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./runs/exp_001"),
        help="Checkpoint ve log'ların yazılacağı klasör",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker sayısı (CPU eğitimde 0-2 yap)")
    parser.add_argument("--mask-blur", type=float, default=5.0,
                        help="Bbox mask Gaussian blur sigma (0=keskin)")
    parser.add_argument("--no-mask", action="store_true",
                        help="Sadece RGB kullan (3 kanal baseline)")
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
