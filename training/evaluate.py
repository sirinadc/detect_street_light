"""
Test Seti Değerlendirmesi + Hata Analizi

Mantık:
    1. Eğitilmiş modeli yükler
    2. Test setinde tahmin yapar
    3. Confusion matrix, P/R/F1, ROC AUC raporu
    4. Threshold-vs-metric grafiği
    5. Yanlış sınıflandırılan crop'ları görsel olarak kaydet
       (FN = arıza kaçırma, FP = yanlış alarm)

Çıktı:
    runs/exp_001/eval/
    ├── test_predictions.csv       (her crop için tahmin + olasılık)
    ├── confusion_matrix.png
    ├── threshold_curves.png
    ├── metrics_summary.json
    └── errors/
        ├── false_negatives/        (gerçek=problem, tahmin=ok — kaçırma)
        └── false_positives/        (gerçek=ok, tahmin=problem — yanlış alarm)

Kullanım:
    python evaluate.py
    python evaluate.py --threshold 0.3   # daha çok problem yakalamak için
    python evaluate.py --max-error-images 50

Bağımlılıklar:
    pip install torch torchvision pandas pillow scikit-learn matplotlib
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)
from torch.utils.data import DataLoader

# Aynı klasördeki train scriptinden import
sys.path.insert(0, str(Path(__file__).parent))
from model_train_test.train_classifier import PoleClassifierDataset, create_model


def evaluate_test_set(model, loader, device):
    """Test seti tahminleri."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, output_path: Path):
    """Confusion matrix görseli."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ok", "problem"])
    ax.set_yticklabels(["ok", "problem"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix (Test)")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_threshold_curves(labels, probs, output_path: Path):
    """Threshold'a göre P/R/F1 değişimi + ROC eğrisi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Threshold scan
    thresholds = np.linspace(0.05, 0.95, 50)
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    ax1.plot(thresholds, precisions, label="Precision", color="#1f77b4")
    ax1.plot(thresholds, recalls, label="Recall", color="#ff7f0e")
    ax1.plot(thresholds, f1s, label="F1", color="#2ca02c", linewidth=2)
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="default 0.5")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Threshold vs Metrics (problem class)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="#d62728", linewidth=2,
             label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return roc_auc


def save_error_images(df_test: pd.DataFrame, labels: np.ndarray,
                      probs: np.ndarray, threshold: float,
                      output_base: Path, max_images: int = 30):
    """Yanlış sınıflandırılan crop'ları görsel olarak kaydet (bbox ile)."""
    preds = (probs >= threshold).astype(int)

    fn_dir = output_base / "false_negatives"
    fp_dir = output_base / "false_positives"
    if fn_dir.exists():
        shutil.rmtree(fn_dir)
    if fp_dir.exists():
        shutil.rmtree(fp_dir)
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    fn_indices = np.where((preds == 0) & (labels == 1))[0]
    fp_indices = np.where((preds == 1) & (labels == 0))[0]

    # En "kötü" hataları seç (probability'e göre)
    # FN: probability düşük (ok sandı, halbuki problem) -> en düşüklerini al
    fn_indices = sorted(fn_indices, key=lambda i: probs[i])[:max_images]
    # FP: probability yüksek (problem sandı, halbuki ok) -> en yüksekler
    fp_indices = sorted(fp_indices, key=lambda i: -probs[i])[:max_images]

    def save_with_bbox(row, prob, dst_path):
        img = Image.open(row["crop_path"]).convert("RGB")
        # Crop içindeki bbox'ı çiz
        draw = ImageDraw.Draw(img)
        bx1 = int(row["bbox_in_crop_x1"])
        by1 = int(row["bbox_in_crop_y1"])
        bx2 = int(row["bbox_in_crop_x2"])
        by2 = int(row["bbox_in_crop_y2"])
        draw.rectangle([bx1, by1, bx2, by2], outline="red", width=3)
        draw.text((5, 5), f"prob={prob:.2f}", fill="yellow")
        img.save(dst_path, "JPEG", quality=90)

    print(f"FN örnek kaydediliyor: {len(fn_indices)} -> {fn_dir}")
    for i, idx in enumerate(fn_indices):
        row = df_test.iloc[idx]
        dst = fn_dir / f"FN_{i:03d}_p{probs[idx]:.2f}_{row['crop_id']}.jpg"
        try:
            save_with_bbox(row, probs[idx], dst)
        except Exception as e:
            print(f"  HATA: {e}")

    print(f"FP örnek kaydediliyor: {len(fp_indices)} -> {fp_dir}")
    for i, idx in enumerate(fp_indices):
        row = df_test.iloc[idx]
        dst = fp_dir / f"FP_{i:03d}_p{probs[idx]:.2f}_{row['crop_id']}.jpg"
        try:
            save_with_bbox(row, probs[idx], dst)
        except Exception as e:
            print(f"  HATA: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("./runs/exp_001/best_model.pt"),
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("./splits"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <checkpoint_parent>/eval/")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="problem sınıfı için karar eşiği")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-error-images", type=int, default=30,
                        help="FN ve FP klasörlerine kaç tane kaydedilsin (her biri)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent / "eval"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Checkpoint yükle
    print(f"Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_mask = ckpt.get("use_mask", True)
    image_size = ckpt.get("image_size", 224)
    print(f"Mask kullan: {use_mask}, Image size: {image_size}")

    model = create_model(num_classes=2, use_mask=use_mask, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Test loader
    test_ds = PoleClassifierDataset(
        args.splits_dir / "test.csv",
        image_size=image_size,
        use_mask=use_mask,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Test set: {len(test_ds)} crop")

    # Tahmin
    print("Tahmin yapılıyor...")
    labels, probs = evaluate_test_set(model, test_loader, device)

    # Test predictions CSV
    df_test = pd.read_csv(args.splits_dir / "test.csv")
    df_test["true_label"] = labels
    df_test["pred_prob"] = probs
    df_test["pred_label"] = (probs >= args.threshold).astype(int)
    df_test["correct"] = df_test["true_label"] == df_test["pred_label"]
    df_test.to_csv(args.output_dir / "test_predictions.csv", index=False)

    # Confusion matrix
    preds = (probs >= args.threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plot_confusion_matrix(cm, args.output_dir / "confusion_matrix.png")

    # Metrics
    tn, fp, fn_count, tp = cm.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn_count)

    # Threshold curves + ROC AUC
    roc_auc = plot_threshold_curves(
        labels, probs, args.output_dir / "threshold_curves.png"
    )

    # Özet
    summary = {
        "threshold": args.threshold,
        "n_test": int(len(labels)),
        "n_ok": int((labels == 0).sum()),
        "n_problem": int((labels == 1).sum()),
        "accuracy": float(accuracy),
        "precision_problem": float(precision),
        "recall_problem": float(recall),
        "f1_problem": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn_count), "TP": int(tp)
        },
    }
    with open(args.output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Hata görselleri
    save_error_images(
        df_test, labels, probs,
        threshold=args.threshold,
        output_base=args.output_dir / "errors",
        max_images=args.max_error_images,
    )

    # Konsol özeti
    print()
    print("=" * 60)
    print(f"TEST SONUÇLARI (threshold={args.threshold})")
    print("=" * 60)
    print(f"Test crop sayısı: {len(labels)}")
    print(f"  ok: {(labels==0).sum()}, problem: {(labels==1).sum()}")
    print()
    print(f"Confusion Matrix:")
    print(f"               Predicted")
    print(f"               ok    problem")
    print(f"True  ok      {tn:5d}  {fp:5d}")
    print(f"      problem {fn_count:5d}  {tp:5d}")
    print()
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision (problem): {precision:.4f}")
    print(f"Recall (problem):    {recall:.4f}")
    print(f"F1 (problem):        {f1:.4f}")
    print(f"ROC AUC:             {roc_auc:.4f}")
    print()
    print("Yorumlama:")
    print(f"  - {fn_count} arıza KAÇIRILDI (false negative)")
    print(f"  - {fp} yanlış alarm verildi (false positive)")
    print(f"  - {tp} arıza doğru bulundu (true positive)")
    print()
    print(f"Çıktılar -> {args.output_dir}/")


if __name__ == "__main__":
    main()
