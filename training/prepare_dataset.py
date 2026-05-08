"""
Eğitim Veri Seti Hazırlığı (Train / Val / Test Split)

Mantık:
    crops_index.csv'den ok+problem crop'larını alır.
    Video bazlı (group-aware) split yapar — aynı videonun crop'ları
    aynı bölmede kalır (data leakage önleme).

    Sınıf dengesini stratify olarak korumaya çalışır (mümkün olduğu kadar).

Çıktı:
    splits/
    ├── train.csv  (kullanılacak crop'lar)
    ├── val.csv
    └── test.csv

Her CSV'de tüm orijinal kolonlar + yeni "split" kolonu.

Kullanım:
    python prepare_dataset.py
    python prepare_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
    python prepare_dataset.py --min-crop-size 100   # küçük crop'ları filtrele

Bağımlılıklar:
    pip install pandas scikit-learn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./crops/crops_index.csv"),
        help="generate_crops.py çıktısı (etiketlenmiş)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./splits"),
        help="train.csv, val.csv, test.csv buraya yazılır",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=0,
        help="Bu boyuttan küçük crop'ları (kenar) eğitime alma (0=filtre yok)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Toplam ratio kontrolü
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"HATA: train+val+test = {total} (1.0 olmalı)", file=sys.stderr)
        sys.exit(1)

    if not args.input.exists():
        print(f"HATA: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Yüklenen toplam crop: {len(df)}")

    # Sadece ok ve problem etiketli olanlar
    df = df[df["label"].isin(["ok", "problem"])].copy().reset_index(drop=True)
    print(f"Eğitim için (ok+problem): {len(df)}")
    print(f"  ok: {(df['label']=='ok').sum()}")
    print(f"  problem: {(df['label']=='problem').sum()}")

    # Min crop size filtresi (opsiyonel)
    if args.min_crop_size > 0:
        before = len(df)
        df = df[
            (df["crop_width"] >= args.min_crop_size)
            & (df["crop_height"] >= args.min_crop_size)
        ].reset_index(drop=True)
        print(
            f"Min crop size {args.min_crop_size} filtresi: {before} -> {len(df)} "
            f"({100*len(df)/before:.1f}% kaldı)"
        )

    # Group split: aynı video aynı bölmede
    # 1. Önce test'i ayır
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=args.test_ratio, random_state=args.seed
    )
    train_val_idx, test_idx = next(gss1.split(df, groups=df["video"]))

    df_test = df.iloc[test_idx].copy()
    df_train_val = df.iloc[train_val_idx].copy()

    # 2. Geri kalandan val'i ayır
    val_size_relative = args.val_ratio / (args.train_ratio + args.val_ratio)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_size_relative, random_state=args.seed + 1
    )
    train_idx, val_idx = next(
        gss2.split(df_train_val, groups=df_train_val["video"])
    )

    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()

    # Split kolonu ekle
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # Dosyaya yaz
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(args.output_dir / "train.csv", index=False)
    df_val.to_csv(args.output_dir / "val.csv", index=False)
    df_test.to_csv(args.output_dir / "test.csv", index=False)

    # Birleşik dosya da yazalım (analiz için)
    df_all = pd.concat([df_train, df_val, df_test])
    df_all.to_csv(args.output_dir / "all_splits.csv", index=False)

    # Özet
    print()
    print("=" * 60)
    print("SPLIT SONUÇLARI")
    print("=" * 60)
    for name, dd in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n = len(dd)
        n_ok = (dd["label"] == "ok").sum()
        n_prob = (dd["label"] == "problem").sum()
        n_videos = dd["video"].nunique()
        prob_pct = 100 * n_prob / n if n > 0 else 0
        print(
            f"{name:6s}: {n:5d} crop  (ok:{n_ok:5d}, prob:{n_prob:4d} = "
            f"{prob_pct:.1f}%)  [{n_videos:3d} video]"
        )

    # Video çakışma kontrolü (data leakage)
    train_videos = set(df_train["video"])
    val_videos = set(df_val["video"])
    test_videos = set(df_test["video"])
    overlap_tv = train_videos & val_videos
    overlap_tt = train_videos & test_videos
    overlap_vt = val_videos & test_videos
    print()
    print("Video çakışma kontrolü (0 olmalı):")
    print(f"  train ∩ val:  {len(overlap_tv)}")
    print(f"  train ∩ test: {len(overlap_tt)}")
    print(f"  val ∩ test:   {len(overlap_vt)}")

    if overlap_tv or overlap_tt or overlap_vt:
        print("UYARI: Video çakışması var! Data leakage riski.", file=sys.stderr)
    else:
        print("✓ Video çakışması yok, data leakage riski yok")

    # Light dağılımı
    print()
    print("Light dağılımı (split bazlı):")
    print(pd.crosstab(df_all["split"], df_all["light"]).to_string())

    print()
    print(f"CSV'ler -> {args.output_dir}/")


if __name__ == "__main__":
    main()
