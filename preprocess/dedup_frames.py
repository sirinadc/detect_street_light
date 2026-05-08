"""
Perceptual Hash Tabanlı Kare Deduplication

Mantık:
    Her video içinde ardışık kareleri pHash ile karşılaştırır.
    Bir önceki "tutulan" kareye yeterince benzeyen kareleri atar.

Kullanım:
    python dedup_frames.py --input frames_index.csv --output frames_index_dedup.csv

    # Daha agresif eleme:
    python dedup_frames.py --threshold 3

    # Atılan/tutulan örnek çiftleri kaydet (kalibrasyon için):
    python dedup_frames.py --sample-pairs 20 --pairs-dir ./dedup_samples

Bağımlılıklar:
    pip install ImageHash pandas pillow tqdm
"""

import argparse
import shutil
import sys
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def compute_hash(path: str, hash_size: int = 8):
    """Bir görselin pHash'ini döner. Hata durumunda None."""
    try:
        with Image.open(path) as img:
            return imagehash.phash(img, hash_size=hash_size)
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        print(f"  HATA: {path} okunamadı ({e})", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("frames_index.csv"),
                        help="Kare indeksi CSV (extract_frames.py çıktısı)")
    parser.add_argument("--output", type=Path, default=Path("frames_index_dedup.csv"),
                        help="Dedup sonrası CSV")
    parser.add_argument("--threshold", type=int, default=5,
                        help="Hamming distance eşiği. Düşük = daha agresif eleme. Tipik: 3-8")
    parser.add_argument("--hash-size", type=int, default=8,
                        help="pHash boyutu (8 = 64-bit hash, hızlı; 16 = 256-bit, daha hassas)")
    parser.add_argument("--sample-pairs", type=int, default=0,
                        help="Kalibrasyon için kaç örnek atılan/tutulan çifti kaydedilsin (0 = kapalı)")
    parser.add_argument("--pairs-dir", type=Path, default=Path("./dedup_samples"),
                        help="Örnek çiftlerin yazılacağı klasör")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Hata: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    df = df.sort_values(["video", "frame_idx"]).reset_index(drop=True)
    print(f"{len(df)} kare yüklendi, {df['video'].nunique()} videoda dağılmış")

    keep = []
    last_hash_per_video = {}
    last_path_per_video = {}
    dropped_pairs = []  # kalibrasyon için: (atılan_path, referans_path, mesafe)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Hashing"):
        h = compute_hash(row["frame_path"], hash_size=args.hash_size)

        # Hash hesaplanamadıysa kareyi at (dosya bozuk)
        if h is None:
            keep.append(False)
            continue

        video = row["video"]
        last = last_hash_per_video.get(video)

        if last is None:
            # Bu videodaki ilk geçerli kare → tut
            keep.append(True)
            last_hash_per_video[video] = h
            last_path_per_video[video] = row["frame_path"]
        else:
            distance = h - last
            if distance > args.threshold:
                keep.append(True)
                last_hash_per_video[video] = h
                last_path_per_video[video] = row["frame_path"]
            else:
                keep.append(False)
                if len(dropped_pairs) < args.sample_pairs:
                    dropped_pairs.append((row["frame_path"], last_path_per_video[video], distance))

    df["keep"] = keep
    df_filtered = df[df["keep"]].drop(columns=["keep"]).reset_index(drop=True)

    # Çıktıyı yaz
    df_filtered.to_csv(args.output, index=False)

    # Özet rapor
    print()
    print(f"Genel: {len(df)} -> {len(df_filtered)} ({100*len(df_filtered)/len(df):.1f}% kaldı)")
    print(f"Atılan: {len(df) - len(df_filtered)} kare")
    print()

    # Video başına dağılım
    before = df.groupby("video").size()
    after = df_filtered.groupby("video").size()
    summary = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
    summary["kept_pct"] = (100 * summary["after"] / summary["before"]).round(1)
    print("Video başına:")
    print(summary.to_string())

    # Işık dağılımı (varsa)
    if "light" in df_filtered.columns:
        print()
        print("Işık koşulu dağılımı (dedup sonrası):")
        print(df_filtered["light"].value_counts().to_string())

    # Kalibrasyon örnekleri
    if dropped_pairs:
        args.pairs_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"Kalibrasyon için {len(dropped_pairs)} örnek çift -> {args.pairs_dir}")
        for i, (dropped, ref, dist) in enumerate(dropped_pairs):
            shutil.copy(ref,     args.pairs_dir / f"pair_{i:03d}_dist{dist}_KEPT.jpg")
            shutil.copy(dropped, args.pairs_dir / f"pair_{i:03d}_dist{dist}_DROPPED.jpg")
        print("KEPT/DROPPED dosyalarına bakıp threshold'un uygun olup olmadığını gözle:")
        print("  - DROPPED'ler gerçekten KEPT'e çok benziyorsa threshold doğru")
        print("  - DROPPED'lerde anlamlı yeni içerik varsa threshold'u düşür (--threshold 3)")


if __name__ == "__main__":
    main()
