"""
Direk Crop Üretimi (v3 — native, square crop bbox merkezde)

Mantık:
    select_main_poles.py'nin çıktısındaki her direk için, kareyi bbox + bağlam
    ile kırpıp ayrı bir JPEG dosyasına kaydeder.

    Crop'lar SQUARE (kare) — bbox merkezde, --expand-ratio kadar geniş.
    Sınır clamp: kenardaki direkler için crop kaydırılır (boyut korunur).
    NATIVE — resize YAPILMAZ. Eğitimde DataLoader resize yapacak.

    Native tutmanın faydaları:
    - Bilgi kaybı yok (uzak/yakın direklerde detay korunur)
    - Esneklik: farklı modeller farklı çözünürlükle eğitilebilir
    - Yeniden üretmeye gerek yok

Çıktı yapısı:
    crops/
    ├── images/
    │   ├── frame00045_pole1.jpg
    │   └── ...
    └── crops_index.csv

crops_index.csv kolonları:
    crop_id, crop_path, frame_path, video, city, light, timestamp,
    pole_rank,
    bbox_x1/y1/x2/y2, bbox_conf, bbox_class,
    crop_x1/y1/x2/y2 (imaj koordinatlarında),
    crop_width, crop_height,
    bbox_in_crop_x1/y1/x2/y2 (crop içindeki göreceli bbox koordinatları)

Kullanım:
    python generate_crops.py

    # select_main_poles ile aynı expand_ratio kullanılmalı:
    python generate_crops.py --expand-ratio 1.6

    # Daha düşük JPEG kalitesi (disk tasarrufu):
    python generate_crops.py --jpeg-quality 85

Bağımlılıklar:
    pip install pandas pillow tqdm
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def crop_centered_square(
    img: Image.Image, bx1: int, by1: int, bx2: int, by2: int, expand_ratio: float
):
    """Bbox merkezde square crop, sınırları clamp.

    Returns: (crop_img, (cx1, cy1, cx2, cy2)) — imaj koordinatlarında crop kutusu.
    """
    w, h = img.size
    bw = bx2 - bx1
    bh = by2 - by1

    side = int(max(bw, bh) * expand_ratio)
    side = min(side, min(w, h))  # imajdan büyük olamaz

    # Bbox merkezi
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2

    half = side // 2
    cx1 = cx - half
    cy1 = cy - half
    cx2 = cx1 + side
    cy2 = cy1 + side

    # Sınır clamp
    if cx1 < 0:
        cx1, cx2 = 0, side
    if cy1 < 0:
        cy1, cy2 = 0, side
    if cx2 > w:
        cx1, cx2 = w - side, w
    if cy2 > h:
        cy1, cy2 = h - side, h

    crop = img.crop((cx1, cy1, cx2, cy2))
    return crop, (cx1, cy1, cx2, cy2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("frames_index_selected.csv"),
        help="select_main_poles.py çıktısı",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./crops"),
        help="crops/images/ ve crops_index.csv buraya yazılır",
    )
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=1.6,
        help="Crop side = max(bbox_w, bbox_h) * expand_ratio. "
        "select_main_poles.py ile aynı olmalı.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG kalitesi (1-100, 85-95 makul)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Hata: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"{len(df)} direk yüklendi")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = 0
    cached_img_path = None
    cached_img = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Crop'lanıyor"):
        frame_path = row["frame_path"]

        # Aynı kareden ardışık crop'lar geliyorsa görseli bir kez aç (cache)
        if frame_path != cached_img_path:
            try:
                if cached_img is not None:
                    cached_img.close()
                cached_img = Image.open(frame_path).convert("RGB")
                cached_img_path = frame_path
            except Exception as e:
                print(f"  HATA: {frame_path} açılamadı ({e})", file=sys.stderr)
                errors += 1
                cached_img = None
                cached_img_path = None
                continue

        if cached_img is None:
            errors += 1
            continue

        bx1, by1 = int(row["bbox_x1"]), int(row["bbox_y1"])
        bx2, by2 = int(row["bbox_x2"]), int(row["bbox_y2"])

        try:
            crop, crop_box = crop_centered_square(
                cached_img,
                bx1,
                by1,
                bx2,
                by2,
                expand_ratio=args.expand_ratio,
            )
        except Exception as e:
            print(f"  HATA: {row['crop_id']} kırpılamadı ({e})", file=sys.stderr)
            errors += 1
            continue

        # Bbox'ın crop içindeki göreceli koordinatları
        cx1, cy1, cx2, cy2 = crop_box
        bbox_in_crop_x1 = bx1 - cx1
        bbox_in_crop_y1 = by1 - cy1
        bbox_in_crop_x2 = bx2 - cx1
        bbox_in_crop_y2 = by2 - cy1

        # Kaydet
        crop_path = images_dir / f"{row['crop_id']}.jpg"
        try:
            crop.save(crop_path, "JPEG", quality=args.jpeg_quality, optimize=True)
        except Exception as e:
            print(f"  HATA: {crop_path} yazılamadı ({e})", file=sys.stderr)
            errors += 1
            continue

        rows.append(
            {
                "crop_id": row["crop_id"],
                "crop_path": str(crop_path.resolve()),
                "frame_path": frame_path,
                "video": row.get("video", ""),
                "city": row.get("city", ""),
                "light": row.get("light", ""),
                "timestamp": row.get("timestamp", ""),
                "pole_rank": row.get("pole_rank", ""),
                "bbox_x1": bx1,
                "bbox_y1": by1,
                "bbox_x2": bx2,
                "bbox_y2": by2,
                "bbox_conf": row.get("bbox_conf", 0.0),
                "bbox_class": row.get("bbox_class", ""),
                "crop_x1": cx1,
                "crop_y1": cy1,
                "crop_x2": cx2,
                "crop_y2": cy2,
                "crop_width": crop.width,
                "crop_height": crop.height,
                "bbox_in_crop_x1": bbox_in_crop_x1,
                "bbox_in_crop_y1": bbox_in_crop_y1,
                "bbox_in_crop_x2": bbox_in_crop_x2,
                "bbox_in_crop_y2": bbox_in_crop_y2,
            }
        )

    if cached_img is not None:
        cached_img.close()

    out_df = pd.DataFrame(rows)
    crops_csv = args.output_dir / "crops_index.csv"
    out_df.to_csv(crops_csv, index=False)

    # Özet
    print()
    print(f"Üretilen crop sayısı: {len(rows)}")
    print(f"Hata: {errors}")
    print(f"Index: {crops_csv}")
    print(f"Görüntüler: {images_dir}")

    if not out_df.empty:
        avg_w = out_df["crop_width"].mean()
        avg_h = out_df["crop_height"].mean()
        min_side = out_df[["crop_width", "crop_height"]].min(axis=1).min()
        max_side = out_df[["crop_width", "crop_height"]].max(axis=1).max()
        print(f"Ortalama crop boyutu: {avg_w:.0f}x{avg_h:.0f}")
        print(f"Crop boyut aralığı: min_kenar={min_side}px, max_kenar={max_side}px")

        total_size = sum(p.stat().st_size for p in images_dir.glob("*.jpg"))
        size_mb = total_size / (1024 * 1024)
        print(
            f"Toplam disk: {size_mb:.1f} MB ({size_mb*1024/len(rows):.0f} KB/crop ortalama)"
        )


if __name__ == "__main__":
    main()
