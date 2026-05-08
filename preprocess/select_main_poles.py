"""
Direk Seçimi (v2 — conf-tabanlı + crop alanı IoU eleme)

Mantık:
    detect_poles.py her karede bulduğu TÜM direklerin bbox'larını çıktı verdi.
    Bu script o direklerden SINIFLANDIRMA için en uygun olanları seçer:

    1. Min conf eşiği (mutlak): bbox_conf >= --min-conf
    2. En yüksek conf'lu = ana hedef (pole_rank=1)
    3. Yan direkler için göreceli conf: ana'nın --rel-conf-threshold'undan büyük
    4. Crop alanı IoU eleme: önceki seçilenlerin crop'u ile çakışıyorsa atla
    5. Maks --max-poles-per-frame direk seç

    Crop alanı IoU eleme önemli — aynı sahnedeki üst üste binmiş bbox'lar
    (örn. çift tespit, çok yakın direkler) eğitime aynı bağlamı çoklu kez
    sokar. IoU eşiğiyle bunları ayıklarız.

Kullanım:
    # Varsayılan parametrelerle:
    python select_main_poles.py

    # Daha gevşek conf:
    python select_main_poles.py --min-conf 0.18

    # Daha sıkı IoU eleme:
    python select_main_poles.py --max-crop-iou 0.2

    # Daha çok bağlam (ama pole_rank=1 hariç direkler için):
    python select_main_poles.py --rel-conf-threshold 0.6

Bağımlılıklar:
    pip install pandas pillow tqdm
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_bboxes(bbox_str: str):
    """'x1,y1,x2,y2,conf,class;...' -> list of dicts"""
    if not bbox_str or pd.isna(bbox_str):
        return []
    bboxes = []
    for s in bbox_str.split(";"):
        parts = s.split(",")
        if len(parts) < 6:
            continue
        bboxes.append(
            {
                "x1": int(parts[0]),
                "y1": int(parts[1]),
                "x2": int(parts[2]),
                "y2": int(parts[3]),
                "conf": float(parts[4]),
                "cls": parts[5],
            }
        )
    return bboxes


def get_image_size(path: str):
    """Görsel boyutunu döner. Hata durumunda None."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def compute_crop_box(bbox: dict, img_w: int, img_h: int, expand_ratio: float):
    """Bbox merkezde square crop hesapla (sınırları clamp et).

    crop_box = (x1, y1, x2, y2) imaj koordinatlarında.
    """
    bx1, by1, bx2, by2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    bw = bx2 - bx1
    bh = by2 - by1

    # Square side = max(bw, bh) * expand_ratio
    side = int(max(bw, bh) * expand_ratio)
    side = min(side, min(img_w, img_h))  # imajdan büyük olamaz

    # Bbox merkezi
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2

    # Square crop merkez bbox merkezi olacak şekilde
    half = side // 2
    cx1 = cx - half
    cy1 = cy - half
    cx2 = cx1 + side
    cy2 = cy1 + side

    # Sınır clamp — kareyi kaydır (boyut korunur)
    if cx1 < 0:
        cx1, cx2 = 0, side
    if cy1 < 0:
        cy1, cy2 = 0, side
    if cx2 > img_w:
        cx1, cx2 = img_w - side, img_w
    if cy2 > img_h:
        cy1, cy2 = img_h - side, img_h

    return (cx1, cy1, cx2, cy2)


def compute_iou(box_a, box_b):
    """İki dikdörtgen için IoU (Intersection over Union)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def select_poles_for_frame(
    bboxes: list,
    img_w: int,
    img_h: int,
    min_conf: float,
    rel_conf_threshold: float,
    max_crop_iou: float,
    expand_ratio: float,
    max_poles: int,
):
    """Bir karedeki bbox'lardan etiketlenmek üzere seçim yap.

    Dönen değer: seçilmiş direklerin listesi (her biri dict),
    her birine `pole_rank`, `crop_box` eklenmiş halde.
    """
    if not bboxes:
        return []

    # Min conf'tan büyük olanları al
    candidates = [b for b in bboxes if b["conf"] >= min_conf]
    if not candidates:
        return []

    # Conf'a göre sırala (yüksekten düşüğe)
    candidates.sort(key=lambda x: x["conf"], reverse=True)

    # Ana hedef = en yüksek conf
    main_conf = candidates[0]["conf"]
    rel_threshold = main_conf * rel_conf_threshold

    selected = []
    selected_crops = []  # IoU karşılaştırması için

    for cand in candidates:
        # Conf eleme (yan direkler için): ana'nın %X'inden düşükse atla
        # (ana direk her zaman geçer, kendisiyle karşılaştırma)
        if selected and cand["conf"] < rel_threshold:
            break  # sıralı olduğu için bundan sonrası da düşük

        # Crop kutusunu hesapla
        crop_box = compute_crop_box(cand, img_w, img_h, expand_ratio)

        # IoU eleme: önceki seçilenlerden herhangi biriyle çok örtüşüyorsa atla
        skip = False
        for prev_crop in selected_crops:
            if compute_iou(crop_box, prev_crop) > max_crop_iou:
                skip = True
                break
        if skip:
            continue

        # Geçti — ekle
        cand["crop_box"] = crop_box
        cand["pole_rank"] = len(selected) + 1
        selected.append(cand)
        selected_crops.append(crop_box)

        if len(selected) >= max_poles:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("frames_index_with_poles.csv"),
        help="detect_poles.py çıktısı",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frames_index_selected.csv"),
        help="Seçilmiş direkler CSV'si (her satır = 1 direk)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.23,
        help="Mutlak min confidence eşiği (ana direk için)",
    )
    parser.add_argument(
        "--rel-conf-threshold",
        type=float,
        default=0.7,
        help="Yan direkler için ana direğe göre conf oranı (0-1). "
        "0.7 = ana'nın %%70'i kadar conf olmalı",
    )
    parser.add_argument(
        "--max-crop-iou",
        type=float,
        default=0.3,
        help="Crop alanları IoU eşiği — bu değerden fazla "
        "örtüşürse direk elenir (sadece daha düşük conf'lu)",
    )
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=1.6,
        help="Crop side = max(bbox_w, bbox_h) * expand_ratio. "
        "generate_crops.py ile aynı olmalı.",
    )
    parser.add_argument(
        "--max-poles-per-frame",
        type=int,
        default=3,
        help="Bir kareden alınacak maksimum direk sayısı",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Hata: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"{len(df)} kare yüklendi")

    # Sadece direk içerenler
    df_p = df[df["has_pole"] == True].copy()
    print(f"  Direk içeren: {len(df_p)}")
    print(
        f"  Parametreler: min_conf={args.min_conf}, "
        f"rel_threshold={args.rel_conf_threshold}, "
        f"max_iou={args.max_crop_iou}, "
        f"expand={args.expand_ratio}, "
        f"max_per_frame={args.max_poles_per_frame}"
    )
    print()

    rows = []
    skipped_no_size = 0
    skipped_no_candidates = 0

    for _, row in tqdm(df_p.iterrows(), total=len(df_p), desc="Seçiliyor"):
        bboxes = parse_bboxes(row["pole_bboxes"])
        if not bboxes:
            continue

        size = get_image_size(row["frame_path"])
        if size is None:
            skipped_no_size += 1
            continue
        img_w, img_h = size

        selected = select_poles_for_frame(
            bboxes,
            img_w,
            img_h,
            min_conf=args.min_conf,
            rel_conf_threshold=args.rel_conf_threshold,
            max_crop_iou=args.max_crop_iou,
            expand_ratio=args.expand_ratio,
            max_poles=args.max_poles_per_frame,
        )

        if not selected:
            skipped_no_candidates += 1
            continue

        frame_stem = Path(row["frame_path"]).stem
        for s in selected:
            rows.append(
                {
                    "crop_id": f"{frame_stem}_pole{s['pole_rank']}",
                    "frame_path": row["frame_path"],
                    "video": row["video"],
                    "city": row.get("city", ""),
                    "timestamp": row.get("timestamp", ""),
                    "light": row.get("light", ""),
                    "image_width": img_w,
                    "image_height": img_h,
                    "bbox_x1": s["x1"],
                    "bbox_y1": s["y1"],
                    "bbox_x2": s["x2"],
                    "bbox_y2": s["y2"],
                    "bbox_conf": s["conf"],
                    "bbox_class": s["cls"],
                    "pole_rank": s["pole_rank"],
                    # crop_x1...y2 burada hesaplanır ama generate_crops da
                    # tekrar hesaplar (çünkü target_size resize için lazım olabilir)
                    "crop_x1": s["crop_box"][0],
                    "crop_y1": s["crop_box"][1],
                    "crop_x2": s["crop_box"][2],
                    "crop_y2": s["crop_box"][3],
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    # Özet
    print()
    print(f"Sonuç: {len(rows)} direk seçildi -> {args.output}")
    print(f"  Atlanan kareler (boyut okunamadı): {skipped_no_size}")
    print(f"  Atlanan kareler (uygun direk yok): {skipped_no_candidates}")
    print()

    if not out_df.empty:
        per_frame = out_df.groupby("frame_path").size()
        print("Kare başına seçilen direk sayısı:")
        for n in sorted(per_frame.unique()):
            count = (per_frame == n).sum()
            print(f"  {n} direk: {count} kare")
        print(f"  Toplam etiketlenebilir kare: {len(per_frame)}")
        print()

        if "light" in out_df.columns:
            print("Işık koşulu (seçilen direkler):")
            print(out_df["light"].value_counts().to_string())
            print()

        # pole_rank dağılımı
        print("pole_rank dağılımı (1=ana, 2=yan, 3=yan):")
        print(out_df["pole_rank"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
