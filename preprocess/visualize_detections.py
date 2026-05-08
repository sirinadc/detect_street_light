"""
Bbox Görselleştirme — Tespit Kalitesi Kontrolü

Mantık:
    detect_poles.py çıktısındaki bbox'ları kareler üzerine çizer.
    Confidence'a göre renklendirir, dosya adına bilgi gömer.
    Böylece gözle inceleyip yanlış pozitifleri yakalayabilirsin.

Renk kodlaması:
    - Yeşil: conf >= 0.30  (muhtemelen gerçek direk)
    - Sarı:  conf 0.20-0.30 (sınır vakası)
    - Kırmızı: conf < 0.20  (muhtemelen yanlış pozitif)

Kullanım:
    # Yüksek pole_count'lu 10 kareye bak (yanlış pozitif yakalama):
    python visualize_detections.py --mode high-count --n 10

    # Rastgele 20 kareye bak (genel kalite kontrolü):
    python visualize_detections.py --mode random --n 20

    # Düşük confidence'lı kareleri görmek için:
    python visualize_detections.py --mode low-conf --n 15

    # Belirli bir confidence aralığı:
    python visualize_detections.py --mode range --conf-min 0.15 --conf-max 0.25 --n 10

Bağımlılıklar:
    pip install opencv-python pandas
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd


def parse_bboxes(bbox_str: str):
    """'x1,y1,x2,y2,conf,class;...' -> list of dicts"""
    if not bbox_str or pd.isna(bbox_str):
        return []
    bboxes = []
    for s in bbox_str.split(";"):
        parts = s.split(",")
        if len(parts) < 6:
            continue
        bboxes.append({
            "x1": int(parts[0]), "y1": int(parts[1]),
            "x2": int(parts[2]), "y2": int(parts[3]),
            "conf": float(parts[4]),
            "cls": parts[5],
        })
    return bboxes


def conf_color(conf: float):
    """Confidence'a göre BGR rengi döner."""
    if conf >= 0.30:
        return (0, 200, 0)      # yeşil
    elif conf >= 0.20:
        return (0, 200, 255)    # sarı
    else:
        return (0, 0, 255)      # kırmızı


def draw_detections(img_path: str, bboxes: list, output_path: Path):
    """Bbox'ları görsele çizer ve kaydeder."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  HATA: {img_path} okunamadı", file=sys.stderr)
        return False

    h, w = img.shape[:2]

    # Confidence'a göre sırala (yüksek conf üste çizilsin)
    bboxes_sorted = sorted(bboxes, key=lambda b: b["conf"])

    for b in bboxes_sorted:
        color = conf_color(b["conf"])
        # Çerçeve kalınlığı kareye göre orantılı
        thickness = max(2, h // 400)
        cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thickness)

        # Etiket
        label = f"{b['conf']:.2f}"
        font_scale = max(0.5, h / 1500)
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Etiket arka planı
        cv2.rectangle(img, (b["x1"], b["y1"] - th_ - 5),
                      (b["x1"] + tw + 4, b["y1"]), color, -1)
        cv2.putText(img, label, (b["x1"] + 2, b["y1"] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    # Üstte özet bilgi
    n = len(bboxes)
    high = sum(1 for b in bboxes if b["conf"] >= 0.30)
    mid = sum(1 for b in bboxes if 0.20 <= b["conf"] < 0.30)
    low = sum(1 for b in bboxes if b["conf"] < 0.20)
    summary = f"Total: {n}  Hi: {high}  Mid: {mid}  Lo: {low}"
    cv2.rectangle(img, (5, 5), (5 + len(summary) * 14, 35), (0, 0, 0), -1)
    cv2.putText(img, summary, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), img)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("frames_index_with_poles.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("./detection_check"))
    parser.add_argument("--mode", choices=["high-count", "random", "low-conf", "high-conf", "range"],
                        default="high-count",
                        help="high-count: çok direkli kareler (yanlış pozitif yakalama)\n"
                             "random: rastgele örnek\n"
                             "low-conf: düşük güvenli tespitler içeren kareler\n"
                             "high-conf: yüksek güvenli tespitler içeren kareler\n"
                             "range: belirli confidence aralığı")
    parser.add_argument("--n", type=int, default=10, help="Kaç kare seçilsin")
    parser.add_argument("--conf-min", type=float, default=0.15)
    parser.add_argument("--conf-max", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Hata: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    df_p = df[df["has_pole"] == True].copy()
    print(f"Direk içeren kare sayısı: {len(df_p)}")

    if args.mode == "high-count":
        sample = df_p.nlargest(args.n, "pole_count")
        print(f"En yüksek pole_count'lu {args.n} kare seçildi")

    elif args.mode == "random":
        sample = df_p.sample(min(args.n, len(df_p)), random_state=args.seed)
        print(f"Rastgele {len(sample)} kare seçildi")

    elif args.mode == "low-conf":
        # max_conf düşük olan kareler (model emin değilmiş)
        sample = df_p.nsmallest(args.n, "max_conf")
        print(f"En düşük max_conf'lu {args.n} kare seçildi")

    elif args.mode == "high-conf":
        sample = df_p.nlargest(args.n, "max_conf")
        print(f"En yüksek max_conf'lu {args.n} kare seçildi")

    elif args.mode == "range":
        in_range = df_p[(df_p["max_conf"] >= args.conf_min) & (df_p["max_conf"] <= args.conf_max)]
        sample = in_range.sample(min(args.n, len(in_range)), random_state=args.seed)
        print(f"max_conf [{args.conf_min}, {args.conf_max}] aralığından {len(sample)} kare")

    # Çıktı klasörünü temizle
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    ok = 0
    for _, row in sample.iterrows():
        bboxes = parse_bboxes(row["pole_bboxes"])
        # Dosya adına metadata göm
        light = row.get("light", "?")
        out_name = (
            f"count{row['pole_count']:02d}_"
            f"conf{row['max_conf']:.2f}_"
            f"{light}_"
            f"{Path(row['frame_path']).stem}.jpg"
        )
        if draw_detections(row["frame_path"], bboxes, args.output_dir / out_name):
            ok += 1

    print()
    print(f"{ok} görsel yazıldı -> {args.output_dir}")
    print()
    print("İncele:")
    print(f"  - YEŞİL kutular (conf >= 0.30): büyük ihtimalle gerçek direk")
    print(f"  - SARI kutular (conf 0.20-0.30): sınır vakası, gözle değerlendir")
    print(f"  - KIRMIZI kutular (conf < 0.20): muhtemelen yanlış pozitif")
    print()
    print("Karar verme rehberi:")
    print("  - Çoğu kırmızı/sarı yanlış pozitif → --conf eşiğini 0.25-0.30'a yükselt")
    print("  - Yeşiller bile yanlış pozitif → POLE_CLASSES sorgularını revize et")
    print("  - Sonuçlar genelde iyi → mevcut conf ile devam et")


if __name__ == "__main__":
    main()
