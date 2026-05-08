"""
YOLO-World ile Direk Tespiti

Mantık:
    Her kareye YOLO-World çalıştırır, "street light", "lamp post"
    metin sorgularıyla direkleri tespit eder. Sonuçları CSV'ye işler.

    Kareleri SİLMEZ — sadece işaretler. Direk içermeyen kareler de
    has_pole=False olarak CSV'de kalır (negatif örnek olarak değerli).

Çıktı CSV'ye eklenen kolonlar:
    - has_pole       : direk tespit edildi mi (True/False)
    - pole_count     : tespit edilen direk sayısı
    - max_conf       : en yüksek güven skoru
    - max_class      : en yüksek skorlu direğin sınıfı
    - pole_bboxes    : tüm direklerin "x1,y1,x2,y2,conf,class" listesi (; ile ayrılmış)

Kullanım:
    # Tek bir kare ile test:
    python detect_poles.py --test path/to/frame.jpg

    # CSV'deki tüm kareleri işle:
    python detect_poles.py --input frames_index_dedup.csv --output frames_index_with_poles.csv

    # Daha büyük model (daha doğru, daha yavaş):
    python detect_poles.py --model yolov8m-worldv2.pt --conf 0.15

    # Sadece yüksek güvenli tespitleri tut:
    python detect_poles.py --conf 0.3

Bağımlılıklar:
    pip install ultralytics pandas tqdm
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLOWorld

# YOLO-World'e söyleyeceğimiz metin sorguları.
# Her biri ayrı bir "sınıf" olarak tespit edilir — sonradan ayırt edebiliriz.
POLE_CLASSES = [
    "street light",  # sokak aydınlatması (ana hedef)
    "lamp post",  # alternatif terim
    # "utility pole",  # elektrik direği (bonus)
]


def detect_in_image(model, image_path: str, conf_threshold: float, imgsz: int):
    """Bir görselde direk tespit eder, sonuçları dict olarak döner."""
    try:
        results = model.predict(
            image_path,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as e:
        print(f"  HATA: {image_path} işlenemedi ({e})", file=sys.stderr)
        return None

    if not results:
        return _empty_result()

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return _empty_result()

    # boxes.xyxy: [N, 4] piksel koordinatları
    # boxes.conf: [N] güven skorları
    # boxes.cls : [N] sınıf indeksleri (POLE_CLASSES içindeki sıra)
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_idx = boxes.cls.cpu().numpy().astype(int)

    bboxes_str = []
    for (x1, y1, x2, y2), conf, ci in zip(xyxy, confs, cls_idx):
        cls_name = POLE_CLASSES[ci] if ci < len(POLE_CLASSES) else "unknown"
        bboxes_str.append(f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f},{conf:.3f},{cls_name}")

    max_i = confs.argmax()
    return {
        "has_pole": True,
        "pole_count": len(confs),
        "max_conf": float(confs[max_i]),
        "max_class": (
            POLE_CLASSES[cls_idx[max_i]]
            if cls_idx[max_i] < len(POLE_CLASSES)
            else "unknown"
        ),
        "pole_bboxes": ";".join(bboxes_str),
    }


def _empty_result():
    return {
        "has_pole": False,
        "pole_count": 0,
        "max_conf": 0.0,
        "max_class": "",
        "pole_bboxes": "",
    }


def load_model(model_name: str):
    print(f"Model yükleniyor: {model_name}")
    model = YOLOWorld(model_name)
    model.set_classes(POLE_CLASSES)
    print(f"Sınıflar: {POLE_CLASSES}")
    return model


def cmd_test(args):
    """Tek bir görsel üzerinde test."""
    model = load_model(args.model)
    result = detect_in_image(model, args.test, args.conf, args.imgsz)
    if result is None:
        print("Tespit yapılamadı.")
        return
    print()
    print(f"Sonuç: {args.test}")
    for k, v in result.items():
        print(f"  {k}: {v}")


def cmd_process(args):
    """CSV'deki tüm kareleri işle."""
    if not args.input.exists():
        print(f"Hata: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"{len(df)} kare yüklendi")

    model = load_model(args.model)

    new_cols = {
        "has_pole": [],
        "pole_count": [],
        "max_conf": [],
        "max_class": [],
        "pole_bboxes": [],
    }

    for path in tqdm(df["frame_path"], desc="Tespit"):
        result = detect_in_image(model, path, args.conf, args.imgsz)
        if result is None:
            result = _empty_result()
        for k in new_cols:
            new_cols[k].append(result[k])

    for k, v in new_cols.items():
        df[k] = v

    df.to_csv(args.output, index=False)

    # Özet
    print()
    n_with = int(df["has_pole"].sum())
    print(f"Toplam: {len(df)} kare")
    print(f"Direk içeren: {n_with} ({100*n_with/len(df):.1f}%)")
    print(f"Direk içermeyen: {len(df)-n_with} ({100*(len(df)-n_with)/len(df):.1f}%)")
    print()

    if n_with > 0:
        print("Sınıf dağılımı (tespit edilen kareler):")
        print(df[df["has_pole"]]["max_class"].value_counts().to_string())
        print()
        print("Direk sayısı dağılımı:")
        print(df[df["has_pole"]]["pole_count"].value_counts().sort_index().to_string())

    # Işık × direk çapraz tablosu (varsa)
    if "light" in df.columns:
        print()
        print("Işık × direk varlığı:")
        print(pd.crosstab(df["light"], df["has_pole"]).to_string())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Tek bir görsel yolu — test modu. Verilirse CSV işlenmez.",
    )
    parser.add_argument("--input", type=Path, default=Path("frames_index_dedup.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("frames_index_with_poles.csv")
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-worldv2.pt",
        help="yolov8s-worldv2 (hızlı), yolov8m-worldv2 (dengeli), yolov8l-worldv2 (yavaş, doğru)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Güven eşiği. Filtreleme için düşük tutmak iyi (0.1-0.2)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Çıkarım çözünürlüğü. Daha yüksek = daha doğru, daha yavaş",
    )
    args = parser.parse_args()

    if args.test:
        cmd_test(args)
    else:
        cmd_process(args)


if __name__ == "__main__":
    main()
