"""
Klasör Tabanlı Triage Etiketleme (v1 — veri gelince UX kalibre edilecek)

Mantık:
    crops_index.csv'deki crop'ları sırayla gösterir, klavye ile etiketler.
    Etiketlenen crop'lar sınıf klasörlerine taşınır.
    crops_index.csv'ye 'label' kolonu eklenir.

    Crop'la birlikte ekranda metadata (light/city/timestamp) görünür —
    karar vermeyi kolaylaştırır.

Klavye kısayolları (etiketleme sırasında):
    O = ok            (direk normal — lamba yanıyor)
    P = problem       (lamba yanmıyor — arıza adayı)
    U = unclear       (emin değilim)
    N = not_pole      (bu aslında direk değil — YOLO hatası)
    Backspace = bir önceki crop'a dön (etiketi geri al)
    S = skip — bu crop'u şimdilik atla, sonra bakarım
    Q = çık — ilerlemeyi kaydet ve kapat
    H = yardım göster

Kullanım:
    python triage_label.py
    python triage_label.py --input ./crops/crops_index.csv

    # Sadece belirli koşulu etiketle:
    python triage_label.py --filter-light night
    python triage_label.py --filter-light twilight,night    # multi-value
    python triage_label.py --filter-city Hatay

    # Pencere boyutunu ayarla:
    python triage_label.py --window-width 1200

Bağımlılıklar:
    pip install pandas opencv-python
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LABELS = {
    ord("o"): "ok",
    ord("O"): "ok",
    ord("p"): "problem",
    ord("P"): "problem",
    ord("u"): "unclear",
    ord("U"): "unclear",
    ord("n"): "not_pole",
    ord("N"): "not_pole",
}

# Tuş kodları (cv2 waitKey)
KEY_BACKSPACE = 8
KEY_BACKSPACE_ALT = 127  # Mac
KEY_Q = ord("q")
KEY_Q_UPPER = ord("Q")
KEY_S = ord("s")
KEY_S_UPPER = ord("S")
KEY_H = ord("h")
KEY_H_UPPER = ord("H")
KEY_ESC = 27


def render_frame(crop_path: str, info: dict, window_w: int):
    """Crop'u info paneliyle birlikte gösterilebilir bir görsele çevirir."""
    img = cv2.imread(crop_path)
    if img is None:
        # Hata durumunda boş canvas
        img = 50 * np.ones((400, 300, 3), dtype="uint8")

    h, w = img.shape[:2]

    # Crop'u pencerenin sol yarısına sığdır
    target_w = window_w // 2 - 40
    target_h = 600
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    crop_resized = cv2.resize(img, (new_w, new_h))

    # Canvas (siyah arka plan)
    canvas_h = max(700, new_h + 100)
    canvas = np.zeros((canvas_h, window_w, 3), dtype="uint8")

    # Crop'u sola yerleştir
    y_offset = (canvas_h - new_h) // 2
    x_offset = 20
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = crop_resized

    # Sağa info paneli
    panel_x = window_w // 2 + 20
    line_y = 60

    def put(label, value, color=(255, 255, 255)):
        nonlocal line_y
        cv2.putText(
            canvas,
            label,
            (panel_x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            canvas,
            str(value),
            (panel_x + 180, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        line_y += 35

    # Başlık
    cv2.putText(
        canvas,
        info["progress"],
        (panel_x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (100, 200, 255),
        2,
    )
    line_y = 70

    put("crop_id:", info["crop_id"])
    put("light:", info["light"], color=light_color(info["light"]))
    put("city:", info["city"])
    put("time:", info.get("time", "?"))
    put(
        "pole_rank:",
        f"{info['pole_rank']}",
    )
    put(
        "bbox_conf:",
        f"{info.get('bbox_conf', 0):.2f}" if info.get("bbox_conf") else "?",
    )

    # Mevcut etiket (varsa)
    cur = info.get("current_label")
    if cur and isinstance(cur, str):
        put(
            "CURRENT:",
            cur.upper(),
            color=label_color(cur),
        )

    # Klavye yardımı
    line_y = canvas_h - 200
    helps = [
        ("[O]", "ok (yaniyor)"),
        ("[P]", "problem (yanmiyor)"),
        ("[U]", "unclear"),
        ("[N]", "not_pole (direk degil)"),
        ("[S]", "skip"),
        ("[Backspace]", "geri"),
        ("[Q]", "cik"),
    ]
    for k, v in helps:
        cv2.putText(
            canvas,
            f"{k} {v}",
            (panel_x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (150, 150, 150),
            1,
        )
        line_y += 25

    return canvas


def light_color(light):
    return {
        "daylight": (100, 200, 255),
        "twilight": (100, 165, 255),
        "night": (255, 150, 100),
    }.get(light, (200, 200, 200))


def label_color(label):
    return {
        "ok": (100, 255, 100),
        "problem": (100, 100, 255),
        "unclear": (150, 150, 150),
        "not_pole": (100, 100, 100),
    }.get(label, (200, 200, 200))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./crops/crops_index.csv"),
        help="generate_crops.py çıktısı",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("./crops"),
        help="ok/, problem/, unclear/ klasörlerinin oluşturulacağı yer",
    )
    parser.add_argument(
        "--filter-light",
        default=None,
        help="Show only crops with this light condition. "
        "Comma-separated for multiple values: --filter-light twilight,night",
    )
    parser.add_argument(
        "--filter-city", default=None, help="Sadece bu şehirdeki crop'ları göster"
    )
    parser.add_argument(
        "--window-width", type=int, default=1400, help="Pencere genişliği"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Hata: {args.input} bulunamadı", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"{len(df)} crop yüklendi")

    # 'label' kolonu yoksa ekle
    if "label" not in df.columns:
        df["label"] = ""

    # Filtre
    if args.filter_light:
        # Comma-separated değerleri ayır
        light_values = [v.strip() for v in args.filter_light.split(",")]
        df = df[df["light"].isin(light_values)].reset_index(drop=True)
        print(f"Filtre: light in {light_values} -> {len(df)} crop")
    if args.filter_city:
        df = df[df["city"] == args.filter_city].reset_index(drop=True)
        print(f"Filtre: city={args.filter_city} -> {len(df)} crop")

    if len(df) == 0:
        print("Filtre sonrası hiç crop kalmadı.")
        sys.exit(0)

    # Sınıf klasörlerini hazırla
    for cls in ["ok", "problem", "unclear", "not_pole"]:
        (args.output_base / cls).mkdir(parents=True, exist_ok=True)

    # Etiketlenmemişlerin indekslerini bul (NaN VEYA boş string)
    todo_indices = df.index[df["label"].isna() | (df["label"] == "")].tolist()
    if not todo_indices:
        print("Tüm crop'lar zaten etiketlenmiş.")
        return
    print(f"Etiketlenecek: {len(todo_indices)}")

    # Pencere
    win = "Triage"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.window_width, 720)

    history = []  # geri alma için
    cursor = 0  # todo_indices içindeki konum
    counts = {"ok": 0, "problem": 0, "unclear": 0, "not_pole": 0, "skipped": 0}

    while 0 <= cursor < len(todo_indices):
        idx = todo_indices[cursor]
        row = df.iloc[idx]

        info = {
            "crop_id": row["crop_id"],
            "light": row.get("light", "?"),
            "city": row.get("city", "?"),
            "time": (
                str(row.get("timestamp", "?"))[11:19]
                if pd.notna(row.get("timestamp"))
                else "?"
            ),
            "pole_rank": row.get("pole_rank", "?"),
            "bbox_conf": row.get("bbox_conf"),
            "progress": f"{cursor+1} / {len(todo_indices)}  "
            f"(ok:{counts['ok']} prob:{counts['problem']} "
            f"unc:{counts['unclear']} not:{counts['not_pole']})",
            "current_label": (
                row["label"] if pd.notna(row["label"]) and row["label"] != "" else None
            ),
        }

        canvas = render_frame(row["crop_path"], info, args.window_width)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF

        # Pencere kapatıldıysa (X butonu, yön tuşu kazası vs.) güvenli çık
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                print("Pencere kapatıldı, ilerleme kaydediliyor...")
                break
        except cv2.error:
            print("Pencere kapatıldı, ilerleme kaydediliyor...")
            break

        if key in LABELS:
            label = LABELS[key]
            # Crop'u sınıf klasörüne taşı
            src = Path(row["crop_path"])
            dst = args.output_base / label / src.name
            try:
                shutil.move(str(src), str(dst))
                df.at[idx, "label"] = label
                df.at[idx, "crop_path"] = str(dst.resolve())
                counts[label] += 1
                history.append((idx, label, str(src), str(dst)))
            except Exception as e:
                print(f"HATA: {src} -> {dst} taşınamadı ({e})", file=sys.stderr)
                continue
            cursor += 1

        elif key in (KEY_S, KEY_S_UPPER):
            # CSV'ye 'skipped' yaz — tekrar açıldığında bu crop görünmesin
            df.at[idx, "label"] = "skipped"
            counts["skipped"] += 1
            history.append((idx, "skipped", None, None))  # geri alma için kayıt
            cursor += 1

        elif key in (KEY_BACKSPACE, KEY_BACKSPACE_ALT):
            if not history:
                continue
            # Son işlemi geri al
            last_idx, last_label, last_src, last_dst = history.pop()
            try:
                if last_label == "skipped":
                    # Skip'in geri alınması: dosya taşınmadı, sadece label'i temizle
                    df.at[last_idx, "label"] = ""
                    counts["skipped"] -= 1
                else:
                    # Etiketleme geri alma: dosyayı images/'a taşı, label temizle
                    shutil.move(last_dst, last_src)
                    df.at[last_idx, "label"] = ""
                    df.at[last_idx, "crop_path"] = last_src
                    counts[last_label] -= 1
                # Cursor'u geri al
                if last_idx in todo_indices:
                    cursor = todo_indices.index(last_idx)
            except Exception as e:
                print(f"HATA: geri alma başarısız ({e})", file=sys.stderr)

        elif key in (KEY_Q, KEY_Q_UPPER, KEY_ESC):
            break

        elif key in (KEY_H, KEY_H_UPPER):
            print()
            print("Klavye:")
            print("  O = ok    P = problem    U = unclear    N = not_pole")
            print("  S = skip    Backspace = geri al")
            print("  Q = çık")

    cv2.destroyAllWindows()

    # CSV'yi kaydet
    df.to_csv(args.input, index=False)

    # Özet
    print()
    print(
        f"Etiketlenen: {sum(counts[c] for c in ['ok','problem','unclear','not_pole'])}"
    )
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"CSV güncellendi: {args.input}")


if __name__ == "__main__":
    main()
