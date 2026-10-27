"""Датасет Binus (Zenodo 17558630) -> таблица признаков позы для классификатора.

Запуск: python scripts/build_binus_features.py
Вход:   data/external/binus/{train,test}/<класс>/*.jpg
Выход:  data/processed/binus_features.csv  — по строке на фото
        data/processed/binus_thumbs.npz    — миниатюры 16x16 (чтобы найти кадры одного видео)
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formcoach import pose as P  # noqa: E402
from formcoach.features import frame_features  # noqa: E402

SRC = ROOT / "data" / "external" / "binus"
OUT = ROOT / "data" / "processed"
LABELS = {"good": "good", "bad back": "bad_back", "bad heel": "bad_heel"}


def file_index(name: str) -> int:
    """Номер кадра из имени: "Good (12).jpg" -> 12, "frame_339.jpg" -> 100339 (отдельная серия)."""
    if m := re.search(r"\((\d+)\)", name):
        return int(m.group(1))
    if m := re.search(r"frame_(\d+)", name):
        return 100_000 + int(m.group(1))
    raise ValueError(f"Не понимаю номер в имени файла: {name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*/*/*.jpg"))
    if not files:
        raise SystemExit(f"Нет фото в {SRC}. Скачай датасет (см. README).")

    rows, thumbs = [], []
    with P.ImagePoseDetector() as det:
        for i, f in enumerate(files):
            bgr = cv2.imread(str(f))
            if bgr is None:
                print("не читается:", f)
                continue
            points = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            row = {
                "split": f.parent.parent.name,
                "label": LABELS[f.parent.name.lower()],
                "file": str(f.relative_to(SRC)),
                "idx": file_index(f.name),
                "width": bgr.shape[1],
                "detected": points is not None,
            }
            if points is not None:
                row |= frame_features(points)
                # надёжность лучше видимой ноги (таз, колено, лодыжка)
                left = points[[P.L_HIP, P.L_KNEE, P.L_ANKLE], 3].min()
                right = points[[P.R_HIP, P.R_KNEE, P.R_ANKLE], 3].min()
                row["min_visibility"] = float(max(left, right))
            rows.append(row)
            thumbs.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA))
            if i % 200 == 0:
                print(f"{i}/{len(files)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "binus_features.csv", index=False)
    np.savez_compressed(OUT / "binus_thumbs.npz", thumbs=np.array(thumbs), files=df["file"].to_numpy())
    print(f"Готово: {len(df)} фото, поза найдена на {df['detected'].mean():.1%}")


if __name__ == "__main__":
    main()
