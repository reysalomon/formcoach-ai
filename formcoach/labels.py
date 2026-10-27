"""Своя разметка повторений: что пользователь на самом деле делал в каждом приседе.

Хранение:
  data/labels/own_reps.csv              — строка на повторение (метки + признаки позы)
  data/labels/landmarks/<video_id>.npy  — точки тела всего видео (чтобы пересчитать признаки позже)

video_id — md5 содержимого видео: одно и то же видео, загруженное повторно,
перезаписывает старую разметку, а не дублирует её.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import RepReport
from .features import FRAME_FEATURES, frame_features

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "labels"
LABELS_CSV = LABELS_DIR / "own_reps.csv"

# Варианты ответа: "good" нельзя совмещать с ошибками
LABEL_KEYS = ["good", "shallow", "lean", "heels", "fast"]
LABEL_RU = {"good": "правильно", "shallow": "неглубоко", "lean": "наклон", "heels": "пятки", "fast": "быстро"}


def validate(labels: list[str]) -> str | None:
    """Текст ошибки или None, если разметка повторения корректна."""
    if not labels:
        return "отметь хотя бы один вариант"
    if "good" in labels and len(labels) > 1:
        return "«правильно» нельзя совмещать с ошибками"
    unknown = set(labels) - set(LABEL_KEYS)
    if unknown:
        return f"неизвестные метки: {unknown}"
    return None


def load_labels(path: Path = LABELS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def labels_for_video(video_id: str, path: Path = LABELS_CSV) -> dict[int, list[str]]:
    """Уже сохранённая разметка видео: {номер повторения: [метки]}."""
    df = load_labels(path)
    if df.empty:
        return {}
    df = df[df["video_id"] == video_id]
    return {int(r["rep"]): [k for k in LABEL_KEYS if r[k]] for _, r in df.iterrows()}


def save_labels(
    video_id: str,
    video_name: str,
    fps: float,
    reps: list[RepReport],
    labels: dict[int, list[str]],
    landmarks: np.ndarray,
    path: Path = LABELS_CSV,
) -> int:
    """Сохраняет разметку повторений одного видео (перезаписывает прежнюю). Возвращает число строк."""
    for rep in reps:
        if err := validate(labels.get(rep.number, [])):
            raise ValueError(f"Повторение {rep.number}: {err}")

    rows = []
    for rep in reps:
        row = {
            "video_id": video_id, "video_name": video_name, "rep": rep.number,
            "start": rep.start, "end": rep.end, "bottom_frame": rep.bottom_frame, "fps": fps,
            **{k: k in labels[rep.number] for k in LABEL_KEYS},
            "min_thigh": rep.min_thigh, "min_knee": rep.min_knee, "max_lean": rep.max_lean,
            "max_heel": rep.max_heel, "seconds": rep.seconds,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        points = landmarks[rep.bottom_frame]
        row |= frame_features(points) if not np.isnan(points[0, 0]) else dict.fromkeys(FRAME_FEATURES, np.nan)
        rows.append(row)

    path = Path(path)
    (path.parent / "landmarks").mkdir(parents=True, exist_ok=True)
    np.save(path.parent / "landmarks" / f"{video_id}.npy", landmarks.astype(np.float32))

    old = load_labels(path)
    if not old.empty:
        old = old[old["video_id"] != video_id]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    return len(rows)
