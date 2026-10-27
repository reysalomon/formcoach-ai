"""Датчик позы: видео -> координаты 33 точек тела на каждом кадре.

Это НЕ одна из трёх AI-тем проекта: MediaPipe здесь используется как готовый
"сенсор" (как камера с распознаванием суставов). Вся логика анализа техники
написана нами в features.py, hmm.py, agent.py и classifier.py.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"

# Номера нужных нам точек в модели MediaPipe Pose (всего их 33)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


@dataclass
class PoseSequence:
    """Результат обработки видео.

    landmarks: массив (кадры, 33, 4) — x и y в пикселях, z (глубина), visibility.
               Если человек на кадре не найден, строка заполнена NaN.
    fps:       кадров в секунду исходного видео.
    """

    landmarks: np.ndarray
    fps: float

    @property
    def n_frames(self) -> int:
        return len(self.landmarks)

    @property
    def detected_ratio(self) -> float:
        """Доля кадров, на которых человек найден."""
        if self.n_frames == 0:
            return 0.0
        return float(np.mean(~np.isnan(self.landmarks[:, 0, 0])))


def extract_pose(video_path: str | Path, max_frames: int | None = None) -> PoseSequence:
    """Прогоняет видео через MediaPipe и возвращает точки тела по кадрам."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Нет файла модели {MODEL_PATH}. Смотри README, раздел установки.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео (формат не поддерживается?): {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    frames = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(i * 1000 / fps))

            if result.pose_landmarks:
                pts = result.pose_landmarks[0]
                # MediaPipe даёт x, y в долях кадра (0..1) — переводим в пиксели,
                # иначе углы на не квадратном видео будут искажены.
                frames.append([[p.x * w, p.y * h, p.z * w, p.visibility] for p in pts])
            else:
                frames.append(np.full((33, 4), np.nan))
            i += 1
    cap.release()

    if not frames:
        raise ValueError("В видео нет ни одного кадра.")
    return PoseSequence(landmarks=np.asarray(frames, dtype=float), fps=float(fps))


# Линии скелета для рисования: пары номеров точек
SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_SHOULDER, R_HIP), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]


def read_frame(video_path: str | Path, index: int) -> np.ndarray | None:
    """Достаёт один кадр видео (в RGB) по номеру; None, если кадра нет."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def draw_skeleton(rgb: np.ndarray, points: np.ndarray, color=(42, 120, 214)) -> np.ndarray:
    """Рисует скелет поверх кадра. points: (33, 4) — одна строка из PoseSequence.landmarks."""
    out = rgb.copy()
    if np.isnan(points[0, 0]):
        return out
    thick = max(2, out.shape[1] // 300)
    for a, b in SKELETON:
        pa, pb = points[a, :2].astype(int), points[b, :2].astype(int)
        cv2.line(out, tuple(pa), tuple(pb), color, thick, cv2.LINE_AA)
    for i in {i for pair in SKELETON for i in pair}:
        cv2.circle(out, tuple(points[i, :2].astype(int)), thick * 2, (255, 255, 255), -1, cv2.LINE_AA)
    return out
