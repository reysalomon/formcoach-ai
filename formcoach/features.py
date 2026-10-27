"""Признаки техники: из координат суставов считаем углы и их скорость.

Эти числа — "наблюдения" для HMM (hmm.py) и входы для классификатора.
"""

import numpy as np

from . import pose as P

FOOT_MIN_VISIBILITY = 0.5  # ниже — точки стопы считаем ненадёжными


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Угол ABC в вершине b, в градусах (0..180).

    a, b, c — массивы формы (..., 2) с координатами x, y.
    Формула: cos(угла) = (BA · BC) / (|BA| * |BC|).
    """
    ba = a - b
    bc = c - b
    # np.maximum защищает от деления на 0, если две точки совпали
    norms = np.maximum(np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1), 1e-9)
    cos = np.sum(ba * bc, axis=-1) / norms
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def lean_from_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Наклон отрезка bottom->top от вертикали, в градусах (0 = стоит ровно)."""
    v = top - bottom
    # В картинке ось y направлена вниз, поэтому "вверх" — это -y
    return np.degrees(np.arctan2(np.abs(v[..., 0]), -v[..., 1]))


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Скользящее среднее; пропуски (NaN) сначала заполняем соседними значениями."""
    x = np.asarray(x, dtype=float).copy()
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum() == 0:
        return x
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    if window <= 1 or len(x) < window:
        return x
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def squat_features(seq: P.PoseSequence, smooth_window: int = 5) -> dict[str, np.ndarray]:
    """Покадровые признаки приседания.

    Берём ту сторону тела (левую или правую), которую камера видит лучше.
    Возвращает словарь массивов длины n_frames:
      knee_angle  — угол в колене (180 = нога прямая, ~70-90 = глубокий присед)
      hip_angle   — угол в тазобедренном суставе
      torso_lean  — наклон корпуса от вертикали
      thigh_angle — угол бедра к полу (0 = параллельно полу) — по нему меряем глубину
      knee_vel    — скорость изменения угла колена, град/сек (<0 — опускается)
      heel_lift   — подъём пятки, доля голени (NaN — стопу плохо видно)
      hip_y       — высота таза, в длинах голени (ось y вниз: больше = ниже)
      ankle_lx, ankle_rx — положение стоп по горизонтали, в длинах голени
    hip_y и ankle_* нужны, чтобы отличить присед от ходьбы.
    """
    lm = seq.landmarks
    vis_left = np.nanmean(lm[:, [P.L_HIP, P.L_KNEE, P.L_ANKLE], 3])
    vis_right = np.nanmean(lm[:, [P.R_HIP, P.R_KNEE, P.R_ANKLE], 3])
    if np.isnan(vis_left) and np.isnan(vis_right):
        raise ValueError("Человек не найден ни на одном кадре.")
    if np.nan_to_num(vis_left) >= np.nan_to_num(vis_right):
        sh, hip, knee, ankle, heel, foot = P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_HEEL, P.L_FOOT
    else:
        sh, hip, knee, ankle, heel, foot = P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_HEEL, P.R_FOOT

    xy = lm[:, :, :2]
    knee_angle = smooth(joint_angle(xy[:, hip], xy[:, knee], xy[:, ankle]), smooth_window)
    hip_angle = smooth(joint_angle(xy[:, sh], xy[:, hip], xy[:, knee]), smooth_window)
    torso_lean = smooth(lean_from_vertical(xy[:, sh], xy[:, hip]), smooth_window)
    # Угол бедра к полу: 90 = стоит, 0 = бедро параллельно полу, <0 = ниже параллели.
    # Нужны только таз и колено — они видны надёжнее, чем лодыжки и стопы.
    thigh = xy[:, knee] - xy[:, hip]
    thigh_angle = smooth(np.degrees(np.arctan2(thigh[:, 1], np.abs(thigh[:, 0]))), smooth_window)
    knee_vel = np.gradient(knee_angle) * seq.fps

    # Подъём пятки: насколько пятка выше носка, в долях длины голени
    # (делим на голень, чтобы не зависеть от расстояния до камеры).
    # Вычитаем медиану: большую часть видео пятки стоят на полу — это "ноль".
    shin = np.nanmedian(np.linalg.norm(xy[:, knee] - xy[:, ankle], axis=-1))
    heel_above_toe = (xy[:, foot, 1] - xy[:, heel, 1]) / max(shin, 1e-6)
    heel_lift = smooth(heel_above_toe - np.nanmedian(heel_above_toe), smooth_window)
    # Стопы MediaPipe видит хуже всего: в нижней точке их часто "теряет", и точки
    # пятки/носка прыгают. Где стопу видно плохо — не оцениваем (NaN), а не врём.
    foot_vis = smooth(np.nanmin(lm[:, [heel, foot], 3], axis=1), smooth_window)
    heel_lift[foot_vis < FOOT_MIN_VISIBILITY] = np.nan

    return {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_lean": torso_lean,
        "thigh_angle": thigh_angle,
        "knee_vel": knee_vel,
        "heel_lift": heel_lift,
        "hip_y": smooth((xy[:, P.L_HIP, 1] + xy[:, P.R_HIP, 1]) / 2, smooth_window) / shin,
        "ankle_lx": smooth(xy[:, P.L_ANKLE, 0], smooth_window) / shin,
        "ankle_rx": smooth(xy[:, P.R_ANKLE, 0], smooth_window) / shin,
    }


# Признаки одного кадра — входы классификатора (неделя 8).
# Все — углы или отношения длин, поэтому не зависят от размера фото и расстояния до камеры.
FRAME_FEATURES = ["knee_angle", "hip_angle", "torso_lean", "thigh_angle", "shin_lean", "ankle_angle", "heel_above_toe"]


def frame_features(points: np.ndarray) -> dict[str, float]:
    """Признаки позы на одном кадре. points: (33, 4) из MediaPipe (x, y в пикселях)."""
    left = np.mean(points[[P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_HEEL, P.L_FOOT], 3])
    right = np.mean(points[[P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_HEEL, P.R_FOOT], 3])
    if left >= right:
        sh, hip, knee, ankle, heel, foot = P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_HEEL, P.L_FOOT
    else:
        sh, hip, knee, ankle, heel, foot = P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_HEEL, P.R_FOOT
    xy = points[:, :2]
    thigh = xy[knee] - xy[hip]
    shin = max(float(np.linalg.norm(xy[knee] - xy[ankle])), 1e-6)
    return {
        "knee_angle": float(joint_angle(xy[hip], xy[knee], xy[ankle])),
        "hip_angle": float(joint_angle(xy[sh], xy[hip], xy[knee])),
        "torso_lean": float(lean_from_vertical(xy[sh], xy[hip])),
        "thigh_angle": float(np.degrees(np.arctan2(thigh[1], abs(thigh[0])))),
        "shin_lean": float(lean_from_vertical(xy[knee], xy[ankle])),  # насколько колено ушло вперёд
        "ankle_angle": float(joint_angle(xy[knee], xy[ankle], xy[foot])),  # меняется, когда встаёшь на носки
        "heel_above_toe": float((xy[foot, 1] - xy[heel, 1]) / shin),  # >0 — пятка выше носка
    }


def feet_visible(points: np.ndarray, min_visibility: float = FOOT_MIN_VISIBILITY) -> bool:
    """Видны ли надёжно пятка и носок хотя бы одной ноги на кадре."""
    left = points[[P.L_HEEL, P.L_FOOT], 3].min()
    right = points[[P.R_HEEL, P.R_FOOT], 3].min()
    return bool(max(left, right) >= min_visibility)
