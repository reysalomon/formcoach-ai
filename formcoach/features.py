"""Признаки техники: из координат суставов считаем углы и их скорость.

Эти числа — "наблюдения" для HMM (hmm.py) и входы для классификатора.
"""

import numpy as np

from . import pose as P


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
      knee_vel    — скорость изменения угла колена, град/сек (<0 — опускается)
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
    knee_vel = np.gradient(knee_angle) * seq.fps

    # Подъём пятки: насколько пятка выше носка, в долях длины голени
    # (делим на голень, чтобы не зависеть от расстояния до камеры).
    # Вычитаем медиану: большую часть видео пятки стоят на полу — это "ноль".
    shin = np.nanmedian(np.linalg.norm(xy[:, knee] - xy[:, ankle], axis=-1))
    heel_above_toe = (xy[:, foot, 1] - xy[:, heel, 1]) / max(shin, 1e-6)
    heel_lift = smooth(heel_above_toe - np.nanmedian(heel_above_toe), smooth_window)

    return {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_lean": torso_lean,
        "knee_vel": knee_vel,
        "heel_lift": heel_lift,
    }
