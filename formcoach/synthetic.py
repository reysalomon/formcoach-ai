"""Синтетические приседания: признаки, сгенерированные формулой, без видео.

Нужны для тестов и демо-режима: можно задать ошибку (неглубоко, наклон, пятки,
слишком быстро) и проверить, что HMM и агент её находят.
"""

from dataclasses import dataclass

import numpy as np

from .features import smooth


@dataclass
class RepSpec:
    depth: float = 85.0  # минимальный угол колена
    lean: float = 20.0  # максимальный наклон корпуса в нижней точке
    heel: float = 0.0  # максимальный подъём пятки (доля голени)
    seconds: float = 2.0  # длительность повторения
    walk: bool = False  # не присед, а шаги: таз не опускается, стопы смещаются


GOOD = RepSpec()
SHALLOW = RepSpec(depth=130)
LEANING = RepSpec(lean=40)
HEELS_UP = RepSpec(heel=0.2)
TOO_FAST = RepSpec(seconds=0.6)
WALKING = RepSpec(depth=130, seconds=0.8, walk=True)


def synthetic_session(
    reps: list[RepSpec], fps: float = 30.0, top: float = 165.0, pause_sec: float = 1.0, noise: float = 1.5, seed: int = 0
) -> dict[str, np.ndarray]:
    """Склеивает повторения с паузами стоя и возвращает признаки как squat_features()."""
    rng = np.random.default_rng(seed)
    n_pause = int(pause_sec * fps)
    knee, lean, heel = [np.full(n_pause, top)], [np.full(n_pause, 10.0)], [np.zeros(n_pause)]
    thigh = [np.full(n_pause, 85.0)]
    hip, feet = [np.full(n_pause, 4.0)], [np.zeros(n_pause)]
    foot_x = 0.0
    for r in reps:
        t = np.linspace(0, 2 * np.pi, max(int(r.seconds * fps), 3))
        k = (1 - np.cos(t)) / 2  # 0 -> 1 -> 0: плавно вниз и обратно
        knee += [top - (top - r.depth) * k, np.full(n_pause, top)]
        lean += [10 + (r.lean - 10) * k, np.full(n_pause, 10.0)]
        heel += [r.heel * k, np.zeros(n_pause)]
        # бедро: 85° стоя; при колене 85° — почти параллельно полу (10°), при 130° — 55°
        thigh += [85 - (85 - (r.depth - 75)) * k, np.full(n_pause, 85.0)]
        if r.walk:
            hip += [np.full(len(t), 4.0), np.full(n_pause, 4.0)]
            feet += [foot_x + np.linspace(0, 1, len(t)), np.full(n_pause, foot_x + 1)]
            foot_x += 1
        else:
            # таз опускается тем сильнее, чем глубже присед (~0.7 голени при 85°)
            hip += [4.0 + 0.7 * (top - r.depth) / 80 * k, np.full(n_pause, 4.0)]
            feet += [np.full(len(t), foot_x), np.full(n_pause, foot_x)]

    knee_angle = smooth(np.concatenate(knee) + rng.normal(0, noise, sum(map(len, knee))), 5)
    # сглаживаем так же, как squat_features() делает с настоящим видео
    torso_lean = smooth(np.concatenate(lean) + rng.normal(0, noise, len(knee_angle)), 5)
    heel_lift = np.concatenate(heel) + rng.normal(0, 0.005, len(knee_angle))
    return {
        "knee_angle": knee_angle,
        "hip_angle": knee_angle - 10,  # грубое приближение, в синтетике не важно
        "torso_lean": torso_lean,
        "thigh_angle": smooth(np.concatenate(thigh) + rng.normal(0, noise, len(knee_angle)), 5),
        "knee_vel": np.gradient(knee_angle) * fps,
        "heel_lift": heel_lift,
        "hip_y": np.concatenate(hip),
        "ankle_lx": np.concatenate(feet),
        "ankle_rx": np.concatenate(feet),
    }
