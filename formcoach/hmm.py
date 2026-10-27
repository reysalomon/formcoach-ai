"""Неделя 7 — Hidden Markov Model: распознаём фазы приседания и считаем повторения.

Скрытые состояния (их не видно напрямую):  ВЕРХ -> СПУСК -> НИЗ -> ПОДЪЁМ -> ВЕРХ ...
Наблюдения (видим на каждом кадре):        угол колена и скорость его изменения.

Почему HMM, а не просто порог "угол < 100"? Порог дёргается от шума и не отличает
спуск от подъёма (угол одинаковый!). HMM учитывает и скорость, и то, что фазы идут
строго по порядку, поэтому выдаёт гладкую и логичную последовательность фаз.

Реализовано вручную: гауссовы вероятности наблюдений, алгоритм Витерби
(самая вероятная последовательность фаз) и прямой алгоритм (вероятность всего видео).
Всё считаем в логарифмах, чтобы произведение сотен маленьких чисел не стало нулём.
"""

from dataclasses import dataclass

import numpy as np

TOP, DOWN, BOTTOM, UP = 0, 1, 2, 3
STATE_NAMES = ["ВЕРХ", "СПУСК", "НИЗ", "ПОДЪЁМ"]


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """log(sum(exp(a))) без переполнения."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


@dataclass
class GaussianHMM:
    """HMM с гауссовыми наблюдениями (признаки считаем независимыми — диагональная ковариация).

    start: (S,)    вероятности начального состояния
    trans: (S, S)  trans[i, j] = P(следующее состояние j | текущее i)
    means: (S, D)  среднее значение каждого признака в каждом состоянии
    stds:  (S, D)  стандартное отклонение
    """

    start: np.ndarray
    trans: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.trans = np.asarray(self.trans, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.stds = np.asarray(self.stds, dtype=float)
        if not np.allclose(self.trans.sum(axis=1), 1.0):
            raise ValueError("Каждая строка матрицы переходов должна суммироваться в 1.")
        if not np.isclose(self.start.sum(), 1.0):
            raise ValueError("Начальные вероятности должны суммироваться в 1.")
        if np.any(self.stds <= 0):
            raise ValueError("Стандартные отклонения должны быть > 0.")

    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """log P(наблюдение x_t | состояние s) для всех t и s. X: (T, D) -> (T, S)."""
        X = np.asarray(X, dtype=float)[:, None, :]  # (T, 1, D)
        z = (X - self.means) / self.stds  # (T, S, D)
        log_pdf = -0.5 * z**2 - np.log(self.stds) - 0.5 * np.log(2 * np.pi)
        return log_pdf.sum(axis=2)

    def viterbi(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Самая вероятная последовательность состояний (динамическое программирование).

        delta[t, j] = лучший log-вероятность пути, который на кадре t пришёл в состояние j.
        back[t, j]  = из какого состояния мы туда пришли (чтобы потом восстановить путь).
        """
        log_e = self.log_emission(X)
        T, S = log_e.shape
        with np.errstate(divide="ignore"):
            log_a = np.log(self.trans)
            log_pi = np.log(self.start)

        delta = np.empty((T, S))
        back = np.zeros((T, S), dtype=int)
        delta[0] = log_pi + log_e[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_a  # scores[i, j]: из i в j
            back[t] = np.argmax(scores, axis=0)
            delta[t] = scores[back[t], np.arange(S)] + log_e[t]

        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(delta[-1].max())

    def log_likelihood(self, X: np.ndarray) -> float:
        """Прямой алгоритм: log P(все наблюдения | модель), по всем возможным путям."""
        log_e = self.log_emission(X)
        with np.errstate(divide="ignore"):
            log_a = np.log(self.trans)
            alpha = np.log(self.start) + log_e[0]
        for t in range(1, len(log_e)):
            alpha = logsumexp(alpha[:, None] + log_a, axis=0) + log_e[t]
        return float(logsumexp(alpha, axis=0))


def squat_hmm(
    top_angle: float = 165.0,
    bottom_angle: float = 85.0,
    stay_prob: float = 0.9,
    speed: float = 90.0,
) -> GaussianHMM:
    """HMM для приседания с понятными параметрами (их можно менять в интерфейсе).

    top_angle    — угол колена стоя
    bottom_angle — угол колена в нижней точке
    stay_prob    — вероятность остаться в той же фазе на следующем кадре
                   (чем больше, тем "ленивее" модель меняет фазу и тем меньше реагирует на шум)
    speed        — типичная скорость сгибания колена, град/сек
    """
    if not 0 < stay_prob < 1:
        raise ValueError("stay_prob должен быть строго между 0 и 1.")
    if bottom_angle >= top_angle:
        raise ValueError("Угол внизу должен быть меньше угла вверху.")

    mid = (top_angle + bottom_angle) / 2
    span = top_angle - bottom_angle
    move = 1 - stay_prob
    trans = np.array(
        [
            # в ВЕРХ    СПУСК      НИЗ          ПОДЪЁМ
            [stay_prob, move, 0.0, 0.0],  # из ВЕРХ
            [0.0, stay_prob, move * 0.8, move * 0.2],  # из СПУСК (0.2 — развернулся, не дойдя до низа)
            [0.0, 0.0, stay_prob, move],  # из НИЗ
            [move, 0.0, 0.0, stay_prob],  # из ПОДЪЁМ
        ]
    )
    #               угол колена              скорость
    means = [[top_angle, 0.0], [mid, -speed], [bottom_angle, 0.0], [mid, speed]]
    stds = [[span * 0.15, speed * 0.4], [span * 0.35, speed * 0.6], [span * 0.2, speed * 0.4], [span * 0.35, speed * 0.6]]
    return GaussianHMM(start=[0.97, 0.01, 0.01, 0.01], trans=trans, means=means, stds=stds)


@dataclass
class Rep:
    start: int  # кадр, где начался спуск
    end: int  # кадр, где снова встал
    bottom: int  # кадр с минимальным углом колена
    reached_bottom: bool  # прошёл ли через фазу НИЗ


def segment_reps(path: np.ndarray, knee_angle: np.ndarray, min_frames: int = 5) -> list[Rep]:
    """Режем последовательность фаз на повторения: ВЕРХ -> ... -> ВЕРХ = одно повторение."""
    reps = []
    start = None
    for t in range(1, len(path)):
        if path[t - 1] == TOP and path[t] != TOP:
            start = t
        elif start is not None and path[t - 1] != TOP and path[t] == TOP:
            if t - start >= min_frames:  # слишком короткие "повторения" — это шум
                seg = path[start:t]
                bottom = start + int(np.argmin(knee_angle[start:t]))
                reps.append(Rep(start, t, bottom, bool(np.any(seg == BOTTOM))))
            start = None
    return reps
