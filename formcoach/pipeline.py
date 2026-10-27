"""Вся цепочка анализа в одном месте: признаки -> HMM (фазы) -> агент (подсказки)."""

from dataclasses import dataclass

import numpy as np

from .agent import Feedback, RepReport, SquatCoachAgent, SquatRules, percepts_from
from .hmm import squat_hmm


@dataclass
class Analysis:
    features: dict[str, np.ndarray]
    phases: np.ndarray  # фаза HMM на каждом кадре
    log_likelihood: float  # насколько хорошо HMM объясняет видео (прямой алгоритм)
    reps: list[RepReport]
    feedback: list[Feedback]
    fps: float


def analyze_squat(features: dict, fps: float, hmm_params: dict | None = None, rules: SquatRules | None = None) -> Analysis:
    if len(features["knee_angle"]) < 2:
        raise ValueError("Слишком короткое видео для анализа.")
    hmm = squat_hmm(**(hmm_params or {}))
    X = np.column_stack([features["knee_angle"], features["knee_vel"]])
    phases, _ = hmm.viterbi(X)

    agent = SquatCoachAgent(rules, fps)
    feedback = agent.run(percepts_from(features, phases))
    return Analysis(features, phases, hmm.log_likelihood(X), agent.reps, feedback, fps)
