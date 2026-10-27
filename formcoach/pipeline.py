"""Вся цепочка анализа в одном месте: признаки -> HMM (фазы) -> агент (подсказки)."""

from dataclasses import dataclass

import numpy as np

from .agent import Feedback, RepReport, SquatCoachAgent, SquatRules, percepts_from
from .classifier import FormClassifier, RepVerdict, classify_rep
from .hmm import squat_hmm


@dataclass
class Analysis:
    features: dict[str, np.ndarray]
    phases: np.ndarray  # фаза HMM на каждом кадре
    log_likelihood: float  # насколько хорошо HMM объясняет видео (прямой алгоритм)
    reps: list[RepReport]
    feedback: list[Feedback]
    fps: float
    ml: list[RepVerdict | None]  # мнение классификатора (неделя 8) по каждому повторению


def analyze_squat(
    features: dict,
    fps: float,
    hmm_params: dict | None = None,
    rules: SquatRules | None = None,
    landmarks: np.ndarray | None = None,
    classifier: FormClassifier | None = None,
) -> Analysis:
    """landmarks и classifier — необязательные: без них (демо-режим) ML-оценки не будет."""
    if len(features["knee_angle"]) < 2:
        raise ValueError("Слишком короткое видео для анализа.")
    hmm = squat_hmm(**(hmm_params or {}))
    X = np.column_stack([features["knee_angle"], features["knee_vel"]])
    phases, _ = hmm.viterbi(X)

    agent = SquatCoachAgent(rules, fps)
    feedback = agent.run(percepts_from(features, phases))
    ml = [None] * len(agent.reps)
    if landmarks is not None and classifier is not None:
        ml = [classify_rep(classifier, landmarks, r.bottom_frame) for r in agent.reps]
    return Analysis(features, phases, hmm.log_likelihood(X), agent.reps, feedback, fps, ml)
