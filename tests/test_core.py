import numpy as np
import pytest

from formcoach.features import joint_angle, lean_from_vertical, smooth
from formcoach.hmm import BOTTOM, DOWN, TOP, UP, GaussianHMM, segment_reps, squat_hmm


def synthetic_squats(n_reps=3, depth=85.0, top=165.0, fps=30, rep_sec=2.0, pause_sec=1.0, noise=2.0, seed=0):
    """Синтетический угол колена: пауза стоя, затем плавный спуск-подъём (косинус), и так n раз."""
    rng = np.random.default_rng(seed)
    pause = np.full(int(pause_sec * fps), top)
    t = np.linspace(0, 2 * np.pi, int(rep_sec * fps))
    rep = top - (top - depth) * (1 - np.cos(t)) / 2
    angle = np.concatenate([pause] + [np.concatenate([rep, pause]) for _ in range(n_reps)])
    angle = angle + rng.normal(0, noise, len(angle))
    vel = np.gradient(smooth(angle, 5)) * fps
    return angle, np.column_stack([angle, vel])


# ---------- features ----------

def test_joint_angle_right_angle():
    assert joint_angle(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1])) == pytest.approx(90)


def test_joint_angle_straight_leg():
    assert joint_angle(np.array([0.0, 0]), np.array([0.0, 1]), np.array([0.0, 2])) == pytest.approx(180)


def test_lean_vertical_and_45():
    # плечо прямо над тазом (y вниз) -> 0 градусов
    assert lean_from_vertical(np.array([0.0, 0]), np.array([0.0, 10])) == pytest.approx(0)
    assert lean_from_vertical(np.array([10.0, 0]), np.array([0.0, 10])) == pytest.approx(45)


def test_smooth_fills_nan():
    out = smooth(np.array([1.0, np.nan, 3.0]), window=1)
    assert out.tolist() == [1.0, 2.0, 3.0]


# ---------- HMM ----------

def test_counts_three_reps():
    angle, X = synthetic_squats(n_reps=3)
    path, _ = squat_hmm().viterbi(X)
    reps = segment_reps(path, angle)
    assert len(reps) == 3
    assert all(r.reached_bottom for r in reps)


def test_phase_order_is_valid():
    """Фазы идут только по кругу ВЕРХ->СПУСК->(НИЗ)->ПОДЪЁМ->ВЕРХ, без скачков назад."""
    _, X = synthetic_squats(n_reps=2)
    path, _ = squat_hmm().viterbi(X)
    allowed = {(TOP, DOWN), (DOWN, BOTTOM), (DOWN, UP), (BOTTOM, UP), (UP, TOP)}
    for a, b in zip(path[:-1], path[1:]):
        assert a == b or (a, b) in allowed


def test_shallow_squat_counted_but_not_deep():
    angle, X = synthetic_squats(n_reps=2, depth=135)
    path, _ = squat_hmm().viterbi(X)
    reps = segment_reps(path, angle)
    assert len(reps) == 2
    assert not any(r.reached_bottom for r in reps)


def test_standing_still_gives_zero_reps():
    angle = np.full(150, 165.0) + np.random.default_rng(1).normal(0, 2, 150)
    X = np.column_stack([angle, np.zeros(150)])
    path, _ = squat_hmm().viterbi(X)
    assert segment_reps(path, angle) == []


def test_squat_video_more_likely_than_standing():
    """Прямой алгоритм: видео с приседаниями должно лучше объясняться моделью, чем шум."""
    hmm = squat_hmm()
    _, X = synthetic_squats(n_reps=2)
    rng = np.random.default_rng(2)
    noise = np.column_stack([rng.uniform(60, 180, len(X)), rng.uniform(-300, 300, len(X))])
    assert hmm.log_likelihood(X) > hmm.log_likelihood(noise)


def test_viterbi_score_not_above_total_likelihood():
    """Вероятность одного лучшего пути <= сумме вероятностей всех путей."""
    hmm = squat_hmm()
    _, X = synthetic_squats(n_reps=1)
    _, best = hmm.viterbi(X)
    assert best <= hmm.log_likelihood(X) + 1e-9


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        squat_hmm(stay_prob=1.0)
    with pytest.raises(ValueError):
        squat_hmm(top_angle=90, bottom_angle=120)
    with pytest.raises(ValueError):
        GaussianHMM(start=[1.0], trans=[[0.5]], means=[[0.0]], stds=[[1.0]])
