import numpy as np
import pytest

from formcoach.agent import Percept, SquatCoachAgent, SquatRules
from formcoach.hmm import BOTTOM, DOWN, TOP, UP, segment_reps
from formcoach.pipeline import analyze_squat
from formcoach.synthetic import GOOD, HEELS_UP, LEANING, SHALLOW, TOO_FAST, WALKING, RepSpec, synthetic_session

FPS = 30.0


def run(reps, rules=None):
    return analyze_squat(synthetic_session(reps, fps=FPS), FPS, rules=rules)


def test_good_reps_have_no_errors():
    a = run([GOOD, GOOD, GOOD])
    assert len(a.reps) == 3
    assert all(r.good for r in a.reps)
    assert all(f.level == "ok" for f in a.feedback)


@pytest.mark.parametrize(
    "spec, code",
    [(SHALLOW, "shallow"), (LEANING, "lean"), (HEELS_UP, "heels"), (TOO_FAST, "fast")],
)
def test_each_error_detected(spec, code):
    a = run([GOOD, spec, GOOD])
    assert len(a.reps) == 3
    assert a.reps[0].good and a.reps[2].good
    assert a.reps[1].errors == [code]


def test_live_hint_comes_before_rep_ends():
    """Подсказка про наклон должна прийти ВО ВРЕМЯ повторения, а не только в итогах."""
    a = run([LEANING])
    live = [f for f in a.feedback if f.message == "Держи спину ровнее"]
    assert len(live) == 1
    assert live[0].frame < a.reps[0].end


def test_dangerous_lean_is_danger_level():
    a = run([RepSpec(lean=75)])
    assert any(f.level == "danger" for f in a.feedback)


def test_rules_are_configurable():
    """Если ослабить порог глубины, неглубокий присед перестаёт быть ошибкой."""
    a = run([SHALLOW], rules=SquatRules(max_thigh_angle=60))
    assert a.reps[0].good


def test_agent_agrees_with_hmm_segmentation():
    a = run([GOOD, SHALLOW, LEANING, GOOD])
    hmm_reps = segment_reps(a.phases, a.features["knee_angle"])
    assert [(r.start, r.end) for r in hmm_reps] == [(r.start, r.end) for r in a.reps]


def test_noise_blip_is_not_a_rep():
    agent = SquatCoachAgent()
    phases = [TOP] * 10 + [DOWN, UP] + [TOP] * 10
    for t, ph in enumerate(phases):
        agent.step(Percept(t, ph, 160.0, 10.0, 0.0))
    assert agent.reps == []


def test_unfinished_rep_not_counted():
    agent = SquatCoachAgent()
    for t, ph in enumerate([TOP] * 5 + [DOWN] * 20 + [BOTTOM] * 10):
        agent.step(Percept(t, ph, 90.0, 20.0, 0.0))
    assert agent.reps == []


def test_too_short_input_rejected():
    with pytest.raises(ValueError):
        analyze_squat({"knee_angle": np.array([160.0]), "knee_vel": np.array([0.0])}, FPS)


def test_tiny_dip_is_not_a_rep():
    """Колено согнулось всего на 15° — это не присед (так было в начале реального видео)."""
    a = run([RepSpec(depth=150)])
    assert a.reps == []


def test_heels_not_judged_when_feet_invisible():
    f = synthetic_session([HEELS_UP], fps=FPS)
    f["heel_lift"][:] = np.nan  # стопы не видно
    a = analyze_squat(f, FPS)
    assert len(a.reps) == 1 and a.reps[0].good
    assert np.isnan(a.reps[0].max_heel)


def test_walking_is_not_a_rep():
    """Колено при шагах сгибается, но таз не опускается и стопы смещаются — не повторение."""
    a = run([GOOD, WALKING, GOOD])
    assert len(a.reps) == 2 and all(r.good for r in a.reps)
    assert any(f.level == "info" and "ходьб" in f.message for f in a.feedback)
