import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from formcoach.classifier import MODEL_PATH, NO_FEET_FEATURES, FormClassifier, Scores, assign_clips, rule_baseline
from formcoach.features import FRAME_FEATURES, frame_features


def test_clips_split_on_scene_change():
    """Похожие подряд идущие кадры — один клип; резкая смена картинки — новый клип."""
    df = pd.DataFrame({"split": ["train"] * 4, "label": ["good"] * 4, "idx": [1, 2, 3, 4]})
    dark, bright = np.zeros((16, 16)), np.full((16, 16), 200.0)
    clips = assign_clips(df, np.array([dark, dark + 1, bright, bright - 1]))
    assert clips[0] == clips[1] != clips[2] == clips[3]


def test_clips_never_cross_classes():
    df = pd.DataFrame({"split": ["train"] * 2, "label": ["good", "bad_back"], "idx": [1, 1]})
    clips = assign_clips(df, np.zeros((2, 16, 16)))
    assert clips[0] != clips[1]


def test_rule_baseline():
    X = pd.DataFrame({"torso_lean": [10, 60, 10], "heel_above_toe": [0.0, 0.0, 0.5]})
    assert list(rule_baseline(X)) == ["good", "bad_back", "bad_heel"]


def test_scores_macro_f1_ignores_absent_classes():
    s = Scores.of(["good", "bad_back"], ["good", "bad_back"])
    assert s.accuracy == 1.0 and s.f1_macro == 1.0


def test_frame_features_upright_person():
    pts = np.zeros((33, 4))
    pts[:, 3] = 1.0
    # человек стоит прямо: плечо над тазом над коленом над лодыжкой (y вниз)
    for i, y in [(11, 0), (12, 0), (23, 100), (24, 100), (25, 200), (26, 200), (27, 300), (28, 300)]:
        pts[i, :2] = [50, y]
    pts[[29, 30], :2] = [40, 310]  # пятки
    pts[[31, 32], :2] = [80, 310]  # носки
    f = frame_features(pts)
    assert f["knee_angle"] == pytest.approx(180)
    assert f["torso_lean"] == pytest.approx(0)
    assert f["thigh_angle"] == pytest.approx(90)
    assert f["heel_above_toe"] == pytest.approx(0)
    assert set(f) == set(FRAME_FEATURES)


def _toy_classifier():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(FRAME_FEATURES))), columns=FRAME_FEATURES)
    full = LogisticRegression().fit(X, ["good", "bad_back", "bad_heel"] * 20)
    no_feet = LogisticRegression().fit(X[NO_FEET_FEATURES], ["good", "bad_back"] * 30)
    return FormClassifier(full, "toy", no_feet)


def test_fallback_model_when_feet_hidden():
    clf = _toy_classifier()
    feats = dict.fromkeys(FRAME_FEATURES, 0.0)
    full = clf.predict_proba(feats, feet_visible=True)
    no_feet = clf.predict_proba(feats, feet_visible=False)
    assert sum(full.values()) == pytest.approx(1)
    assert no_feet["bad_heel"] is None  # пятки без стоп не оцениваем
    assert no_feet["good"] + no_feet["bad_back"] == pytest.approx(1)


def test_nan_features_rejected():
    feats = dict.fromkeys(FRAME_FEATURES, 0.0) | {"torso_lean": float("nan")}
    with pytest.raises(ValueError):
        _toy_classifier().predict_proba(feats)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="модель ещё не обучена")
def test_saved_model_flags_extreme_lean():
    clf = FormClassifier.load()
    feats = {"knee_angle": 80, "hip_angle": 40, "torso_lean": 70, "thigh_angle": 15, "shin_lean": 25, "ankle_angle": 90, "heel_above_toe": 0.0}
    p = clf.predict_proba(feats)
    assert max(p, key=p.get) == "bad_back"


def _standing_landmarks(n=10, foot_vis=1.0):
    pts = np.zeros((33, 4))
    pts[:, 3] = 1.0
    for i, y in [(11, 0), (12, 0), (23, 100), (24, 100), (25, 200), (26, 200), (27, 300), (28, 300)]:
        pts[i, :2] = [50, y]
    pts[[29, 30], :2] = [40, 310]
    pts[[31, 32], :2] = [80, 310]
    pts[[29, 30, 31, 32], 3] = foot_vis
    return np.repeat(pts[None], n, axis=0)


def test_classify_rep_switches_to_no_feet_model():
    from formcoach.classifier import classify_rep

    clf = _toy_classifier()
    seen = classify_rep(clf, _standing_landmarks(foot_vis=0.9), bottom=5)
    hidden = classify_rep(clf, _standing_landmarks(foot_vis=0.1), bottom=5)
    assert not seen.no_feet and seen.probs["bad_heel"] is not None
    assert hidden.no_feet and hidden.probs["bad_heel"] is None
    assert seen.frames_used == 7  # bottom ± 3 кадра


def test_classify_rep_none_when_person_missing():
    from formcoach.classifier import classify_rep

    lm = np.full((10, 33, 4), np.nan)
    assert classify_rep(_toy_classifier(), lm, bottom=5) is None
