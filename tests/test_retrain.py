import numpy as np
import pandas as pd

from formcoach.features import FRAME_FEATURES, frame_features
from formcoach.retrain import fit_classifier, leave_one_video_out, own_class, own_frames, predict_reps

LR = "Логистическая регрессия"


def _pose(lean_deg: float, rng) -> np.ndarray:
    """Человек в приседе с наклоном корпуса lean_deg (небольшой случайный шум)."""
    pts = np.zeros((33, 4))
    pts[:, 3] = 1.0
    hip, knee, ankle = np.array([100.0, 200]), np.array([180.0, 210]), np.array([160.0, 300])
    a = np.radians(lean_deg + rng.normal(0, 2))
    shoulder = hip + 100 * np.array([np.sin(a), -np.cos(a)])
    for i, p in [(11, shoulder), (12, shoulder), (23, hip), (24, hip), (25, knee), (26, knee), (27, ankle), (28, ankle)]:
        pts[i, :2] = p + rng.normal(0, 1, 2)
    pts[[29, 30], :2] = ankle + [-15, 10]
    pts[[31, 32], :2] = ankle + [30, 10]
    return pts


def _fake_binus(rng) -> pd.DataFrame:
    """Как в Binus: правильно — почти без наклона, «спина» — преувеличенный наклон ~70°."""
    rows = [{**frame_features(_pose(lean, rng)), "label": label}
            for label, lean in [("good", 5), ("bad_back", 70)] for _ in range(100)]
    return pd.DataFrame(rows)


def _own_video(vid: str, lean: float, rng, n_reps=4, frames_per_rep=20):
    """Своё видео: каждое повторение — 20 кадров, нижняя точка в середине."""
    lm = np.stack([_pose(lean, rng) for _ in range(n_reps * frames_per_rep)])
    reps = pd.DataFrame({"video_id": vid, "rep": range(1, n_reps + 1),
                         "bottom_frame": [i * frames_per_rep + 10 for i in range(n_reps)], "label": "bad_back"})
    return reps, lm


def test_own_class_mapping():
    row = dict(good=False, shallow=False, lean=False, heels=False, fast=False)
    assert own_class(row | {"good": True}) == "good"
    assert own_class(row | {"lean": True, "shallow": True}) == "bad_back"
    assert own_class(row | {"heels": True}) == "bad_heel"
    assert own_class(row | {"lean": True, "heels": True}) == "bad_back"
    assert own_class(row | {"shallow": True}) is None  # глубину оценивает HMM, не классификатор


def test_own_frames_window_and_missing_person():
    rng = np.random.default_rng(0)
    reps, lm = _own_video("v1", 30, rng, n_reps=1)
    lm[9] = np.nan  # на одном кадре человек не найден
    frames = own_frames(reps, {"v1": lm})
    assert len(frames) == 6  # bottom±3 = 7 кадров, минус пропавший
    assert set(FRAME_FEATURES) <= set(frames.columns) and frames["feet"].all()


def test_own_videos_teach_mild_lean():
    """Binus знает только сильный наклон; после дообучения модель замечает мягкий (35°) на НОВОМ видео."""
    rng = np.random.default_rng(0)
    binus = _fake_binus(rng)
    r1, lm1 = _own_video("v1", 35, rng)
    r2, lm2 = _own_video("v2", 35, rng)
    reps, landmarks = pd.concat([r1, r2], ignore_index=True), {"v1": lm1, "v2": lm2}
    frames = own_frames(reps, landmarks)

    result = leave_one_video_out(binus, reps, landmarks, frames, LR, repeat=10)
    assert result["before"].accuracy == 0.0  # без своих данных — «всё правильно»
    assert result["after"].accuracy == 1.0


def test_repeat_one_vs_many():
    """Свои кадры с весом 1 тонут среди Binus; повтор усиливает их влияние."""
    rng = np.random.default_rng(1)
    binus = _fake_binus(rng)
    reps, lm = _own_video("v1", 35, rng, n_reps=1)
    frames = own_frames(reps, {"v1": lm})
    weak = fit_classifier(binus, frames, LR, repeat=1)
    strong = fit_classifier(binus, frames, LR, repeat=30)
    p = lambda clf: clf.model.predict_proba(frames[FRAME_FEATURES])[:, list(clf.model.classes_).index("bad_back")].mean()  # noqa: E731
    assert p(strong) > p(weak)
    assert predict_reps(strong, reps, {"v1": lm}) == ["bad_back"]
