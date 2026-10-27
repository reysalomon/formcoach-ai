import numpy as np
import pytest

from formcoach.agent import RepReport
from formcoach.labels import labels_for_video, load_labels, save_labels, validate


def _rep(n, bottom):
    return RepReport(n, bottom - 10, bottom + 10, bottom, 90.0, 20.0, 30.0, 0.0, 1.5, True)


def _landmarks(n=50):
    lm = np.random.default_rng(0).uniform(0, 500, size=(n, 33, 4))
    lm[:, :, 3] = 1.0
    return lm


def test_validate():
    assert validate(["good"]) is None
    assert validate(["shallow", "lean"]) is None
    assert validate([]) is not None
    assert validate(["good", "lean"]) is not None  # правильно + ошибка — противоречие
    assert validate(["jump"]) is not None


def test_save_and_reload(tmp_path):
    path = tmp_path / "own.csv"
    reps = [_rep(1, 15), _rep(2, 35)]
    n = save_labels("vid1", "a.mp4", 30.0, reps, {1: ["lean"], 2: ["shallow", "heels"]}, _landmarks(), path)
    assert n == 2
    assert labels_for_video("vid1", path) == {1: ["lean"], 2: ["shallow", "heels"]}
    assert (tmp_path / "landmarks" / "vid1.npy").exists()
    df = load_labels(path)
    assert {"knee_angle", "torso_lean", "max_lean", "bottom_frame"} <= set(df.columns)


def test_resave_replaces_not_duplicates(tmp_path):
    path = tmp_path / "own.csv"
    reps = [_rep(1, 15)]
    save_labels("vid1", "a.mp4", 30.0, reps, {1: ["lean"]}, _landmarks(), path)
    save_labels("vid2", "b.mp4", 30.0, reps, {1: ["good"]}, _landmarks(), path)
    save_labels("vid1", "a.mp4", 30.0, reps, {1: ["shallow"]}, _landmarks(), path)  # исправил разметку
    df = load_labels(path)
    assert len(df) == 2
    assert labels_for_video("vid1", path) == {1: ["shallow"]}
    assert labels_for_video("vid2", path) == {1: ["good"]}


def test_invalid_labels_not_saved(tmp_path):
    path = tmp_path / "own.csv"
    with pytest.raises(ValueError, match="Повторение 2"):
        save_labels("vid1", "a.mp4", 30.0, [_rep(1, 15), _rep(2, 35)], {1: ["good"]}, _landmarks(), path)
    assert not path.exists()


def test_missing_person_on_bottom_frame(tmp_path):
    lm = _landmarks()
    lm[15] = np.nan
    path = tmp_path / "own.csv"
    save_labels("vid1", "a.mp4", 30.0, [_rep(1, 15)], {1: ["good"]}, lm, path)
    assert np.isnan(load_labels(path)["knee_angle"].iloc[0])
