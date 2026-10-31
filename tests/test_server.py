"""Проверка сервера мобильного сайта (FastAPI) без браузера и без настоящего видео."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from formcoach.pose import PoseSequence

client = TestClient(server.app)
OWN_LANDMARKS = Path(__file__).resolve().parent.parent / "data" / "labels" / "landmarks" / "f691fd75ae576ae48ec5e8ae25afeb7e.npy"


def test_index_and_pwa_files_served():
    assert "FormCoach" in client.get("/").text
    assert client.get("/manifest.webmanifest").json()["display"] == "standalone"
    assert client.get("/sw.js").headers["content-type"].startswith("application/javascript")
    assert client.get("/icons/icon-192.png").status_code == 200


def test_demo_finds_chosen_errors():
    r = client.get("/api/demo", params={"errors": "shallow,lean"}).json()
    # хорошее, неглубокое, хорошее, наклон, хорошее — как в Streamlit-демо
    assert (r["summary"]["reps"], r["summary"]["good"], r["summary"]["bad"]) == (5, 3, 2)
    assert [e["key"] for e in r["reps"][1]["errors"]] == ["shallow"]
    assert [e["key"] for e in r["reps"][3]["errors"]] == ["lean"]
    assert len(r["series"]["t"]) == len(r["series"]["knee"]) == len(r["series"]["phase"])
    assert r["has_ml"] is False and r["reps"][0]["image"] is None


def test_demo_settings_change_result():
    loose = json.dumps({"max_thigh_angle": 60})  # порог глубины 60° — неглубокий присед теперь ок
    r = client.get("/api/demo", params={"errors": "shallow,lean", "settings": loose}).json()
    assert r["summary"]["good"] == 4


@pytest.mark.parametrize("settings", ['{"max_torso_lean": 500}', '{"stay_prob": "abc"}', "не json", "[1, 2]"])
def test_bad_settings_rejected(settings):
    r = client.get("/api/demo", params={"settings": settings})
    assert r.status_code == 400
    assert r.json()["detail"]


def test_unknown_demo_error_rejected():
    assert client.get("/api/demo", params={"errors": "jump"}).status_code == 400


def test_info_has_defaults_and_models():
    r = client.get("/api/info").json()
    assert r["defaults"]["max_torso_lean"] == 30
    assert r["models"] is None or all("cv_f1" in m for m in r["models"])


def test_upload_rejects_non_video_and_empty():
    r = client.post("/api/analyze", files={"video": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "видео" in r.json()["detail"]
    r = client.post("/api/analyze", files={"video": ("a.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_upload_broken_video_gives_readable_error():
    # это не настоящее mp4 — OpenCV не откроет файл, пользователь должен получить понятное сообщение
    r = client.post("/api/analyze", files={"video": ("a.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 422
    assert "Не получилось обработать видео" in r.json()["detail"]


def test_upload_without_person_gives_hint(monkeypatch):
    empty = np.full((60, 33, 4), np.nan)
    monkeypatch.setattr(server, "extract_pose", lambda path: PoseSequence(empty, 30.0))
    r = client.post("/api/analyze", files={"video": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 422 and "всё тело" in r.json()["detail"]


@pytest.mark.skipif(not OWN_LANDMARKS.exists(), reason="нет размеченного видео")
def test_upload_full_pipeline_on_real_landmarks(monkeypatch):
    """Подменяем только MediaPipe: точки тела берём из настоящего видео (своя разметка, 4 повторения)."""
    landmarks = np.load(OWN_LANDMARKS)
    monkeypatch.setattr(server, "extract_pose", lambda path: PoseSequence(landmarks, 30.0))
    monkeypatch.setattr(server, "read_frame", lambda path, i: np.zeros((480, 640, 3), np.uint8))
    r = client.post("/api/analyze", files={"video": ("squats.mp4", b"x", "video/mp4")})
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["reps"] == 4
    assert all(rep["image"].startswith("data:image/jpeg;base64,") for rep in data["reps"])
    if data["has_ml"]:
        assert all(rep["ml"]["label"] in ("good", "bad_back", "bad_heel") for rep in data["reps"])
