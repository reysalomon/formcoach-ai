"""FormCoach AI — сервер для мобильного сайта. Запуск: uvicorn server:app --host 0.0.0.0 --port 8000

Сервер не содержит своей AI-логики: он принимает видео с телефона, вызывает тот же
код, что и Streamlit-приложение (formcoach/: HMM, агент, классификатор), и отдаёт
результат в JSON. Интерфейс лежит в папке web/ (HTML + CSS + JavaScript).
"""

import base64
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from formcoach import synthetic as syn
from formcoach.agent import ERROR_TEXT, SquatRules
from formcoach.classifier import CLASS_RU, DATA as ML_DATA, FormClassifier
from formcoach.features import squat_features
from formcoach.hmm import STATE_NAMES, squat_hmm
from formcoach.pipeline import Analysis, analyze_squat
from formcoach.pose import draw_skeleton, extract_pose, read_frame

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD_MB = 100
MAX_SECONDS = 90  # длиннее не обрабатываем: на бесплатном сервере это слишком долго
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
DEMO_SPECS = {"shallow": syn.SHALLOW, "lean": syn.LEANING, "heels": syn.HEELS_UP, "fast": syn.TOO_FAST}
ERROR_SHORT = {"shallow": "неглубоко", "lean": "наклон", "heels": "пятки", "fast": "быстро"}

app = FastAPI(title="FormCoach AI")


@lru_cache(maxsize=1)
def load_classifier() -> FormClassifier | None:
    return FormClassifier.load()


def parse_settings(raw: str | None) -> tuple[dict, SquatRules]:
    """Настройки из интерфейса (JSON-строка) -> параметры HMM и правила агента.
    Неизвестные ключи игнорируем, значения проверяем по тем же границам, что и ползунки."""
    try:
        s = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "Настройки переданы в неверном формате.")
    if not isinstance(s, dict):
        raise HTTPException(400, "Настройки переданы в неверном формате.")

    def num(key, default, lo, hi):
        v = s.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise HTTPException(400, f"Параметр «{key}» должен быть числом от {lo} до {hi}.")
        return float(v)

    hmm = dict(top_angle=num("top_angle", 165, 140, 180), bottom_angle=num("bottom_angle", 85, 50, 130),
               stay_prob=num("stay_prob", 0.9, 0.5, 0.99))
    d = SquatRules()
    rules = SquatRules(
        max_thigh_angle=num("max_thigh_angle", d.max_thigh_angle, -10, 70),
        max_torso_lean=num("max_torso_lean", d.max_torso_lean, 15, 70),
        max_heel_lift=num("max_heel_lift", d.max_heel_lift, 0.05, 0.5),
        min_rep_seconds=num("min_rep_seconds", d.min_rep_seconds, 0.3, 3.0),
    )
    return hmm, rules


def crop_to_body(img: np.ndarray, points: np.ndarray, aspect: float = 4 / 3, margin: float = 0.15) -> np.ndarray:
    """Вырезаем из кадра прямоугольник 4:3 вокруг человека — на вертикальном видео иначе в карточке виден потолок."""
    xy = points[:, :2][~np.isnan(points[:, 0])]
    if len(xy) == 0:
        return img
    h, w = img.shape[:2]
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bh = (y1 - y0) * (1 + 2 * margin)
    bw = (x1 - x0) * (1 + 2 * margin)
    bw, bh = max(bw, bh * aspect), max(bh, bw / aspect)  # дотягиваем до пропорции 4:3
    bw, bh = min(bw, w), min(bh, h)
    left = int(np.clip(cx - bw / 2, 0, w - bw))
    top = int(np.clip(cy - bh / 2, 0, h - bh))
    return img[top:top + int(bh), left:left + int(bw)]


def thumbnail(video_path: Path, landmarks: np.ndarray, frame: int, width: int = 360) -> str | None:
    """Кадр нижней точки со скелетом -> маленькая JPEG-картинка в base64 (для <img src=...>)."""
    rgb = read_frame(video_path, frame)
    if rgb is None:
        return None
    img = draw_skeleton(rgb, landmarks[frame], color=(198, 255, 61))
    img = crop_to_body(img, landmarks[frame])
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def clean(x) -> float | None:
    """NaN нельзя положить в JSON — превращаем его в null."""
    return None if x is None or np.isnan(x) else round(float(x), 2)


def to_json(a: Analysis, hmm_params: dict, video_path: Path | None = None, landmarks: np.ndarray | None = None) -> dict:
    """Результат анализа в виде, удобном для интерфейса."""
    reps = []
    for r, v in zip(a.reps, a.ml):
        reps.append({
            "number": r.number,
            "start_s": round(r.start / a.fps, 2), "end_s": round(r.end / a.fps, 2),
            "good": r.good,
            "errors": [{"key": e, "short": ERROR_SHORT[e], "text": ERROR_TEXT[e]} for e in r.errors],
            "min_knee": clean(r.min_knee), "min_thigh": clean(r.min_thigh), "max_lean": clean(r.max_lean),
            "max_heel": clean(r.max_heel), "seconds": clean(r.seconds),
            "ml": None if v is None else {
                "label": v.label, "label_ru": CLASS_RU[v.label], "no_feet": v.no_feet,
                "probs": {k: clean(p) for k, p in v.probs.items()},
            },
            "image": thumbnail(video_path, landmarks, r.bottom_frame) if video_path is not None else None,
        })
    # для графика достаточно ~300 точек, иначе ответ получается тяжёлым
    step = max(1, len(a.phases) // 300)
    good = sum(r.good for r in a.reps)
    return {
        "summary": {"reps": len(a.reps), "good": good, "bad": len(a.reps) - good,
                    "seconds": round(len(a.phases) / a.fps, 1)},
        "reps": reps,
        "feedback": [{"t": round(f.frame / a.fps, 1), "level": f.level, "message": f.message} for f in a.feedback],
        "series": {
            "t": [round(i / a.fps, 2) for i in range(0, len(a.phases), step)],
            "knee": [clean(x) for x in a.features["knee_angle"][::step]],
            "phase": [int(p) for p in a.phases[::step]],
            "bottoms": [{"t": round(r.bottom_frame / a.fps, 2), "knee": clean(r.min_knee), "n": r.number} for r in a.reps],
        },
        "phase_names": STATE_NAMES,
        "hmm": {"trans": np.round(squat_hmm(**hmm_params).trans, 3).tolist(),
                "log_likelihood": round(a.log_likelihood, 1),
                "per_frame": round(a.log_likelihood / len(a.phases), 2)},
        "has_ml": video_path is not None and load_classifier() is not None,
    }


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...), settings: str | None = Form(None)):
    hmm_params, rules = parse_settings(settings)
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_TYPES:
        raise HTTPException(400, "Это не видеофайл. Подходят mp4, mov, webm, avi, mkv, 3gp.")
    data = await video.read()
    if not data:
        raise HTTPException(400, "Файл пустой.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Видео больше {MAX_UPLOAD_MB} МБ. Сними покороче (20–60 секунд хватает).")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if n > 0 and n / fps > MAX_SECONDS:
            raise HTTPException(400, f"Видео длиннее {MAX_SECONDS} секунд. Обрежь его или сними 5–10 повторений.")
        try:
            seq = extract_pose(path)
            if seq.detected_ratio < 0.5:
                raise HTTPException(422, f"Человек найден только на {seq.detected_ratio:.0%} кадров. "
                                         "Сними так, чтобы всё тело было в кадре.")
            a = analyze_squat(squat_features(seq), seq.fps, hmm_params, rules,
                              landmarks=seq.landmarks, classifier=load_classifier())
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, f"Не получилось обработать видео: {e}")
        return to_json(a, hmm_params, path, seq.landmarks)


@app.get("/api/demo")
def demo(errors: str = "shallow,lean", settings: str | None = None):
    """Демо без видео: синтетический спортсмен делает выбранные ошибки (между ними — правильные повторения)."""
    hmm_params, rules = parse_settings(settings)
    chosen = [e for e in errors.split(",") if e]
    unknown = [e for e in chosen if e not in DEMO_SPECS]
    if unknown:
        raise HTTPException(400, f"Неизвестные ошибки: {', '.join(unknown)}")
    plan = [syn.GOOD]
    for e in chosen:
        plan += [DEMO_SPECS[e], syn.GOOD]
    fps = 30.0
    return to_json(analyze_squat(syn.synthetic_session(plan, fps=fps), fps, hmm_params, rules), hmm_params)


@app.get("/api/info")
def info():
    """Настройки по умолчанию и метрики моделей — для экрана «Как это работает»."""
    d = SquatRules()
    out = {
        "defaults": {"top_angle": 165, "bottom_angle": 85, "stay_prob": 0.9, "max_thigh_angle": d.max_thigh_angle,
                     "max_torso_lean": d.max_torso_lean, "max_heel_lift": d.max_heel_lift,
                     "min_rep_seconds": d.min_rep_seconds},
        "errors": ERROR_SHORT,
        "models": None, "retrain": None,
    }
    clf_report, retrain_report = ML_DATA / "classifier_report.json", ML_DATA / "retrain_report.json"
    if clf_report.exists():
        rep = json.loads(clf_report.read_text())
        out["models"] = [{"name": k, "cv_f1": round(v["cv"]["f1_macro"], 3), "test_f1": round(v["test"]["f1_macro"], 3)}
                         for k, v in rep["results"].items()]
    if retrain_report.exists():
        rr = json.loads(retrain_report.read_text())
        out["retrain"] = {"videos": rr["videos"], "reps": rr["reps"]}
    return out


# Service worker должен отдаваться из корня сайта, иначе браузер не даст ему управлять всеми страницами
@app.get("/sw.js")
def service_worker():
    return FileResponse(WEB / "sw.js", media_type="application/javascript")


# Всё остальное — файлы интерфейса (index.html, стили, скрипт, иконки)
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
