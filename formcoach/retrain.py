"""Дообучение классификатора на своих размеченных видео + честная оценка «до / после».

Зачем: в датасете Binus ошибки преувеличенные (наклон ~64°), а у реального человека — мягкие.
Модель, обученная только на Binus, такие ошибки не замечает. Добавляем свои повторения.

Как оцениваем (чтобы не обмануть себя):
  «leave-one-video-out» — по очереди убираем одно своё видео целиком,
  обучаем на Binus train + остальных своих видео и проверяем на убранном.
  «До» — та же проверка модели, обученной только на Binus.
  Оценка — по повторениям (как в приложении: усредняем кадры bottom±3).
  Дополнительно смотрим Binus test: дообучение не должно испортить старое.

Запуск:  python -m formcoach.retrain            — только отчёт
         python -m formcoach.retrain --save     — отчёт + сохранить новую модель для приложения
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classifier import CLASSES, DATA, MODEL_PATH, NO_FEET_FEATURES, FormClassifier, Scores, classify_rep, load_dataset, make_models
from .features import FRAME_FEATURES, feet_visible, frame_features
from .labels import LABELS_CSV, load_labels

WINDOW = 3  # кадры bottom±3 — так же, как classify_rep в приложении
REPORT_PATH = DATA / "retrain_report.json"


def own_class(row) -> str | None:
    """Своя метка повторения -> класс классификатора.

    Классификатор смотрит на позу в нижней точке, поэтому понимает только спину и пятки.
    «Неглубоко» и «быстро» оценивают HMM и агент — такие повторения сюда не берём.
    Если и наклон, и пятки — считаем «спина» (тот же приоритет, что в rule_baseline).
    """
    if row["lean"]:
        return "bad_back"
    if row["heels"]:
        return "bad_heel"
    if row["good"]:
        return "good"
    return None


def load_own(labels_csv=LABELS_CSV) -> tuple[pd.DataFrame, dict]:
    """Свои повторения с классом + точки тела их видео: (таблица повторений, {video_id: landmarks})."""
    reps = load_labels(labels_csv)
    if reps.empty:
        return reps, {}
    reps = reps.assign(label=[own_class(r) for _, r in reps.iterrows()])
    reps = reps[reps["label"].notna()].reset_index(drop=True)
    landmarks = {}
    for vid in reps["video_id"].unique():
        path = labels_csv.parent / "landmarks" / f"{vid}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Нет точек тела для видео {vid}: {path}. Сохрани разметку этого видео заново.")
        landmarks[vid] = np.load(path)
    return reps, landmarks


def own_frames(reps: pd.DataFrame, landmarks: dict, window: int = WINDOW) -> pd.DataFrame:
    """Кадры вокруг нижней точки каждого повторения -> строки признаков (как в датасете Binus)."""
    rows = []
    for _, r in reps.iterrows():
        lm = landmarks[r["video_id"]]
        for t in range(int(r["bottom_frame"]) - window, int(r["bottom_frame"]) + window + 1):
            if 0 <= t < len(lm) and not np.isnan(lm[t, 0, 0]):
                rows.append({**frame_features(lm[t]), "label": r["label"], "video_id": r["video_id"], "feet": feet_visible(lm[t])})
    return pd.DataFrame(rows, columns=[*FRAME_FEATURES, "label", "video_id", "feet"])


def fit_classifier(binus_train: pd.DataFrame, own: pd.DataFrame, model_name: str, repeat: int) -> FormClassifier:
    """Обучает основную и запасную модели на Binus + своих кадрах.

    repeat — во сколько раз повторяем свои кадры: их ~десятки против ~2800 кадров Binus,
    без повтора модель их почти не заметит. Повтор = вес примера, просто и понятно.
    Основная модель получает свои кадры только там, где видны стопы (иначе признаки стопы — мусор).
    """
    own_full = own[own["feet"]] if len(own) else own
    X = pd.concat([binus_train[FRAME_FEATURES]] + [own_full[FRAME_FEATURES]] * repeat, ignore_index=True)
    y = np.concatenate([binus_train["label"].to_numpy()] + [own_full["label"].to_numpy()] * repeat)
    model = make_models()[model_name].fit(X, y)

    nb, no = binus_train[binus_train["label"] != "bad_heel"], own[own["label"] != "bad_heel"]
    Xn = pd.concat([nb[NO_FEET_FEATURES]] + [no[NO_FEET_FEATURES]] * repeat, ignore_index=True)
    yn = np.concatenate([nb["label"].to_numpy()] + [no["label"].to_numpy()] * repeat)
    no_feet = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xn, yn)
    return FormClassifier(model, model_name, no_feet)


def predict_reps(clf: FormClassifier, reps: pd.DataFrame, landmarks: dict) -> list[str]:
    """Ответ модели на каждое повторение — ровно так, как в приложении."""
    out = []
    for _, r in reps.iterrows():
        v = classify_rep(clf, landmarks[r["video_id"]], int(r["bottom_frame"]), WINDOW)
        out.append(v.label if v else "good")  # человека не нашли — ошибку не обнаружили
    return out


def leave_one_video_out(binus_train, reps, landmarks, frames, model_name, repeat) -> dict:
    """Оценка «до» и «после» по своим повторениям; каждое видео проверяется моделью, которая его не видела."""
    before = fit_classifier(binus_train, frames.iloc[0:0], model_name, repeat)
    pred_before = predict_reps(before, reps, landmarks)
    pred_after = [None] * len(reps)
    for vid in reps["video_id"].unique():
        test = (reps["video_id"] == vid).to_numpy()
        clf = fit_classifier(binus_train, frames[frames["video_id"] != vid], model_name, repeat)
        for i, p in zip(np.flatnonzero(test), predict_reps(clf, reps[test], landmarks)):
            pred_after[i] = p
    y = reps["label"].tolist()
    return {"before": Scores.of(y, pred_before), "after": Scores.of(y, pred_after)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="какую модель сохранить (по умолчанию — лучшая из classifier_report.json)")
    ap.add_argument("--repeat", type=int, default=10, help="во сколько раз повторить свои кадры (вес)")
    ap.add_argument("--save", action="store_true", help="сохранить дообученную модель для приложения")
    args = ap.parse_args(argv)
    if args.repeat < 1:
        ap.error("--repeat должен быть >= 1")

    reps, landmarks = load_own()
    if reps.empty:
        raise SystemExit("Нет своих размеченных повторений со спиной/пятками/правильно. Размечай видео во вкладке «Разметка».")
    n_videos = reps["video_id"].nunique()
    print(f"Свои данные: {n_videos} видео, {len(reps)} повторений:", reps["label"].value_counts().to_dict())
    if n_videos < 2:
        print("ВНИМАНИЕ: видео одно — при проверке «после» не на чем дообучаться, «до» и «после» совпадут.")

    df, _ = load_dataset()
    binus_train, binus_test = df[df["split"] == "train"], df[df["split"] == "test"]
    frames = own_frames(reps, landmarks)
    print(f"Своих кадров: {len(frames)} (стопы видны на {int(frames['feet'].sum())})\n")

    report_path = DATA / "classifier_report.json"
    default = json.loads(report_path.read_text())["best"] if report_path.exists() else "Логистическая регрессия"
    chosen = args.model or default
    if chosen not in make_models():
        raise SystemExit(f"Нет модели «{chosen}». Есть: {list(make_models())}")

    results = {}
    print(f"{'Модель':28s} {'свои F1 до':>11s} {'после':>7s} {'acc до':>7s} {'после':>7s} {'Binus test F1':>14s}")
    for name in make_models():
        r = leave_one_video_out(binus_train, reps, landmarks, frames, name, args.repeat)
        full = fit_classifier(binus_train, frames, name, args.repeat)
        r["binus_test_after"] = Scores.of(binus_test["label"], full.model.predict(binus_test[FRAME_FEATURES]))
        results[name] = r
        print(f"{name:28s} {r['before'].f1_macro:11.3f} {r['after'].f1_macro:7.3f} "
              f"{r['before'].accuracy:7.3f} {r['after'].accuracy:7.3f} {r['binus_test_after'].f1_macro:14.3f}")

    print(f"\nМатрица ошибок «после» для «{chosen}» (строки — правда, столбцы — ответ):", CLASSES)
    for c, row in zip(CLASSES, results[chosen]["after"].confusion):
        print(f"  {c:9s} {row}")

    out = {"videos": int(n_videos), "reps": int(len(reps)), "frames": int(len(frames)), "repeat": args.repeat,
           "chosen": chosen, "saved": args.save,
           "results": {k: {s: vars(v) for s, v in r.items()} for k, r in results.items()}}
    REPORT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {REPORT_PATH.relative_to(DATA.parent.parent)}")

    if args.save:
        clf = fit_classifier(binus_train, frames, chosen, args.repeat)  # на всех своих видео
        joblib.dump({"model": clf.model, "name": f"{chosen} + свои видео", "features": FRAME_FEATURES,
                     "no_feet_model": clf.no_feet_model, "own_videos": int(n_videos)}, MODEL_PATH)
        print(f"Модель сохранена: {MODEL_PATH.relative_to(DATA.parent.parent)} (вернуть старую: python -m formcoach.classifier)")


if __name__ == "__main__":
    main()
