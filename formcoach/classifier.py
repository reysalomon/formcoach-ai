"""Неделя 8 — Classical ML: классификатор техники приседа по позе на кадре.

Классы: good (правильно), bad_back (сильный наклон / круглая спина), bad_heel (пятки оторваны).
Входы:  7 признаков позы из features.frame_features() — углы и отношения длин.
Данные: датасет Binus (Zenodo 17558630), фото сбоку, ~6 разных людей.

Главная ловушка: соседние фото в датасете — почти одинаковые кадры одного видео.
Если делить на train/test случайно, модель "видела" почти такой же кадр, и точность
получается завышенной. Поэтому мы группируем кадры в клипы и проверяем модель только
на клипах, которых она не видела (GroupKFold).

Запуск обучения и отчёта:  python -m formcoach.classifier
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .features import FRAME_FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "squat_form_clf.joblib"
CLASSES = ["good", "bad_back", "bad_heel"]
CLASS_RU = {"good": "правильно", "bad_back": "спина", "bad_heel": "пятки"}
MIN_VISIBILITY = 0.5
# Запасная модель для кадров, где стопы не видны (вышли за кадр): только верх тела.
# Пятки она оценить не может, поэтому учится отличать только good от bad_back.
NO_FEET_FEATURES = ["torso_lean", "hip_angle", "thigh_angle"]
CLIP_THRESHOLD = 12.0  # разница миниатюр (0..255), выше — начинается новый клип


def make_models() -> dict:
    """Модели, которые сравниваем. Все — классические, объяснимые на защите."""
    return {
        "Логистическая регрессия": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Дерево решений (глубина 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Случайный лес": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0),
        "k ближайших соседей (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }


def rule_baseline(X: pd.DataFrame, max_lean: float = 30.0, max_heel: float = 0.2) -> np.ndarray:
    """Наши ручные правила (как у агента) — чтобы честно сравнить их с ML."""
    pred = np.full(len(X), "good", dtype=object)
    pred[X["heel_above_toe"].to_numpy() > max_heel] = "bad_heel"
    pred[X["torso_lean"].to_numpy() > max_lean] = "bad_back"
    return pred


def assign_clips(df: pd.DataFrame, thumbs: np.ndarray, threshold: float = CLIP_THRESHOLD) -> np.ndarray:
    """Номер клипа для каждого фото: внутри (split, класс) идём по номерам файлов,
    и если картинка резко отличается от предыдущей — значит, начался другой клип."""
    clips = np.empty(len(df), dtype=int)
    clip = 0
    for _, g in df.groupby(["split", "label"], sort=True):
        order = g.sort_values("idx").index.to_numpy()
        prev = None
        for i in order:
            cur = thumbs[i].astype(float)
            if prev is None or np.abs(cur - prev).mean() > threshold:
                clip += 1
            clips[i] = clip
            prev = cur
    return clips


def load_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA / "binus_features.csv")
    thumbs = np.load(DATA / "binus_thumbs.npz", allow_pickle=True)["thumbs"]
    df["clip"] = assign_clips(df, thumbs)
    info = {"всего фото": len(df), "поза не найдена": int((~df["detected"]).sum())}
    df = df[df["detected"]].copy()
    low = df["min_visibility"] < MIN_VISIBILITY
    info["плохо видно ногу (отброшено)"] = int(low.sum())
    df = df[~low].reset_index(drop=True)
    info["осталось"] = len(df)
    info["клипов"] = int(df["clip"].nunique())
    return df, info


@dataclass
class Scores:
    accuracy: float
    f1_macro: float
    confusion: list  # строки — правда, столбцы — предсказание (порядок CLASSES)

    @staticmethod
    def of(y_true, y_pred) -> "Scores":
        return Scores(
            float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, average="macro", labels=[c for c in CLASSES if c in set(y_true)])),
            confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        )


def cross_validate(df: pd.DataFrame, model_factory, n_splits: int = 5) -> Scores:
    """GroupKFold: кадры одного клипа всегда целиком в train ИЛИ в test."""
    X, y, groups = df[FRAME_FEATURES], df["label"].to_numpy(), df["clip"].to_numpy()
    pred = np.empty(len(df), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return Scores.of(y, pred)


def evaluate_all(df: pd.DataFrame) -> dict:
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    results = {"Ручные правила (агент)": {"cv": Scores.of(train["label"], rule_baseline(train)), "test": Scores.of(test["label"], rule_baseline(test))}}
    for name in make_models():
        cv = cross_validate(train, lambda n=name: make_models()[n])
        m = make_models()[name].fit(train[FRAME_FEATURES], train["label"])
        results[name] = {"cv": cv, "test": Scores.of(test["label"], m.predict(test[FRAME_FEATURES]))}
    return results


def evaluate_no_feet(df: pd.DataFrame) -> dict:
    """Запасная модель: good vs bad_back только по верху тела."""
    d = df[df["label"] != "bad_heel"]
    train, test = d[d["split"] == "train"], d[d["split"] == "test"]
    factory = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  # noqa: E731
    X, y = train[NO_FEET_FEATURES], train["label"].to_numpy()
    pred = np.empty(len(train), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(X, y, train["clip"]):
        pred[te] = factory().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    model = factory().fit(X, y)
    return {"cv": Scores.of(y, pred), "test": Scores.of(test["label"], model.predict(test[NO_FEET_FEATURES])), "model": model}


class FormClassifier:
    """Обученные модели для приложения: признаки кадра -> вероятности классов.

    Если стопы видно плохо, признаки стопы — мусор (так было на реальном видео:
    модель уверенно говорила "пятки" на всех повторениях), поэтому берём запасную модель.
    """

    def __init__(self, model, name: str, no_feet_model=None):
        self.model, self.name, self.no_feet_model = model, name, no_feet_model

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "FormClassifier | None":
        if not Path(path).exists():
            return None
        d = joblib.load(path)
        return cls(d["model"], d["name"], d.get("no_feet_model"))

    def predict_proba(self, features: dict, feet_visible: bool = True) -> dict[str, float]:
        """Вероятности классов. Если стопы не видны — "bad_heel" будет None (не оцениваем)."""
        use_full = feet_visible or self.no_feet_model is None
        model, cols = (self.model, FRAME_FEATURES) if use_full else (self.no_feet_model, NO_FEET_FEATURES)
        x = pd.DataFrame([[features[k] for k in cols]], columns=cols)
        if x.isna().any(axis=None):
            raise ValueError("В признаках есть пропуски — кадр распознан плохо.")
        p = dict(zip(model.classes_, model.predict_proba(x)[0]))
        return {c: (float(p[c]) if c in p else None) for c in CLASSES}


def main():
    df, info = load_dataset()
    print("Данные:", info)
    print("Фото по классам (train/test):")
    print(df.groupby(["split", "label"]).size().unstack(0), "\n")

    results = evaluate_all(df)
    print(f"{'Модель':32s} {'CV acc':>7s} {'CV F1':>7s} {'test acc':>9s} {'test F1':>8s}")
    for name, r in results.items():
        print(f"{name:32s} {r['cv'].accuracy:7.3f} {r['cv'].f1_macro:7.3f} {r['test'].accuracy:9.3f} {r['test'].f1_macro:8.3f}")

    # Лучшую по CV F1 (не по test — test только для финальной честной оценки) обучаем на всём train
    ml = {k: v for k, v in results.items() if k in make_models()}
    best = max(ml, key=lambda k: ml[k]["cv"].f1_macro)
    train = df[df["split"] == "train"]
    model = make_models()[best].fit(train[FRAME_FEATURES], train["label"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    nf = evaluate_no_feet(df)
    results["Без стоп: лог. регрессия (good/bad_back)"] = {"cv": nf["cv"], "test": nf["test"]}
    print(f"{'Без стоп (good/bad_back)':32s} {nf['cv'].accuracy:7.3f} {nf['cv'].f1_macro:7.3f} {nf['test'].accuracy:9.3f} {nf['test'].f1_macro:8.3f}")
    joblib.dump({"model": model, "name": best, "features": FRAME_FEATURES, "no_feet_model": nf["model"]}, MODEL_PATH)
    print(f"\nЛучшая модель: {best} -> сохранена в {MODEL_PATH.relative_to(ROOT)}")
    print("Матрица ошибок на test (строки — правда, столбцы — предсказание):", CLASSES)
    for c, row in zip(CLASSES, results[best]["test"].confusion):
        print(f"  {c:9s} {row}")

    tree = make_models()["Дерево решений (глубина 4)"].fit(train[FRAME_FEATURES], train["label"])
    print("\nПравила, которые выучило дерево решений:\n" + export_text(tree, feature_names=FRAME_FEATURES))

    out = {"data": info, "best": best, "results": {k: {s: vars(v) for s, v in r.items()} for k, r in results.items()}}
    (DATA / "classifier_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
