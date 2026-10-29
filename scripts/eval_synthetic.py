"""Метрики HMM (неделя 7) и агента (неделя 1) на синтетических видео с известным ответом.

Запуск: python scripts/eval_synthetic.py
Генерируем много сессий со случайным числом повторений, случайными ошибками и разным шумом.
Сравниваем:
  - счёт повторений: HMM + агент против простого порога «угол колена < 120°»;
  - поиск ошибок агентом: precision / recall / F1 по каждой ошибке.
Результат: data/processed/synthetic_report.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formcoach.pipeline import analyze_squat  # noqa: E402
from formcoach.synthetic import RepSpec, synthetic_session  # noqa: E402

FPS = 30.0
ERRORS = ["shallow", "lean", "heels", "fast"]


def random_rep(rng) -> tuple[RepSpec, str]:
    """Повторение со случайной ошибкой (или без) и случайной силой."""
    kind = rng.choice(["good", *ERRORS], p=[0.4, 0.15, 0.15, 0.15, 0.15])
    spec = RepSpec(depth=rng.uniform(78, 92), lean=rng.uniform(12, 24), heel=rng.uniform(0, 0.08), seconds=rng.uniform(1.6, 2.8))
    if kind == "shallow":
        spec.depth = rng.uniform(120, 135)
    elif kind == "lean":
        spec.lean = rng.uniform(38, 55)
    elif kind == "heels":
        spec.heel = rng.uniform(0.28, 0.4)
    elif kind == "fast":
        spec.seconds = rng.uniform(0.5, 0.7)
    return spec, str(kind)


def threshold_count(knee: np.ndarray, threshold: float = 120.0) -> int:
    """Наивный счётчик: повторение = каждый раз, когда угол опустился ниже порога."""
    below = knee < threshold
    return int(np.sum(below[1:] & ~below[:-1]))


def main(n_sessions: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    count = {"hmm": [], "threshold": []}
    tp, fp, fn = (dict.fromkeys(ERRORS, 0) for _ in range(3))
    times = []
    for s in range(n_sessions):
        specs, kinds = zip(*[random_rep(rng) for _ in range(rng.integers(3, 9))])
        noise = rng.uniform(1.0, 5.0)  # шум угла, градусы
        feats = synthetic_session(list(specs), fps=FPS, noise=noise, seed=s)
        t = time.perf_counter()
        a = analyze_squat(feats, FPS)
        times.append(time.perf_counter() - t)
        count["hmm"].append(len(a.reps) - len(specs))
        count["threshold"].append(threshold_count(feats["knee_angle"]) - len(specs))
        if len(a.reps) != len(specs):
            continue  # ошибки сравниваем только когда повторения совпали один к одному
        for rep, kind in zip(a.reps, kinds):
            for e in ERRORS:
                found, truth = e in rep.errors, e == kind
                tp[e] += found and truth
                fp[e] += found and not truth
                fn[e] += truth and not found

    def summary(diffs):
        d = np.array(diffs)
        return {"exact": float(np.mean(d == 0)), "mae": float(np.mean(np.abs(d)))}

    errors = {}
    for e in ERRORS:
        p = tp[e] / max(tp[e] + fp[e], 1)
        r = tp[e] / max(tp[e] + fn[e], 1)
        errors[e] = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "support": tp[e] + fn[e]}
    out = {
        "sessions": n_sessions,
        "count": {k: summary(v) for k, v in count.items()},
        "errors": errors,
        "analysis_ms_per_session": 1000 * float(np.mean(times)),
    }
    print(f"Сессий: {n_sessions}, шум угла 1–5°")
    print(f"Счёт повторений, точно угадано: HMM {out['count']['hmm']['exact']:.1%} (MAE {out['count']['hmm']['mae']:.2f}), "
          f"порог 120° {out['count']['threshold']['exact']:.1%} (MAE {out['count']['threshold']['mae']:.2f})")
    for e, m in errors.items():
        print(f"  {e:8s} precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}  (n={m['support']})")
    print(f"Анализ одной сессии: {out['analysis_ms_per_session']:.1f} мс")
    (ROOT / "data" / "processed" / "synthetic_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
