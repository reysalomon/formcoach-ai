"""FormCoach AI — веб-интерфейс. Запуск: streamlit run app.py"""

import hashlib
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from formcoach import synthetic as syn
from formcoach.agent import ERROR_TEXT, SquatRules
from formcoach.features import squat_features
from formcoach.hmm import STATE_NAMES, squat_hmm
from formcoach.pipeline import analyze_squat
from formcoach.pose import PoseSequence, draw_skeleton, extract_pose, read_frame

# Цвета фаз — первые 4 слота проверенной категориальной палитры
PHASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LEVEL_ICON = {"ok": "✅", "warn": "⚠️", "danger": "⛔", "info": "ℹ️"}
ERROR_SHORT = {"shallow": "неглубоко", "lean": "наклон", "heels": "пятки", "fast": "быстро"}

st.set_page_config(page_title="FormCoach AI", page_icon="🏋️", layout="wide")
st.title("FormCoach AI — проверка техники приседаний")
st.caption("Загрузи видео сбоку — система посчитает повторения, найдёт фазы движения и подскажет ошибки.")


@st.cache_data(show_spinner=False)
def cached_pose(data: bytes, suffix: str) -> tuple[np.ndarray, float, str]:
    """Распознавание позы долгое, поэтому кешируем по содержимому файла."""
    digest = hashlib.md5(data).hexdigest()
    path = Path(tempfile.gettempdir()) / f"formcoach_{digest}{suffix}"
    path.write_bytes(data)
    seq = extract_pose(path)
    return seq.landmarks, seq.fps, str(path)


# ---------------- боковая панель: вход и параметры ----------------
with st.sidebar:
    st.header("Источник")
    source = st.radio("Данные", ["Загрузить видео", "Демо (синтетические данные)"], label_visibility="collapsed")

    st.header("HMM (фазы)")
    top_angle = st.slider("Угол колена стоя, °", 140, 180, 165)
    bottom_angle = st.slider("Угол колена внизу, °", 50, 130, 85)
    stay_prob = st.slider(
        "Вероятность остаться в фазе", 0.50, 0.99, 0.90, 0.01,
        help="Чем больше, тем реже HMM меняет фазу и тем меньше реагирует на шум.",
    )

    st.header("Правила тренера")
    rules = SquatRules(
        min_depth_angle=st.slider("Глубина: колено не больше, °", 70, 140, 100),
        max_torso_lean=st.slider("Макс. наклон корпуса, °", 20, 70, 45),
        max_heel_lift=st.slider("Макс. подъём пятки (доля голени)", 0.05, 0.50, 0.20, 0.01, help="«—» в таблице: стопу было плохо видно, пятки не оценивались."),
        min_rep_seconds=st.slider("Мин. длительность повторения, с", 0.3, 3.0, 1.0, 0.1),
    )

# ---------------- получаем признаки ----------------
video_path = landmarks = None
if source == "Загрузить видео":
    up = st.file_uploader("Видео приседаний (камера сбоку, всё тело в кадре)", type=["mp4", "mov", "avi", "mkv", "webm"])
    if up is None:
        st.info("Загрузи видео или выбери демо-режим слева.")
        st.stop()
    try:
        with st.spinner("Распознаю позу на каждом кадре... (это может занять минуту)"):
            landmarks, fps, video_path = cached_pose(up.getvalue(), Path(up.name).suffix.lower())
        seq = PoseSequence(landmarks, fps)
        if seq.detected_ratio < 0.5:
            st.error(f"Человек найден только на {seq.detected_ratio:.0%} кадров. Сними так, чтобы всё тело было в кадре.")
            st.stop()
        features = squat_features(seq)
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Не получилось обработать видео: {e}")
        st.stop()
else:
    fps = 30.0
    st.markdown("**Демо:** выбери, какие ошибки «сделает» виртуальный спортсмен (между ними — правильные повторения).")
    chosen = st.multiselect(
        "Ошибки", list(ERROR_TEXT), default=["shallow", "lean"], format_func=lambda c: ERROR_SHORT[c]
    )
    specs = {"shallow": syn.SHALLOW, "lean": syn.LEANING, "heels": syn.HEELS_UP, "fast": syn.TOO_FAST}
    plan = [syn.GOOD]
    for c in chosen:
        plan += [specs[c], syn.GOOD]
    features = syn.synthetic_session(plan, fps=fps)

try:
    a = analyze_squat(features, fps, dict(top_angle=top_angle, bottom_angle=bottom_angle, stay_prob=stay_prob), rules)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------- итоги ----------------
good = sum(r.good for r in a.reps)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Повторений", len(a.reps))
c2.metric("Правильных", good)
c3.metric("С ошибками", len(a.reps) - good)
c4.metric("Длительность", f"{len(a.phases) / fps:.1f} с")

# ---------------- график: угол колена + фазы HMM ----------------
st.subheader("Угол колена и фазы движения (HMM, алгоритм Витерби)")
t = np.arange(len(a.phases)) / fps
df = pd.DataFrame({"t": t, "angle": a.features["knee_angle"], "phase": [STATE_NAMES[p] for p in a.phases]})

# Склеиваем подряд идущие кадры одной фазы в полосы
change = np.flatnonzero(np.diff(a.phases)) + 1
bounds = np.concatenate([[0], change, [len(a.phases)]])
bands = pd.DataFrame(
    {"t0": t[bounds[:-1]], "t1": np.append(t[bounds[1:-1]], t[-1] + 1 / fps), "phase": [STATE_NAMES[a.phases[i]] for i in bounds[:-1]]}
)

phase_scale = alt.Scale(domain=STATE_NAMES, range=PHASE_COLORS)
band_layer = alt.Chart(bands).mark_rect(opacity=0.22).encode(
    x=alt.X("t0:Q", title="Время, с"), x2="t1:Q",
    color=alt.Color("phase:N", scale=phase_scale, title="Фаза", legend=alt.Legend(orient="top")),
)
line = alt.Chart(df).mark_line(strokeWidth=2, color="#8a8985").encode(
    x="t:Q", y=alt.Y("angle:Q", title="Угол колена, °", scale=alt.Scale(domain=[40, 185]))
)
depth_rule = alt.Chart(pd.DataFrame({"y": [rules.min_depth_angle]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y="y:Q")
hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
points = alt.Chart(df).mark_circle(size=70, color="#8a8985").encode(
    x="t:Q", y="angle:Q",
    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[alt.Tooltip("t:Q", title="Время, с", format=".2f"), alt.Tooltip("angle:Q", title="Угол, °", format=".0f"), alt.Tooltip("phase:N", title="Фаза")],
).add_params(hover)
rep_labels = alt.Chart(
    pd.DataFrame({"t": [r.bottom_frame / fps for r in a.reps], "y": [r.min_knee - 8 for r in a.reps], "label": [f"#{r.number}" for r in a.reps]})
).mark_text(fontWeight="bold").encode(x="t:Q", y="y:Q", text="label:N")

st.altair_chart((band_layer + line + depth_rule + points + rep_labels).properties(height=320), width="stretch")
st.caption(f"Пунктир — порог глубины ({rules.min_depth_angle}°). Ниже пунктира — присед достаточно глубокий.")

# ---------------- повторения ----------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Повторения")
    if not a.reps:
        st.warning("Повторений не найдено. Проверь, что камера сбоку и видны колени и стопы.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": r.number,
                        "Итог": "✅ хорошо" if r.good else "⚠️ " + ", ".join(ERROR_SHORT[e] for e in r.errors),
                        "Мин. угол колена, °": round(r.min_knee),
                        "Макс. наклон, °": round(r.max_lean),
                        "Подъём пятки": "—" if np.isnan(r.max_heel) else round(r.max_heel, 2),
                        "Время, с": round(r.seconds, 1),
                    }
                    for r in a.reps
                ]
            ),
            hide_index=True,
            width="stretch",
        )
with right:
    st.subheader("Подсказки тренера (агент)")
    for f in a.feedback:
        st.markdown(f"{LEVEL_ICON[f.level]} `{f.frame / fps:5.1f} с` {f.message}")
    if not a.feedback:
        st.write("Пока нечего сказать.")

# ---------------- нижние точки на видео ----------------
if video_path and a.reps:
    st.subheader("Нижняя точка каждого повторения")
    cols = st.columns(min(len(a.reps), 4))
    for i, r in enumerate(a.reps):
        frame = read_frame(video_path, r.bottom_frame)
        if frame is not None:
            cols[i % len(cols)].image(
                draw_skeleton(frame, landmarks[r.bottom_frame]),
                caption=f"#{r.number}: колено {r.min_knee:.0f}°, наклон {r.max_lean:.0f}°",
            )

# ---------------- объяснение алгоритмов ----------------
with st.expander("Как это работает: 3 AI-темы"):
    st.markdown(
        """
**0. Датчик (не AI-тема проекта).** MediaPipe Pose находит 33 точки тела на каждом кадре.
Из них мы считаем углы: колено (бедро–колено–лодыжка), наклон корпуса, подъём пятки.

**1. Неделя 7 — Hidden Markov Model.** Фазы ВЕРХ → СПУСК → НИЗ → ПОДЪЁМ скрыты, видим только угол и
скорость колена. Каждой фазе соответствует гауссово распределение наблюдений, переходы — по кругу.
Алгоритм **Витерби** (динамическое программирование) находит самую вероятную последовательность фаз;
**прямой алгоритм** считает, насколько видео похоже на приседания вообще.

**2. Неделя 1 — Intelligent Agent.** Тренер — рефлекторный агент с моделью мира. На каждом кадре он
получает восприятие (фаза + углы), обновляет внутреннее состояние (минимальный угол, максимальный наклон
в текущем повторении) и по правилам «условие → действие» выдаёт подсказки.

**3. Неделя 8 — Classical ML** *(в разработке)*: классификатор правильной/неправильной техники с метриками Accuracy/F1.
"""
    )
    hmm = squat_hmm(top_angle, bottom_angle, stay_prob)
    st.markdown("**Матрица переходов HMM** (строка — откуда, столбец — куда):")
    st.dataframe(pd.DataFrame(hmm.trans, index=STATE_NAMES, columns=STATE_NAMES).round(3))
    st.markdown(f"log P(видео | модель) = **{a.log_likelihood:.1f}**  (в среднем {a.log_likelihood / len(a.phases):.2f} на кадр)")
