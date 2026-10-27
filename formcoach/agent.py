"""Неделя 1 — Intelligent Agent: виртуальный тренер по приседаниям.

Тип агента: рефлекторный агент с моделью мира (model-based reflex agent, Russell & Norvig).
  - Восприятие (percept): на каждом кадре агент получает фазу движения (от HMM)
    и углы тела (от features.py). Всего видео сразу он НЕ видит — только текущий кадр.
  - Модель мира (внутреннее состояние): номер текущего повторения, его начало,
    минимальный угол колена, максимальный наклон и т.д. — то, что уже нельзя увидеть
    на текущем кадре, но нужно помнить.
  - Правила "условие -> действие": если корпус наклонён сильнее порога -> подсказка
    "держи спину"; если повторение закончилось неглубоким -> "опустись ниже".
  - Действия: текстовые подсказки разного уровня (ok / warn / danger).

PEAS:
  Performance — правильно посчитанные повторения и точные подсказки об ошибках
  Environment — человек, выполняющий приседания перед камерой
  Actuators   — сообщения на экране
  Sensors     — камера + MediaPipe (точки тела) + HMM (фаза)

Пороги правил вынесены в SquatRules, их можно менять в интерфейсе.
"""

from dataclasses import dataclass, field

from .hmm import BOTTOM, TOP

# Коды ошибок — потом они же станут метками для классификатора (неделя 8)
ERROR_TEXT = {
    "shallow": "Неглубокий присед — опустись ниже (бедро хотя бы параллельно полу)",
    "lean": "Сильный наклон корпуса вперёд — держи грудь выше",
    "heels": "Пятки отрываются от пола — переноси вес на всю стопу",
    "fast": "Слишком быстро — опускайся подконтрольно",
}


@dataclass
class SquatRules:
    min_depth_angle: float = 100.0  # колено должно согнуться хотя бы до этого угла
    max_torso_lean: float = 45.0  # максимально допустимый наклон корпуса, градусы
    danger_extra_lean: float = 15.0  # на сколько градусов сверх порога — уже опасно для спины
    max_heel_lift: float = 0.08  # подъём пятки, доля длины голени
    min_rep_seconds: float = 1.0  # быстрее — значит "падает" вниз без контроля
    min_rep_frames: int = 5  # короче — это шум, а не повторение


@dataclass
class Percept:
    frame: int
    phase: int
    knee_angle: float
    torso_lean: float
    heel_lift: float


@dataclass
class Feedback:
    frame: int
    level: str  # "ok" | "warn" | "danger"
    message: str
    rep: int | None = None


@dataclass
class RepReport:
    number: int
    start: int
    end: int
    bottom_frame: int
    min_knee: float
    max_lean: float
    max_heel: float
    seconds: float
    reached_bottom: bool
    errors: list[str] = field(default_factory=list)

    @property
    def good(self) -> bool:
        return not self.errors


class SquatCoachAgent:
    def __init__(self, rules: SquatRules | None = None, fps: float = 30.0):
        self.rules = rules or SquatRules()
        self.fps = fps
        self.reps: list[RepReport] = []
        self._in_rep = False
        self._warned: set[str] = set()  # чтобы не повторять одну подсказку 30 раз за повторение

    # ---------- модель мира ----------

    def _start_rep(self, p: Percept):
        self._in_rep = True
        self._start = p.frame
        self._min_knee, self._bottom_frame = p.knee_angle, p.frame
        self._max_lean = p.torso_lean
        self._max_heel = p.heel_lift
        self._reached_bottom = False
        self._warned = set()

    def _update_rep(self, p: Percept):
        if p.knee_angle < self._min_knee:
            self._min_knee, self._bottom_frame = p.knee_angle, p.frame
        self._max_lean = max(self._max_lean, p.torso_lean)
        self._max_heel = max(self._max_heel, p.heel_lift)
        self._reached_bottom |= p.phase == BOTTOM

    # ---------- цикл агента: восприятие -> обновление состояния -> действие ----------

    def step(self, p: Percept) -> list[Feedback]:
        actions: list[Feedback] = []
        r = self.rules

        if not self._in_rep:
            if p.phase != TOP:
                self._start_rep(p)
            return actions

        self._update_rep(p)

        # Мгновенные подсказки прямо во время движения
        if p.torso_lean > r.max_torso_lean + r.danger_extra_lean and "danger" not in self._warned:
            self._warned |= {"danger", "lean"}
            actions.append(Feedback(p.frame, "danger", "Опасный наклон! Высокая нагрузка на поясницу", len(self.reps) + 1))
        elif p.torso_lean > r.max_torso_lean and "lean" not in self._warned:
            self._warned.add("lean")
            actions.append(Feedback(p.frame, "warn", "Держи спину ровнее", len(self.reps) + 1))
        if p.heel_lift > r.max_heel_lift and "heels" not in self._warned:
            self._warned.add("heels")
            actions.append(Feedback(p.frame, "warn", "Пятки на пол", len(self.reps) + 1))

        # Повторение закончилось — вернулись в ВЕРХ
        if p.phase == TOP:
            self._in_rep = False
            if p.frame - self._start >= r.min_rep_frames:
                actions.append(self._finish_rep(p.frame))
        return actions

    def _finish_rep(self, end: int) -> Feedback:
        r = self.rules
        rep = RepReport(
            number=len(self.reps) + 1,
            start=self._start,
            end=end,
            bottom_frame=self._bottom_frame,
            min_knee=float(self._min_knee),
            max_lean=float(self._max_lean),
            max_heel=float(self._max_heel),
            seconds=(end - self._start) / self.fps,
            reached_bottom=self._reached_bottom,
        )
        if rep.min_knee > r.min_depth_angle:
            rep.errors.append("shallow")
        if rep.max_lean > r.max_torso_lean:
            rep.errors.append("lean")
        if rep.max_heel > r.max_heel_lift:
            rep.errors.append("heels")
        if rep.seconds < r.min_rep_seconds:
            rep.errors.append("fast")
        self.reps.append(rep)

        if rep.good:
            return Feedback(end, "ok", f"Повторение {rep.number}: отлично!", rep.number)
        level = "danger" if rep.max_lean > r.max_torso_lean + r.danger_extra_lean else "warn"
        text = "; ".join(ERROR_TEXT[e] for e in rep.errors)
        return Feedback(end, level, f"Повторение {rep.number}: {text}", rep.number)

    def run(self, percepts) -> list[Feedback]:
        """Прогоняет агента по всем кадрам по очереди (так же он работал бы с веб-камерой)."""
        log = []
        for p in percepts:
            log.extend(self.step(p))
        return log


def percepts_from(features: dict, path) -> list[Percept]:
    """Собирает восприятия агента из признаков и фаз HMM."""
    return [
        Percept(t, int(path[t]), float(features["knee_angle"][t]), float(features["torso_lean"][t]), float(features["heel_lift"][t]))
        for t in range(len(path))
    ]
