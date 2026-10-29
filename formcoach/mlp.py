"""Неделя 9 — Перцептроны и нейросети: свой многослойный перцептрон на NumPy.

Никаких готовых нейросетевых библиотек: прямой проход, функция потерь,
обратное распространение ошибки (backpropagation) и градиентный спуск — всё здесь.

Архитектура:  x (7 признаков) -> [W1, b1] -> ReLU -> h (hidden нейронов) -> [W2, b2] -> softmax -> p (классы)

Формулы (для одного мини-батча из n примеров, Y — one-hot правильных ответов):
  z1 = X·W1 + b1,  h = max(0, z1)          — скрытый слой
  z2 = h·W2 + b2,  p = softmax(z2)          — вероятности классов
  loss = -mean(sum(Y·log p)) + l2/2·(|W1|² + |W2|²)   — перекрёстная энтропия + штраф за большие веса
Градиенты (цепное правило):
  dz2 = (p - Y) / n
  dW2 = hᵀ·dz2 + l2·W2,   db2 = sum(dz2)
  dh  = dz2·W2ᵀ,          dz1 = dh * [z1 > 0]   (производная ReLU: 1, если нейрон был активен)
  dW1 = Xᵀ·dz1 + l2·W1,   db1 = sum(dz1)
Обновление весов — градиентный спуск с моментом:  v = μ·v - lr·grad,  w = w + v

Класс совместим с sklearn (fit / predict / predict_proba / classes_), поэтому
его можно поставить в make_pipeline(StandardScaler(), NumpyMLP()) рядом с другими моделями.
Признаки обязательно нормировать (StandardScaler): углы ~100, а heel_above_toe ~0.1.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def softmax(z: np.ndarray) -> np.ndarray:
    # вычитаем максимум строки, чтобы exp не переполнился (результат тот же)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прямой проход. Возвращает (z1, h, p) — промежуточные значения нужны для backprop."""
    z1 = X @ params["W1"] + params["b1"]
    h = np.maximum(0.0, z1)
    p = softmax(h @ params["W2"] + params["b2"])
    return z1, h, p


def loss_and_grads(params: dict, X: np.ndarray, Y: np.ndarray, l2: float = 0.0) -> tuple[float, dict]:
    """Потеря на батче и её градиенты по всем весам (обратное распространение ошибки)."""
    n = len(X)
    z1, h, p = forward(params, X)
    loss = -np.sum(Y * np.log(p + 1e-12)) / n + l2 / 2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))

    dz2 = (p - Y) / n
    grads = {"W2": h.T @ dz2 + l2 * params["W2"], "b2": dz2.sum(axis=0)}
    dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
    grads |= {"W1": X.T @ dz1 + l2 * params["W1"], "b1": dz1.sum(axis=0)}
    return float(loss), grads


class NumpyMLP(BaseEstimator, ClassifierMixin):
    """Перцептрон с одним скрытым слоем.

    hidden     — число нейронов скрытого слоя
    lr         — шаг обучения (learning rate)
    epochs     — сколько раз проходим по всем данным
    batch_size — размер мини-батча
    l2         — штраф за большие веса (меньше переобучение)
    momentum   — доля прошлого шага, которая сохраняется (ускоряет спуск)
    seed       — для воспроизводимости: одинаковый seed -> одинаковая сеть
    """

    def __init__(self, hidden=16, lr=0.05, epochs=200, batch_size=64, l2=1e-3, momentum=0.9, seed=0):
        self.hidden, self.lr, self.epochs, self.batch_size = hidden, lr, epochs, batch_size
        self.l2, self.momentum, self.seed = l2, momentum, seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"Ожидаю X формы (n, признаки) и y длины n, получил {X.shape} и {y.shape}.")
        if np.isnan(X).any():
            raise ValueError("В X есть пропуски (NaN) — уберите такие строки перед обучением.")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Для обучения нужно хотя бы 2 класса.")
        Y = (y[:, None] == self.classes_[None, :]).astype(float)  # one-hot

        rng = np.random.default_rng(self.seed)
        d, k = X.shape[1], len(self.classes_)
        # инициализация He: разброс весов ~ sqrt(2/входов) — хорошо подходит для ReLU
        self.params_ = {
            "W1": rng.normal(0, np.sqrt(2 / d), (d, self.hidden)), "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, np.sqrt(2 / self.hidden), (self.hidden, k)), "b2": np.zeros(k),
        }
        velocity = {name: np.zeros_like(w) for name, w in self.params_.items()}
        self.loss_curve_ = []  # средняя потеря за эпоху — рисуем график обучения

        for _ in range(self.epochs):
            order = rng.permutation(len(X))  # каждую эпоху перемешиваем примеры
            total = 0.0
            for start in range(0, len(X), self.batch_size):
                batch = order[start:start + self.batch_size]
                loss, grads = loss_and_grads(self.params_, X[batch], Y[batch], self.l2)
                total += loss * len(batch)
                for name in self.params_:
                    velocity[name] = self.momentum * velocity[name] - self.lr * grads[name]
                    self.params_[name] += velocity[name]
            self.loss_curve_.append(total / len(X))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "params_"):
            raise RuntimeError("Модель ещё не обучена: сначала вызовите fit().")
        return forward(self.params_, np.asarray(X, dtype=float))[2]

    def predict(self, X) -> np.ndarray:
        p = self.predict_proba(X)  # сначала proba: она проверяет, что модель обучена
        return self.classes_[p.argmax(axis=1)]
