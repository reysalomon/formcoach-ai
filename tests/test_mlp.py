import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from formcoach.mlp import NumpyMLP, loss_and_grads


def _random_params(rng, d=3, hidden=5, k=3):
    return {"W1": rng.normal(size=(d, hidden)), "b1": rng.normal(size=hidden),
            "W2": rng.normal(size=(hidden, k)), "b2": rng.normal(size=k)}


def test_backprop_matches_numeric_gradient():
    """Главная проверка backprop: аналитический градиент = численному (loss(w+ε) - loss(w-ε)) / 2ε."""
    rng = np.random.default_rng(1)
    params = _random_params(rng)
    X = rng.normal(size=(8, 3))
    Y = np.eye(3)[rng.integers(0, 3, 8)]
    _, grads = loss_and_grads(params, X, Y, l2=0.01)
    eps = 1e-6
    for name, w in params.items():
        for idx in np.ndindex(w.shape):
            old = w[idx]
            w[idx] = old + eps
            plus, _ = loss_and_grads(params, X, Y, l2=0.01)
            w[idx] = old - eps
            minus, _ = loss_and_grads(params, X, Y, l2=0.01)
            w[idx] = old
            assert grads[name][idx] == pytest.approx((plus - minus) / (2 * eps), abs=1e-5), name


def test_learns_xor():
    """XOR не решается одним линейным перцептроном — нужен скрытый слой."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    y = np.array(["no", "yes", "yes", "no"] * 25)
    m = NumpyMLP(hidden=8, lr=0.1, epochs=300, batch_size=16).fit(X, y)
    assert (m.predict(X) == y).all()
    assert m.loss_curve_[-1] < m.loss_curve_[0]  # потеря падает


def test_probabilities_sum_to_one_and_same_seed_same_model():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), np.array(["a", "b", "c"] * 20)
    a = make_pipeline(StandardScaler(), NumpyMLP(epochs=20)).fit(X, y)
    b = make_pipeline(StandardScaler(), NumpyMLP(epochs=20)).fit(X, y)
    p = a.predict_proba(X)
    assert p.shape == (60, 3) and np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p, b.predict_proba(X))
    assert list(a.classes_) == ["a", "b", "c"]


def test_errors():
    with pytest.raises(RuntimeError):
        NumpyMLP().predict(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        NumpyMLP().fit(np.zeros((3, 2)), ["a", "b"])  # разная длина
    with pytest.raises(ValueError):
        NumpyMLP().fit(np.zeros((3, 2)), ["a", "a", "a"])  # один класс
    with pytest.raises(ValueError):
        NumpyMLP().fit(np.array([[0.0, np.nan], [1.0, 1.0]]), ["a", "b"])
