"""Logistic regression, Platt and isotonic calibration, and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Newton-Raphson (IRLS) for sum(log-loss) + 0.5 * l2 * ||w||^2; intercept unpenalised.

    Targets may be soft (in [0, 1]), which is what Platt scaling needs.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    design = np.hstack([x, np.ones((n, 1))])
    penalty = np.full(d + 1, l2)
    penalty[-1] = 0.0
    theta = np.zeros(d + 1)

    def objective(t: np.ndarray) -> float:
        z = design @ t
        # log(1 + e^z) - y z, written to stay finite for large |z|
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(penalty * t * t))

    current = objective(theta)
    for _ in range(max_iter):
        p = sigmoid(design @ theta)
        grad = design.T @ (p - y) + penalty * theta
        weights = p * (1.0 - p)
        hess = design.T @ (design * weights[:, None]) + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        while True:
            candidate = theta - scale * step
            value = objective(candidate)
            if value <= current + 1e-12 or scale < 1e-8:
                break
            scale *= 0.5
        theta = candidate
        improvement = current - value
        current = value
        if np.max(np.abs(scale * step)) < tol or improvement < tol * max(1.0, abs(current)):
            break
    return theta[:-1], float(theta[-1])


@dataclass
class LogisticModel:
    """Standardised logistic regression: the retrieval evaluator's raw scorer."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> LogisticModel:
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        coef, intercept = fit_logistic((x - mean) / scale, y, l2=l2)
        return cls(mean, scale, coef, intercept)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogisticModel:
        return cls(
            np.array(d["mean"], dtype=np.float64),
            np.array(d["scale"], dtype=np.float64),
            np.array(d["coef"], dtype=np.float64),
            float(d["intercept"]),
        )


class Calibrator(Protocol):
    method: str

    def predict(self, scores: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class SigmoidOnly:
    """No calibration: the logistic model's own probability."""

    method: str = "raw"

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(scores, dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}


@dataclass
class PlattCalibrator:
    """p = sigmoid(a * score + b), fitted with Platt's (1999) smoothed targets."""

    a: float
    b: float
    method: str = "platt"

    @classmethod
    def fit(cls, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        # Smoothed targets keep the fit finite when the scores separate the classes.
        targets = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        coef, intercept = fit_logistic(scores[:, None], targets, l2=0.0)
        return cls(float(coef[0]), intercept)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(scores, dtype=np.float64) + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "a": self.a, "b": self.b}


def pool_adjacent_violators(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit of y (already ordered by x)."""
    values: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        values.append(yi)
        weights.append(wi)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            w_new = weights[-2] + weights[-1]
            v_new = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w_new
            c_new = counts[-2] + counts[-1]
            del values[-2:], weights[-2:], counts[-2:]
            values.append(v_new)
            weights.append(w_new)
            counts.append(c_new)
    return np.repeat(np.array(values), counts)


@dataclass
class IsotonicCalibrator:
    """Monotone step fit via PAV, linearly interpolated and clipped like sklearn's."""

    x: np.ndarray
    y: np.ndarray
    method: str = "isotonic"

    @classmethod
    def fit(cls, scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        unique, inverse = np.unique(scores, return_inverse=True)
        weight = np.bincount(inverse).astype(np.float64)
        mean_y = np.bincount(inverse, weights=y) / weight
        fitted = pool_adjacent_violators(mean_y, weight)
        return cls(unique, fitted)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if len(self.x) == 1:
            return np.full(scores.shape, self.y[0])
        return np.interp(scores, self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "x": self.x.tolist(), "y": self.y.tolist()}


def calibrator_from_dict(d: dict[str, Any]) -> Calibrator:
    method = d["method"]
    if method == "platt":
        return PlattCalibrator(float(d["a"]), float(d["b"]))
    if method == "isotonic":
        return IsotonicCalibrator(np.array(d["x"]), np.array(d["y"]))
    if method == "raw":
        return SigmoidOnly()
    raise ValueError(f"unknown calibration method {method!r}")


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "platt":
        return PlattCalibrator.fit(scores, y)
    if method == "isotonic":
        return IsotonicCalibrator.fit(scores, y)
    if method == "raw":
        return SigmoidOnly()
    raise ValueError(f"unknown calibration method {method!r}")


# Diagnostics ---------------------------------------------------------------------------


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    return np.minimum((p * n_bins).astype(np.int64), n_bins - 1)


def reliability_bins(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | None]]:
    """Equal-width bins on [0, 1]; the last bin is closed so p = 1 lands in it.

    Empty bins report ``None`` for their confidence and frequency.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = _bin_index(p, n_bins)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        bins.append(
            {
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "count": count,
                "mean_confidence": float(p[mask].mean()) if count else None,
                "observed_frequency": float(y[mask].mean()) if count else None,
            }
        )
    return bins


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("ECE of an empty sample is undefined")
    y = np.asarray(y, dtype=np.float64)
    idx = _bin_index(p, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total / len(p)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U statistic normalised to [0, 1]; ties count one half."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
