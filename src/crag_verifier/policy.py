"""Three-way action triage and utility-optimal thresholds.

The evaluator outputs p = P(a gold abstract is among the top-3 retrieved). Each action has
an expected utility that is linear in p,

    EU_a(p) = p * U[a, retrieval correct] + (1 - p) * U[a, retrieval wrong],

where U is measured on SciFact train by actually running every action on every claim.
The thresholds are the points where the upper envelope of these lines switches action,
so they are only as good as the calibration of p.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class Action(StrEnum):
    INCORRECT = "incorrect"  # abstain: answer NOINFO with no evidence
    AMBIGUOUS = "ambiguous"  # rewrite the query, re-retrieve, merge and re-rank
    CORRECT = "correct"  # keep the retrieved abstracts and refine them

    @property
    def order(self) -> int:
        return _ORDER[self]


_ORDER = {Action.INCORRECT: 0, Action.AMBIGUOUS: 1, Action.CORRECT: 2}
# Work each action costs; exact expected-utility ties go to the cheaper action.
_COST = {Action.INCORRECT: 0, Action.CORRECT: 1, Action.AMBIGUOUS: 2}
ACTIONS: tuple[Action, ...] = (Action.INCORRECT, Action.AMBIGUOUS, Action.CORRECT)


@dataclass(frozen=True)
class Thresholds:
    """p < lower -> INCORRECT, p > upper -> CORRECT, otherwise AMBIGUOUS.

    Ties go to the middle action: at a boundary the service hedges instead of committing.
    ``lower == upper`` means there is no AMBIGUOUS band. The single cut point then goes to
    INCORRECT, except a cut at 0, which is how a rule that never abstains is written.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= self.upper <= 1.0):
            raise ValueError(f"need 0 <= lower <= upper <= 1, got {self.lower}, {self.upper}")

    def to_dict(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


def decide_many(p: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """Action index (position in ``ACTIONS``) for every confidence in ``p``."""
    p = np.asarray(p, dtype=np.float64)
    lower, upper = thresholds.lower, thresholds.upper
    codes = np.ones(p.shape, dtype=np.int64)
    codes[p < lower] = 0
    codes[p > upper] = 2
    if lower == upper:
        codes[p == lower] = 2 if lower == 0.0 else 0
    return codes


def decide(p: float, thresholds: Thresholds) -> Action:
    return ACTIONS[int(decide_many(np.array([p]), thresholds)[0])]


@dataclass(frozen=True)
class UtilityTable:
    """Mean per-claim utility of each action given whether retrieval was correct."""

    correct: dict[Action, float]  # retrieval correct (y = 1)
    wrong: dict[Action, float]  # retrieval wrong (y = 0)

    @classmethod
    def from_outcomes(cls, utilities: dict[Action, np.ndarray], y: np.ndarray) -> UtilityTable:
        y = np.asarray(y).astype(bool)
        if y.all() or not y.any():
            raise ValueError("need claims with both correct and wrong retrieval")
        return cls(
            correct={a: float(np.mean(u[y])) for a, u in utilities.items()},
            wrong={a: float(np.mean(u[~y])) for a, u in utilities.items()},
        )

    def expected(self, p: np.ndarray) -> np.ndarray:
        """(n, 3) expected utility of each action in ACTIONS order."""
        p = np.asarray(p, dtype=np.float64)[:, None]
        hi = np.array([self.correct[a] for a in ACTIONS])[None, :]
        lo = np.array([self.wrong[a] for a in ACTIONS])[None, :]
        return p * hi + (1.0 - p) * lo

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "retrieval_correct": {a.value: v for a, v in self.correct.items()},
            "retrieval_wrong": {a.value: v for a, v in self.wrong.items()},
        }


def envelope(table: UtilityTable) -> list[tuple[float, float, Action]]:
    """Segments of [0, 1] on which each action maximises expected utility."""
    slopes = {a: table.correct[a] - table.wrong[a] for a in ACTIONS}
    cuts = {0.0, 1.0}
    for i, a in enumerate(ACTIONS):
        for b in ACTIONS[i + 1 :]:
            ds = slopes[a] - slopes[b]
            if abs(ds) > 1e-15:
                t = (table.wrong[b] - table.wrong[a]) / ds
                if 0.0 < t < 1.0:
                    cuts.add(float(t))
    points = sorted(cuts)
    segments: list[tuple[float, float, Action]] = []
    for lo, hi in itertools.pairwise(points):
        mid = 0.5 * (lo + hi)
        best = max(ACTIONS, key=lambda a: (table.wrong[a] + slopes[a] * mid, -_COST[a]))
        if segments and segments[-1][2] == best:
            segments[-1] = (segments[-1][0], hi, best)
        else:
            segments.append((lo, hi, best))
    return segments


def fit_thresholds(table: UtilityTable, p_train: np.ndarray) -> tuple[Thresholds, str]:
    """Utility-optimal thresholds and the method that produced them.

    When the envelope visits actions in INCORRECT -> AMBIGUOUS -> CORRECT order its
    switch points are the Bayes-optimal thresholds. Otherwise no threshold rule can follow
    the envelope, so we search the monotone rules for the one with the highest expected
    utility over the cross-validated train probabilities.
    """
    segments = envelope(table)
    orders = [s[2].order for s in segments]
    if orders == sorted(orders):
        start = {action: lo for lo, _, action in segments}
        upper = start.get(Action.CORRECT, 1.0)
        lower = start.get(Action.AMBIGUOUS, upper)
        if Action.INCORRECT not in start:
            lower = 0.0
        return Thresholds(lower=lower, upper=upper), "envelope"
    return _grid_thresholds(table, p_train), "grid"


def _grid_thresholds(table: UtilityTable, p: np.ndarray, steps: int = 101) -> Thresholds:
    eu = table.expected(p)
    rows = np.arange(len(p))
    grid = np.linspace(0.0, 1.0, steps)
    best: tuple[float, Thresholds] | None = None
    for i, lower in enumerate(grid):
        for upper in grid[i:]:
            candidate = Thresholds(float(lower), float(upper))
            total = float(eu[rows, decide_many(p, candidate)].sum())
            if best is None or total > best[0] + 1e-12:
                best = (total, candidate)
    assert best is not None
    return best[1]


def action_counts(actions: list[Action]) -> dict[str, int]:
    return {a.value: sum(1 for x in actions if x == a) for a in ACTIONS}


def thresholds_from_dict(d: dict[str, Any]) -> Thresholds:
    return Thresholds(lower=float(d["lower"]), upper=float(d["upper"]))
