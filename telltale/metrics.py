"""Operating-curve measurement: Pd, FAR, time to detection.

Everything here consumes per-round, per-node statistic series (as produced by
`detect.level_series` and `detect.persistence_series`) and a threshold. The
threshold comes from HEALTHY runs only, via `calibrate`; nothing here looks at
a fault when setting it. That ordering is the whole point: a study that tunes
its threshold on the faults it then reports detecting has measured nothing.

With a single healthy seed per alpha, a threshold set at that run's healthy
maximum has a false-alarm rate of exactly zero *on that run* by construction.
The false-alarm rate is therefore only meaningful on healthy runs with a
different `train_seed` from the calibration run. `false_alarm_rate` does not
enforce that, because it cannot see seeds; the grid script must.

Detection rule, deliberately the simplest one: a node is flagged in round t if
its statistic exceeds the threshold. A fault is detected if the target node
is flagged at or after onset; time to detection is the first such round minus
the onset. Non-target nodes flagged after the burn-in are false localisations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Threshold:
    """A detection threshold on one statistic, with how it was calibrated."""

    statistic: str
    value: float
    burn_in: int
    """Rounds excluded from calibration and from counting, because the
    training transient is not the operating regime."""
    method: str
    """'max' or 'quantile:<q>' over the healthy node-rounds after burn-in."""
    n_calibration: int = 0
    """Healthy node-rounds the threshold was taken over. A maximum over
    fifteen values is not a ceiling; report this next to every threshold."""

    def to_dict(self) -> dict:
        """Plain-dict form, for the results JSON."""
        return asdict(self)


def calibrate(
    healthy: np.ndarray, statistic: str, burn_in: int, quantile: float | None = None
) -> Threshold:
    """Threshold from a healthy (rounds, nodes) statistic series.

    `quantile=None` takes the healthy maximum after burn-in; otherwise the
    given quantile of the healthy node-rounds. NaN entries are ignored.
    """
    if not 0 <= burn_in < healthy.shape[0]:
        raise ValueError("burn_in must be within the run")
    pool = healthy[burn_in:]
    pool = pool[~np.isnan(pool)]
    if pool.size == 0:
        raise ValueError("no finite healthy values after burn_in")
    if quantile is None:
        return Threshold(statistic, float(pool.max()), burn_in, "max", int(pool.size))
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    return Threshold(statistic, float(np.quantile(pool, quantile)), burn_in, f"quantile:{quantile}", int(pool.size))


def flags(series: np.ndarray, thr: Threshold) -> np.ndarray:
    """(rounds, nodes) bool: statistic strictly above threshold, after burn-in.
    NaN never flags."""
    out = np.zeros(series.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        above = series > thr.value
    above[np.isnan(series)] = False
    out[thr.burn_in :] = above[thr.burn_in :]
    return out


def calibrate_by_class(
    healthy: np.ndarray, classes: np.ndarray, statistic: str, burn_in: int, quantile: float | None = None
) -> dict:
    """One Threshold per platform class, each from that class's healthy
    node-rounds only. A submarine's healthy ceiling is not a ship's."""
    classes = np.asarray(classes)
    return {
        cls: calibrate(healthy[:, classes == cls], statistic, burn_in, quantile)
        for cls in np.unique(classes)
    }


def flags_by_class(series: np.ndarray, thresholds: dict, classes: np.ndarray) -> np.ndarray:
    """(rounds, nodes) bool: each node judged against its own class's threshold."""
    classes = np.asarray(classes)
    out = np.zeros(series.shape, dtype=bool)
    for cls, thr in thresholds.items():
        cols = classes == cls
        out[:, cols] = flags(series[:, cols], thr)
    return out


def false_alarm_rate(healthy: np.ndarray, thr: Threshold) -> float:
    """Fraction of healthy node-rounds after burn-in that are flagged."""
    f = flags(healthy, thr)[thr.burn_in :]
    return float(f.mean()) if f.size else float("nan")


@dataclass(frozen=True)
class DetectionOutcome:
    """How one faulted run scored against its target node and the rest of the fleet."""

    detected: bool
    detection_round: int | None
    time_to_detection: int | None
    """Rounds from onset to first flag on the target node; 0 = the onset round."""
    false_localisations: int
    """Non-target node-rounds flagged after burn-in."""
    target_flag_count: int
    contacts_to_detection: int | None = None
    """Target contacts from onset up to and including the detection round;
    1 = caught at the first report after onset. Equals time_to_detection + 1
    when the node reports every round."""

    def to_dict(self) -> dict:
        """Plain-dict form, for the results JSON."""
        return asdict(self)


def score_fault_run(
    series: np.ndarray,
    thr: Threshold,
    target: int,
    onset: int,
    observed: np.ndarray | None = None,
) -> DetectionOutcome:
    """Hit / miss / time to detection for one fault run's statistic series.

    `observed` is an optional (rounds, nodes) contact mask; without it every
    node is taken to report every round.
    """
    return score_flags(flags(series, thr), thr.burn_in, target, onset, observed)


def score_flags(
    f: np.ndarray, burn_in: int, target: int, onset: int, observed: np.ndarray | None = None
) -> DetectionOutcome:
    """Score a precomputed (rounds, nodes) flag matrix, e.g. from `flags_by_class`."""
    R, K = f.shape
    if not 0 <= target < K:
        raise ValueError("target node out of range")
    start = max(onset, burn_in)
    hits = np.nonzero(f[start:, target])[0]
    detection_round = int(start + hits[0]) if hits.size else None
    others = np.ones(K, dtype=bool)
    others[target] = False
    contacts = None
    if detection_round is not None:
        mask = np.ones(R, dtype=bool) if observed is None else observed[:, target]
        contacts = int(mask[onset : detection_round + 1].sum())
    return DetectionOutcome(
        detected=detection_round is not None,
        detection_round=detection_round,
        time_to_detection=None if detection_round is None else detection_round - onset,
        false_localisations=int(f[burn_in:, others].sum()),
        target_flag_count=int(f[start:, target].sum()),
        contacts_to_detection=contacts,
    )


def detection_probability(outcomes: list[DetectionOutcome]) -> float:
    """Fraction of fault runs detected. Over seeds, this is Pd at one grid point."""
    if not outcomes:
        return float("nan")
    return float(np.mean([o.detected for o in outcomes]))
