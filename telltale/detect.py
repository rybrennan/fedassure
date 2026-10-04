"""Divergence statistics over probe reports.

Every function here consumes probability arrays and returns numbers. Nothing
here touches the model, the data, or the training loop. The array convention
throughout is

    probs[round, client, probe, class]      float, NaN for an absent client

as produced by `ProbeMonitor.probs_array`.

Three statistics, one per handoff bullet, plus the instrument self-check:

  level        mean JS divergence of a node's rows from its peers' leave-one-out
               consensus, z-scored within the round against that round's own
               median and MAD (the locally measured floor)
  self         mean JS divergence of a node's rows from its own rows last round;
               needs no peers
  persistence  whether the change in a node's offset-from-consensus, measured
               against its own trailing mean, is directional across rounds

No fault exists at this stage, so nothing here is a detector yet. These are
the quantities whose healthy distribution defines the false-alarm side of the
operating curve; the detection side is measured in stage 4 against stage 3's
faults. See docs/stage2_probe_design.md for why *level* alone cannot be the
detector.
"""

from __future__ import annotations

import numpy as np
import torch

from .probes import ProbeBattery, ProbeReport, score_probes

EPS = 1e-12
LN2 = float(np.log(2.0))
MAD_TO_SD = 1.4826


# ── divergences (row-wise over the last axis) ─────────────────────────────────


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) in nats, per row."""
    p, q = _clip(p), _clip(q)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen–Shannon divergence in nats, per row. Symmetric, bounded by ln 2."""
    p, q = _clip(p), _clip(q)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ── cross-node level ──────────────────────────────────────────────────────────


def loo_consensus(probs_round: np.ndarray) -> np.ndarray:
    """Leave-one-out peer mean, (clients, probes, classes).

    Row k is the mean over every *other* present client. A client is never
    compared against a consensus it contributed to. NaN where fewer than one
    peer is present.
    """
    p = np.asarray(probs_round, dtype=np.float64)
    present = ~np.isnan(p).any(axis=(1, 2))
    total = np.nansum(p[present], axis=0)
    n_present = int(present.sum())
    out = np.full_like(p, np.nan)
    for k in range(p.shape[0]):
        if present[k] and n_present >= 2:
            out[k] = (total - p[k]) / (n_present - 1)
        elif not present[k] and n_present >= 1:
            out[k] = total / n_present
    return out


def level_divergence(probs_round: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-client mean JS to the LOO consensus, and the offset vectors.

    Returns (level[clients], offset[clients, probes, classes]).
    """
    p = np.asarray(probs_round, dtype=np.float64)
    q = loo_consensus(p)
    level = np.full(p.shape[0], np.nan)
    offset = np.full_like(p, np.nan)
    for k in range(p.shape[0]):
        if not (np.isnan(p[k]).any() or np.isnan(q[k]).any()):
            level[k] = js(p[k], q[k]).mean()
            offset[k] = p[k] - q[k]
    return level, offset


def robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """z-scores against the median and scaled MAD of `values`, ignoring NaN.

    Returns (z, median, scale). If the MAD is zero the scale falls back to the
    standard deviation; if that is zero too every z is 0 — nothing can be
    outlying in a set of identical numbers.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return np.full_like(v, np.nan), float("nan"), float("nan")
    med = float(np.median(v[ok]))
    scale = MAD_TO_SD * float(np.median(np.abs(v[ok] - med)))
    if scale <= 0.0:
        scale = float(np.std(v[ok]))
    z = np.full_like(v, np.nan)
    z[ok] = 0.0 if scale <= 0.0 else (v[ok] - med) / scale
    return z, med, scale


def level_series(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Level statistics for every round.

    Returns dict with
      level   (rounds, clients)          mean JS to LOO consensus
      z       (rounds, clients)          within-round robust z
      offset  (rounds, clients, probes, classes)
      median  (rounds,)                  within-round median level
      scale   (rounds,)                  within-round robust scale
    """
    R, K = probs.shape[:2]
    level = np.full((R, K), np.nan)
    z = np.full((R, K), np.nan)
    offset = np.full(probs.shape, np.nan)
    median = np.full(R, np.nan)
    scale = np.full(R, np.nan)
    for r in range(R):
        level[r], offset[r] = level_divergence(probs[r])
        z[r], median[r], scale[r] = robust_z(level[r])
    return {"level": level, "z": z, "offset": offset, "median": median, "scale": scale}


# ── temporal self ─────────────────────────────────────────────────────────────


def _observed_rounds(a: np.ndarray, k: int) -> np.ndarray:
    """Round indices where client k's entry is fully finite."""
    x = a[:, k].reshape(a.shape[0], -1)
    return np.nonzero(~np.isnan(x).any(axis=1))[0]


def self_divergence(probs: np.ndarray) -> np.ndarray:
    """(rounds, clients) mean JS between a node's rows and its rows at its
    PREVIOUS CONTACT — the last round it reported, not necessarily the round
    before. NaN at a node's first contact and wherever it is absent."""
    R, K = probs.shape[:2]
    out = np.full((R, K), np.nan)
    for k in range(K):
        obs = _observed_rounds(probs, k)
        for i in range(1, len(obs)):
            out[obs[i], k] = js(probs[obs[i], k], probs[obs[i - 1], k]).mean()
    return out


# ── offset change and persistence ─────────────────────────────────────────────


def offset_change(offset: np.ndarray, window: int) -> np.ndarray:
    """c_t = d_t − mean of the node's `window` PREVIOUS CONTACTS' offsets.

    The level of a node's offset is confounded by legitimate heterogeneity; a
    skewed node has a large offset that is *stable*. Its change against the
    node's own trailing mean is what a moving feed produces and a stable skew
    does not. Trailing means run over the node's own observed rounds, so a
    boat that surfaces every k rounds is compared with its own last `window`
    contacts. NaN until `window` prior contacts exist.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    R, K = offset.shape[:2]
    out = np.full_like(offset, np.nan)
    for k in range(K):
        obs = _observed_rounds(offset, k)
        for i in range(window, len(obs)):
            out[obs[i], k] = offset[obs[i], k] - offset[obs[i - window : i], k].mean(axis=0)
    return out


def persistence(vectors: np.ndarray) -> float:
    """‖mean_t v_t‖ / mean_t ‖v_t‖ over the leading axis, in [0, 1].

    Zero-mean noise across T rounds gives roughly 1/√T; a shift that points
    the same way every round gives 1. NaN if any vector is missing or all are
    zero.
    """
    v = np.asarray(vectors, dtype=np.float64).reshape(vectors.shape[0], -1)
    if np.isnan(v).any():
        return float("nan")
    norms = np.linalg.norm(v, axis=1)
    denom = norms.mean()
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(v.mean(axis=0)) / denom)


def persistence_series(change: np.ndarray, span: int) -> np.ndarray:
    """(rounds, clients) persistence of the offset change over the node's
    trailing `span` CONTACTS ending at each contact. NaN until `span` changes
    exist for that node."""
    if span < 2:
        raise ValueError("span must be >= 2")
    R, K = change.shape[:2]
    out = np.full((R, K), np.nan)
    for k in range(K):
        obs = _observed_rounds(change, k)
        for i in range(span - 1, len(obs)):
            out[obs[i], k] = persistence(change[obs[i - span + 1 : i + 1], k])
    return out


# ── self-referenced level ─────────────────────────────────────────────────────


def self_referenced_level(level: np.ndarray, window: int) -> np.ndarray:
    """(rounds, clients) a node's level this round against its OWN trailing
    `window` contacts: (level_t − trailing mean) / trailing SD.

    Added after the first stage-3 run, not before: a slow bias drift doubled
    the target node's level while its cross-node z *fell*, because the fleet
    threshold is set by the most-skewed healthy node and the drift moved the
    target toward the fleet median. The level is confounded across nodes but
    not across a node's own history. Thresholds for this statistic still come
    from healthy runs only. NaN until `window` prior contacts exist, and where
    the trailing SD is zero.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    R, K = level.shape
    out = np.full((R, K), np.nan)
    for k in range(K):
        obs = _observed_rounds(level[:, :, None], k)
        for i in range(window, len(obs)):
            trailing = level[obs[i - window : i], k]
            sd = trailing.std(ddof=1)
            if sd > 0:
                out[obs[i], k] = (level[obs[i], k] - trailing.mean()) / sd
    return out


# ── cumulative sum against a fixed reference epoch ────────────────────────────


def cusum(
    level: np.ndarray,
    reference: slice,
    k: float = 0.5,
    pool: np.ndarray | None = None,
    min_ref: int = 5,
    two_sided: bool = False,
) -> np.ndarray:
    """(rounds, clients) one-sided CUSUM (Page, 1954) of a node's level against
    its OWN mean and SD over a fixed `reference` epoch.

        S_t = max(0, S_{t-1} + (level_t − μ_ref) / σ_ref − k)

    A step shows up in a sliding-window z; a slow ramp does not, because each
    round's increment sits inside the node's own noise. CUSUM accumulates the
    small consistent excess and is the standard estimator for "directional
    and persists across rounds". `k` is the allowance in SD units; 0.5 is the
    textbook default and is not tuned here. The reference epoch is the same
    fixed rounds for every node and every run, so it never sees an onset.
    Reference statistics and the accumulation run over the node's own
    contacts; a silent round neither adds nor decays. NaN through the
    reference epoch, for nodes with fewer than two reference contacts, and
    wherever the node is absent.

    `pool` is an optional (clients,) array of platform-class labels. The
    reference MEAN is always per node (heterogeneity gives every node its own
    level), but the reference SD is pooled within a class: the deviations of
    every node in the class from its own mean, together. A boat that reports
    every third round has three reference contacts, and a spread estimated
    from three points is what blew one healthy boat's ceiling to 26 in the
    submarine arm. Pooling is the standard remedy, decided from healthy runs.
    It applies only where it is needed: a class whose every node has at least
    `min_ref` reference contacts keeps per-node SDs, so a dense fleet is
    unchanged by passing `pool`; a class with a starved node is pooled.

    `two_sided` also accumulates deviations BELOW the reference and reports
    the larger of the two sums. Added after the stage-4 grid: a gain fault
    moved the node toward the fleet, a decrease in its level, which a
    one-sided accumulator cannot see. Its healthy ceiling is judged on the
    healthy runs before it is scored on any fault.
    """
    R, K = level.shape
    ref_rounds = set(range(R)[reference])
    if len(ref_rounds) < 2:
        raise ValueError("reference epoch must span >= 2 rounds")
    out = np.full((R, K), np.nan)
    stop = range(R)[reference].stop
    refs = {}
    for c in range(K):
        obs = _observed_rounds(level[:, :, None], c)
        ref = [r for r in obs if r in ref_rounds]
        if len(ref) >= 2:
            refs[c] = (obs, ref, level[ref, c].mean())
    pooled_sd: dict = {}
    if pool is not None:
        pool = np.asarray(pool)
        for cls in np.unique(pool):
            counts = [len(ref) for c, (_, ref, _) in refs.items() if pool[c] == cls]
            if counts and min(counts) >= min_ref:
                continue  # well-sampled class: per-node SD
            devs = np.concatenate(
                [level[ref, c] - mu for c, (_, ref, mu) in refs.items() if pool[c] == cls]
                or [np.array([])]
            )
            n_nodes = sum(1 for c in refs if pool[c] == cls)
            dof = len(devs) - n_nodes
            pooled_sd[cls] = float(np.sqrt((devs**2).sum() / dof)) if dof > 0 else 0.0
    for c, (obs, ref, mu) in refs.items():
        sd = pooled_sd.get(pool[c]) if pool is not None else None
        if sd is None:
            sd = level[ref, c].std(ddof=1)
        if sd <= 0:
            continue
        S_up = S_dn = 0.0
        for r in obs:
            if r < stop:
                continue
            z = (level[r, c] - mu) / sd
            S_up = max(0.0, S_up + z - k)
            S_dn = max(0.0, S_dn - z - k)
            out[r, c] = max(S_up, S_dn) if two_sided else S_up
    return out


# ── instrument self-check ─────────────────────────────────────────────────────


def check_reports(
    reports: list[ProbeReport],
    battery: ProbeBattery,
    previous: dict[int, torch.Tensor] | None = None,
    row_tol: float = 1e-4,
) -> list[str]:
    """Integrity issues on the probe pathway itself, independent of what the
    scores say. A degraded scorer and a degraded feed look alike in naive
    monitoring; this check is how they are told apart. Empty list = clean.

    `previous` maps client_id to that client's report from the prior round,
    for the replay check.
    """
    issues: list[str] = []
    shape = (battery.n_probes, battery.n_classes)
    for rep in reports:
        who = f"client {rep.client_id} round {rep.round_idx}"
        if rep.battery_fingerprint != battery.fingerprint:
            issues.append(f"{who}: battery fingerprint {rep.battery_fingerprint} != served {battery.fingerprint}")
        p = rep.probs
        if tuple(p.shape) != shape:
            issues.append(f"{who}: shape {tuple(p.shape)} != {shape}")
            continue
        if not torch.isfinite(p).all():
            issues.append(f"{who}: non-finite values")
            continue
        row_err = float((p.sum(dim=1) - 1.0).abs().max())
        if row_err > row_tol:
            issues.append(f"{who}: rows do not sum to 1 (max err {row_err:.2e})")
        if previous is not None and rep.client_id in previous:
            if torch.equal(p, previous[rep.client_id]):
                issues.append(f"{who}: report is bitwise identical to previous round (replay)")
    return issues


def scorer_is_deterministic(model: torch.nn.Module, battery: ProbeBattery) -> bool:
    """Scoring the same state twice must give bitwise-equal rows."""
    return torch.equal(score_probes(model, battery), score_probes(model, battery))


# ── healthy characterisation ──────────────────────────────────────────────────


def _nan_stat(fn, a: np.ndarray) -> float:
    """fn over the non-NaN entries; NaN (silently) if there are none."""
    a = np.asarray(a, dtype=np.float64)
    return float(fn(a[~np.isnan(a)])) if (~np.isnan(a)).any() else float("nan")


def characterise(
    probs: np.ndarray,
    sizes: list[int],
    tail_rounds: int = 10,
    window: int = 5,
    span: int = 5,
) -> list[dict]:
    """The healthy distribution of every statistic, per battery size.

    `probs` comes from one run over the largest battery; each size is its
    prefix. Everything is summarised over the final `tail_rounds` rounds so the
    training transient is excluded. Returns one dict per size with:

      level_median       typical mean-JS to consensus
      level_round_sd     SD across the tail rounds within a node, mean over
                         nodes: sampling + training noise, the temporal floor
      level_client_sd    SD across nodes within a round, mean over rounds:
                         heterogeneity + sampling noise
      level_z_max        largest |within-round z| any node reached
      self_median        typical temporal self-divergence
      persist_median/max persistence of the offset change (span rounds)
    """
    R = probs.shape[0]
    tail = slice(max(0, R - tail_rounds), R)
    out = []
    for n in sizes:
        sub = probs[:, :, :n, :]
        lv = level_series(sub)
        sd = self_divergence(sub)
        ch = offset_change(lv["offset"], window)
        ps = persistence_series(ch, span)

        level_tail = lv["level"][tail]
        out.append(
            {
                "n_probes": n,
                "level_median": _nan_stat(np.median, level_tail),
                "level_round_sd": _nan_stat(np.mean, np.nanstd(level_tail, axis=0)),
                "level_client_sd": _nan_stat(np.mean, np.nanstd(level_tail, axis=1)),
                "level_z_max": _nan_stat(np.max, np.abs(lv["z"][tail])),
                "self_median": _nan_stat(np.median, sd[tail]),
                "persist_median": _nan_stat(np.median, ps[tail]),
                "persist_max": _nan_stat(np.max, ps[tail]),
            }
        )
    return out
