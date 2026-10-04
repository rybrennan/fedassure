from __future__ import annotations

import numpy as np
import pytest
import torch

from telltale.data import (
    dirichlet_partition,
    partition_label_matrix,
    skew_summary,
)


def make_labels(n_per_class: int = 200, n_classes: int = 10) -> torch.Tensor:
    return torch.arange(n_classes).repeat_interleave(n_per_class)


def test_partition_is_exact_and_disjoint():
    """Every sample lands with exactly one client. Silent data loss here would
    bias every downstream measurement, so it is asserted rather than assumed."""
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=8, alpha=0.5, seed=0)

    joined = np.concatenate(parts)
    assert joined.size == len(y)
    assert np.unique(joined).size == len(y)
    assert set(joined.tolist()) == set(range(len(y)))


def test_partition_respects_minimum_client_size():
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=8, alpha=0.1, seed=3, min_client_samples=25)
    assert min(p.size for p in parts) >= 25


def test_partition_is_deterministic_given_seed():
    y = make_labels()
    a = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=42)
    b = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=42)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)


def test_different_seeds_give_different_partitions():
    y = make_labels()
    a = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=1)
    b = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=2)
    assert any(pa.size != pb.size or not np.array_equal(pa, pb) for pa, pb in zip(a, b))


def test_low_alpha_is_more_skewed_than_high_alpha():
    """The knob has to actually do what the study claims it does.

    Averaged over seeds, because a single Dirichlet draw is noisy enough that
    one low-alpha partition can come out milder than one high-alpha partition.
    """
    y = make_labels()

    def mean_skew(alpha: float) -> float:
        vals = []
        for seed in range(8):
            parts = dirichlet_partition(y, n_clients=8, alpha=alpha, seed=seed)
            m = partition_label_matrix(y, parts, n_classes=10)
            vals.append(skew_summary(m)["mean_tv_from_uniform"])
        return float(np.mean(vals))

    assert mean_skew(0.1) > mean_skew(1.0) > mean_skew(100.0)


def test_near_iid_alpha_approaches_uniform():
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=8, alpha=500.0, seed=0)
    m = partition_label_matrix(y, parts, n_classes=10)
    assert skew_summary(m)["mean_tv_from_uniform"] < 0.10


def test_label_matrix_totals_match_partition_sizes():
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=5, alpha=0.7, seed=11)
    m = partition_label_matrix(y, parts, n_classes=10)
    assert m.sum() == len(y)
    for i, p in enumerate(parts):
        assert m[i].sum() == p.size


def test_impossible_minimum_raises():
    y = make_labels(n_per_class=2, n_classes=10)  # 20 samples total
    with pytest.raises(ValueError, match="cannot give"):
        dirichlet_partition(y, n_clients=10, alpha=0.5, seed=0, min_client_samples=50)


def test_unsatisfiable_partition_raises_after_retries():
    y = make_labels(n_per_class=12, n_classes=10)  # 120 samples, 20 clients
    with pytest.raises(RuntimeError, match="could not draw a partition"):
        dirichlet_partition(
            y, n_clients=20, alpha=0.01, seed=0, min_client_samples=5, max_tries=5
        )
