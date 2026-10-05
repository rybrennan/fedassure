"""Dataset loading and non-IID partitioning.

The whole dataset is held in memory as tensors and batched by hand rather than
through a DataLoader. With 60k 28x28 images that costs ~180MB and removes
worker scheduling from the reproducibility surface — batch order becomes a
pure function of a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# FashionMNIST channel statistics (computed over the training split).
_NORM = {"fashion_mnist": (0.2860, 0.3530), "mnist": (0.1307, 0.3081)}

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Dataset:
    """A dataset held in memory as normalised tensors, train and test splits."""

    train_x: torch.Tensor  # (N, 1, 28, 28) float32, normalised
    train_y: torch.Tensor  # (N,) int64
    test_x: torch.Tensor
    test_y: torch.Tensor
    n_classes: int
    name: str

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return int(self.train_x.shape[0])


def load_dataset(name: str = "fashion_mnist", root: Path | None = None) -> Dataset:
    """Load a standard public dataset into normalised in-memory tensors."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    if name == "deepship":
        from .acoustic import load_deepship  # second modality; see acoustic.py

        return load_deepship(root / "deepship", cache=root / "deepship_28x28.pt")

    from torchvision import datasets  # imported lazily; only needed on first download

    ctor = {"fashion_mnist": datasets.FashionMNIST, "mnist": datasets.MNIST}.get(name)
    if ctor is None:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(_NORM) + ['deepship']}")

    train = ctor(root=str(root), train=True, download=True)
    test = ctor(root=str(root), train=False, download=True)
    mean, std = _NORM[name]

    def prep(split) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw uint8 split -> (N, 1, 28, 28) float32 scaled to [0, 1] then standardised,
        and int64 labels.
        """
        x = split.data.to(torch.float32).div_(255.0).sub_(mean).div_(std)
        return x.unsqueeze(1).contiguous(), split.targets.to(torch.int64).contiguous()

    train_x, train_y = prep(train)
    test_x, test_y = prep(test)
    return Dataset(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        n_classes=int(train_y.max().item()) + 1,
        name=name,
    )


def dirichlet_partition(
    labels: torch.Tensor | np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
    min_client_samples: int = 20,
    max_tries: int = 100,
) -> list[np.ndarray]:
    """Partition sample indices across clients with Dirichlet label skew.

    The standard construction (Hsu, Qi & Brown, 2019): for each class k, draw a
    proportion vector p ~ Dir(alpha * 1_N) over clients and split that class's
    indices accordingly. Small alpha concentrates each class on few clients
    (severe non-IID); large alpha approaches an IID split.

    This function is the source of the study's principal confound. Legitimate
    label skew makes one node's model genuinely differ from its peers, which is
    precisely the signature a naive integrity detector would flag as
    corruption. Characterising the detector across alpha is therefore not a
    robustness afterthought — it is the experiment.

    Returns one sorted index array per client; every index is assigned exactly
    once, and no client falls below `min_client_samples`.
    """
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    n_classes = int(y.max()) + 1
    total = len(y)

    if n_clients * min_client_samples > total:
        raise ValueError(
            f"cannot give {n_clients} clients >= {min_client_samples} samples "
            f"from {total} total"
        )

    idx_by_class = [np.where(y == k)[0] for k in range(n_classes)]

    for attempt in range(max_tries):
        # A distinct stream per attempt keeps redraws deterministic given the seed.
        rng = np.random.default_rng([seed, attempt])
        parts = _draw_partition(idx_by_class, n_clients, alpha, rng)
        if min(p.size for p in parts) >= min_client_samples:
            _assert_exact_partition(parts, total)
            return parts

    raise RuntimeError(
        f"could not draw a partition with >= {min_client_samples} samples per "
        f"client in {max_tries} tries at alpha={alpha}, n_clients={n_clients}. "
        "Raise alpha, lower min_client_samples, or use fewer clients."
    )


def _draw_partition(
    idx_by_class: list[np.ndarray], n_clients: int, alpha: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """One Dirichlet draw: every class split across clients, indices sorted per client.

    Per class the generator is advanced in a fixed order (shuffle, then the
    proportion vector), so a draw is a pure function of `rng`'s state.
    """
    buckets: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
    for idx_k in idx_by_class:
        idx_k = rng.permutation(idx_k)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        # Cut points along the shuffled class indices; np.split needs the
        # interior boundaries only, hence [:-1].
        cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(idx_k, cuts)):
            if chunk.size:
                buckets[client].append(chunk)
    return [
        np.sort(np.concatenate(b)) if b else np.empty(0, dtype=np.int64) for b in buckets
    ]


def _assert_exact_partition(parts: list[np.ndarray], total: int) -> None:
    """Every index used exactly once. Cheap, and catches silent data loss."""
    joined = np.concatenate(parts)
    if joined.size != total:
        raise AssertionError(f"partition covers {joined.size} of {total} samples")
    if np.unique(joined).size != total:
        raise AssertionError("partition assigns some index to more than one client")


def partition_label_matrix(
    labels: torch.Tensor | np.ndarray, parts: list[np.ndarray], n_classes: int
) -> np.ndarray:
    """(n_clients, n_classes) count matrix. Used for reporting skew."""
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    m = np.zeros((len(parts), n_classes), dtype=np.int64)
    for i, idx in enumerate(parts):
        counts = np.bincount(y[idx], minlength=n_classes)
        m[i] = counts
    return m


def skew_summary(matrix: np.ndarray) -> dict:
    """Scalar descriptors of how non-IID a partition actually came out.

    `mean_tv_from_uniform` is the mean total-variation distance between each
    client's label distribution and the uniform distribution: 0.0 is perfectly
    balanced, and it approaches 1 - 1/n_classes as clients specialise. Reported
    so that a run records the skew it actually drew, not just the alpha that
    was requested.
    """
    sizes = matrix.sum(axis=1)
    props = matrix / np.maximum(sizes[:, None], 1)
    uniform = 1.0 / matrix.shape[1]
    tv = 0.5 * np.abs(props - uniform).sum(axis=1)
    return {
        "client_sizes": sizes.tolist(),
        "min_client_size": int(sizes.min()),
        "max_client_size": int(sizes.max()),
        "mean_tv_from_uniform": float(tv.mean()),
        "max_tv_from_uniform": float(tv.max()),
    }
