"""Models under federation.

Deliberately small. The object of study is the integrity layer, not the
classifier: the model needs to be large enough to show realistic federated
dynamics and degradation under corrupted data, and small enough that a grid of
hundreds of runs finishes on a laptop CPU.

No BatchNorm, and no dropout, on purpose. BatchNorm carries running-statistic
buffers whose federated averaging is a research question in its own right and
would confound the integrity signal; dropout injects training-time randomness
that widens the reproducibility surface for no benefit here.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """~200k parameter CNN for 28x28 single-channel inputs."""

    def __init__(self, n_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, n_classes: int, seed: int, device: str = "cpu") -> nn.Module:
    """Construct a model with deterministic initialisation.

    All clients start from a single server-initialised model, so this is called
    once per run, not once per client.
    """
    if name != "small_cnn":
        raise ValueError(f"unknown model {name!r}")
    torch.manual_seed(seed)
    return SmallCNN(n_classes=n_classes).to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
