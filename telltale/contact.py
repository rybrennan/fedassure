"""Contact schedules: which nodes report in which rounds.

A surface ship or shore station reports every round. A submarine reports
only when it has a communications window. The schedule is the third seam on
`run_federated`: `participation(round_idx) -> list[int]`. The federation code
knows nothing about platforms.

Model of a silent round, stated so it is not mistaken for more than it is: a
node that is out of contact does nothing that round — it does not train, does
not report, and on its next contact it receives the current global model. That
is the "surface, sync, train, report" case. It is NOT the case where a boat
keeps training on a stale model while submerged and returns one large update
weeks later; that is asynchronous aggregation, a research question in its own
right, and out of scope for this harness (docs/stage3_fault_design.md s.7).

What this does let the harness measure: every temporal statistic in `detect`
is computed over a node's *own observed rounds*, so a boat is compared with its
last contact rather than with a round it was silent for, and time to detection
is counted in contacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSchedule:
    period: int = 1
    """A submarine node reports every `period` rounds. 1 = every round."""
    sub_nodes: tuple[int, ...] = ()
    """Nodes on the submarine schedule; every other node reports every round.
    Contacts are staggered: the i-th submarine's phase is i mod period."""

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError("period must be >= 1")
        if len(set(self.sub_nodes)) != len(self.sub_nodes) or any(n < 0 for n in self.sub_nodes):
            raise ValueError("sub_nodes must be distinct and >= 0")

    def phase(self, node: int) -> int:
        return self.sub_nodes.index(node) % self.period

    def in_contact(self, node: int, round_idx: int) -> bool:
        if node not in self.sub_nodes:
            return True
        return (round_idx + self.phase(node)) % self.period == 0

    def participants(self, round_idx: int, n_clients: int) -> list[int]:
        return [c for c in range(n_clients) if self.in_contact(c, round_idx)]

    def observed(self, n_rounds: int, n_clients: int) -> np.ndarray:
        """(rounds, clients) bool mask of contacts."""
        return np.array(
            [[self.in_contact(c, r) for c in range(n_clients)] for r in range(n_rounds)], dtype=bool
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sub_nodes"] = list(self.sub_nodes)
        return d

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]
