"""fedassure — a measurement harness for federated aggregation integrity.

Scope, held deliberately narrow:

    detector + fault injection + measured operating curves

Explicitly out of scope: edge appliance integration, a bespoke federated
learning framework, deployment tooling, or anything resembling a product.

Build order:
    1. clean baseline federation            (this layer — fedavg / data / models)
    2. probe batteries and divergence       (probes / detect)  — done
    3. fault injection with known ground truth (faults)  — done
    4. Pd / FAR / time-to-detection curves  (metrics)
"""

from __future__ import annotations

from .config import FedConfig
from .data import (
    Dataset,
    dirichlet_partition,
    load_dataset,
    partition_label_matrix,
    skew_summary,
)
from .contact import ContactSchedule
from .faults import FaultSpec, make_fault
from .fedavg import (
    ClientUpdate,
    FedResult,
    RoundRecord,
    aggregate,
    evaluate,
    run_federated,
    seed_everything,
)
from .metrics import Threshold, calibrate, false_alarm_rate, score_fault_run
from .models import SmallCNN, build_model, count_parameters
from .probes import (
    ProbeBattery,
    ProbeConfig,
    ProbeMonitor,
    ProbeReport,
    build_probe_battery,
    payload_bytes,
    score_probes,
)

__version__ = "0.1.0"

__all__ = [
    "FedConfig",
    "Dataset",
    "load_dataset",
    "dirichlet_partition",
    "partition_label_matrix",
    "skew_summary",
    "ClientUpdate",
    "RoundRecord",
    "FedResult",
    "run_federated",
    "aggregate",
    "evaluate",
    "seed_everything",
    "SmallCNN",
    "build_model",
    "count_parameters",
    "ProbeBattery",
    "ProbeConfig",
    "ProbeMonitor",
    "ProbeReport",
    "build_probe_battery",
    "payload_bytes",
    "score_probes",
    "FaultSpec",
    "make_fault",
    "ContactSchedule",
    "Threshold",
    "calibrate",
    "false_alarm_rate",
    "score_fault_run",
]
