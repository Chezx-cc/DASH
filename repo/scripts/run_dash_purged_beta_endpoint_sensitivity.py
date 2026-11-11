"""Endpoint extension for beta sensitivity at beta in {0.95, 1.00}."""
from __future__ import annotations

import run_dash_purged_hyperparam_sensitivity as experiment


experiment.SETTINGS = (
    {
        "axis": "recent_decay",
        "value": 0.95,
        "cycles": 3,
        "decay": 0.95,
    },
    {
        "axis": "recent_decay",
        "value": 1.00,
        "cycles": 3,
        "decay": 1.00,
    },
)


if __name__ == "__main__":
    experiment.main()
