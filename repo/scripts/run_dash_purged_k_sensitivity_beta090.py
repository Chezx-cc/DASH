"""Memory-depth sensitivity for K in {1,2,3} with beta fixed at 0.90."""
from __future__ import annotations

import run_dash_purged_hyperparam_sensitivity as experiment


experiment.SETTINGS = tuple(
    {
        "axis": "cycles",
        "value": cycles,
        "cycles": cycles,
        "decay": 0.90,
    }
    for cycles in (1, 2, 3)
)


if __name__ == "__main__":
    experiment.main()
