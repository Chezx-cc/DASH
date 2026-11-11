"""Independent 54-config run of the selected default DASH configuration."""
from __future__ import annotations

import run_dash_purged_hyperparam_sensitivity as experiment


experiment.SETTINGS = (
    {
        "axis": "default_full_configuration",
        "value": "K3_beta0.90_P96+168+336",
        "cycles": 3,
        "decay": 0.90,
        "is_default": True,
    },
)


if __name__ == "__main__":
    experiment.main()
