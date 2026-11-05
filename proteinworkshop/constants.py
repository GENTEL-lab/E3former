"""Project paths without machine-specific configuration."""

import os
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SRC_PATH.parent
HYDRA_CONFIG_PATH = SRC_PATH / "config"
DATA_PATH = Path(os.environ.get("E3FORMER_DATA", PROJECT_PATH / "data"))
