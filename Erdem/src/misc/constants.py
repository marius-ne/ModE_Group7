from pathlib import Path

# == Project directories =======================================
PROJ_ROOT = Path(__file__).parent.parent.parent.absolute()
OPT_DIR = PROJ_ROOT / "src" / "optimization"
SAMPLING_DIR = PROJ_ROOT / "src" / "sampling"
RESULTS_DIR = PROJ_ROOT / "results"