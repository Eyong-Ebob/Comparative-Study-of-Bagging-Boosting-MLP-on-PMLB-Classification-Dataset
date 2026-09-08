import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / "bottom_hyperparam.pid"

sys.argv = [str(ROOT / "run_bottom_datasets_sequentially.py")]
PID_FILE.write_text(str(os.getpid()))
runpy.run_path(ROOT / "run_bottom_datasets_sequentially.py", run_name="__main__")
