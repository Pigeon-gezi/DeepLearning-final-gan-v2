from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Avoid OpenMP affinity failures on some Windows/virtualized course machines.
os.environ.setdefault("KMP_AFFINITY", "disabled")

from final_gan.cli import main


if __name__ == "__main__":
    main()
