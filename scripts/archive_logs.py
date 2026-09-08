import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {".git", ".venv"}


def should_skip(p: Path) -> bool:
    for part in p.relative_to(ROOT).parts:
        if part in EXCLUDE:
            return True
    return False


def main():
    dest = ROOT / "archive" / "logs"
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in ROOT.rglob("*"):
        if p.is_file() and not should_skip(p):
            if p.suffix.lower() in {".log", ".pid"} or p.name == "download_log.txt":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{p.stem}_{ts}{p.suffix}"
                shutil.move(str(p), str(dest / new_name))
                moved += 1
    print(f"Archived {moved} log/pid files to {dest}")


if __name__ == "__main__":
    main()
