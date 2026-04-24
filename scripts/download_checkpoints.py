"""
download_checkpoints.py — pull trained checkpoints from HuggingFace Hub

Usage:
  python scripts/download_checkpoints.py --username Chizurin

  Download specific runs only:
  python scripts/download_checkpoints.py --username Chizurin --runs sms_only naive

Requirements:
  pip install huggingface_hub

Checkpoints saved to:
  checkpoints/sms_only/               ← Run A
  checkpoints/naive/                  ← Run B
  checkpoints/dann/                   ← Run C
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHECKPOINTS = ROOT / "checkpoints"

HF_RUNS = {
    "sms_only": {
        "repo_suffix": "spam-detector-sms-only",
        "local_dir": CHECKPOINTS / "sms_only",
        "mode": "snapshot",
    },
    "naive": {
        "repo_suffix": "spam-detector-naive",
        "local_dir": CHECKPOINTS / "naive",
        "mode": "snapshot",
    },
    "dann": {
        "repo_suffix": "spam-detector-dann",
        "local_dir": CHECKPOINTS / "dann",
        "mode": "snapshot",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="HuggingFace username who uploaded the checkpoints")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(HF_RUNS.keys()),
        default=list(HF_RUNS.keys()),
        help="Which runs to download (default: all)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    for run_key in args.runs:
        run = HF_RUNS[run_key]
        repo_id = f"{args.username}/{run['repo_suffix']}"
        local_dir = run["local_dir"]

        print(f"[{run_key}] ← {repo_id}")

        try:
            if run["mode"] == "snapshot":
                local_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    ignore_patterns=["README.md", ".gitattributes"],
                )
                files = list(local_dir.glob("*"))
                print(f"  {len(files)} files → {local_dir.relative_to(ROOT)}/")
                for f in sorted(files):
                    print(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

            else:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=run["filename"],
                    local_dir=str(local_dir),
                )
                f = Path(path)
                print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB) → {local_dir.relative_to(ROOT)}/")

            print(f"  Done\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue


if __name__ == "__main__":
    main()
