"""
upload_checkpoints.py — push trained checkpoints to HuggingFace Hub

Usage:
  python scripts/upload_checkpoints.py --username your-hf-username

  Upload specific runs only:
  python scripts/upload_checkpoints.py --username your-hf-username --runs sms_only naive

Requirements:
  pip install huggingface_hub
  huggingface-cli login   (run once to save your token)

Repos created:
  {username}/spam-detector-sms-only      ← Run A
  {username}/spam-detector-naive         ← Run B
  {username}/spam-detector-dann          ← Run C
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHECKPOINTS = ROOT / "checkpoints"

HF_RUNS = {
    "sms_only": {
        "repo_suffix": "spam-detector-sms-only",
        "description": "Run A — RoBERTa fine-tuned on SMS spam only (roberta-base)",
        "dir": CHECKPOINTS / "sms_only",
        "mode": "hf",
    },
    "naive": {
        "repo_suffix": "spam-detector-naive",
        "description": "Run B — RoBERTa fine-tuned on SMS + Discord (naive combined, weighted loss)",
        "dir": CHECKPOINTS / "naive",
        "mode": "hf",
    },
    "dann": {
        "repo_suffix": "spam-detector-dann",
        "description": "Run C — Domain-Adversarial RoBERTa (DANN) trained on SMS + Discord",
        "dir": CHECKPOINTS / "dann",
        "mode": "files",
    },
}


def upload_hf_model(api, repo_id: str, directory: Path):
    """Upload a HuggingFace-format checkpoint (model.safetensors + tokenizer)."""
    files = list(directory.glob("*"))
    # Skip intermediate epoch checkpoints if somehow present
    files = [f for f in files if f.is_file() and not f.name.startswith("checkpoint-")]
    print(f"  Uploading {len(files)} files from {directory.name}/")
    for f in files:
        print(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
        )


def upload_files(api, repo_id: str, files: list[Path]):
    """Upload a list of arbitrary files."""
    for f in files:
        print(f"  Uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="HuggingFace username")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(HF_RUNS.keys()),
        default=list(HF_RUNS.keys()),
        help="Which runs to upload (default: all)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Verify login
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}\n")
    except Exception:
        print("Not logged in. Run: huggingface-cli login")
        sys.exit(1)

    for run_key in args.runs:
        run = HF_RUNS[run_key]
        repo_id = f"{args.username}/{run['repo_suffix']}"

        # Check files exist before trying to upload
        if run["mode"] == "hf":
            if not run["dir"].exists():
                print(f"[{run_key}] Skipping — {run['dir']} not found")
                continue
        else:
            missing = [f for f in (run.get("files") or list(run["dir"].glob("*"))) if not f.exists()]
            if missing:
                print(f"[{run_key}] Skipping — missing: {[f.name for f in missing]}")
                continue

        print(f"[{run_key}] → {repo_id}")

        try:
            api.create_repo(repo_id=repo_id, exist_ok=True)

            # Write a minimal README so the repo isn't blank
            readme = f"# {run['repo_suffix']}\n\n{run['description']}\n\nPart of the Multi-Domain Spam & Phishing Detection project (COMP5415, UMass Lowell).\n"
            api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
            )

            if run["mode"] == "hf":
                upload_hf_model(api, repo_id, run["dir"])
            else:
                files = run.get("files") or [f for f in run["dir"].iterdir() if f.is_file()]
                upload_files(api, repo_id, files)

            print(f"  Done → https://huggingface.co/{repo_id}\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue


if __name__ == "__main__":
    main()
