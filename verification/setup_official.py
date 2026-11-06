"""Setup script to clone and pin official RanDumb repository into verification/_official/.

This script is for Huy to prepare the official author code for E9 verification.
The official code is cloned into verification/_official/ which is ignored by git (.gitignore).
DO NOT commit any files from the official repo into the team's repository.
"""

import os
import subprocess
import sys
from pathlib import Path

OFFICIAL_REPO_URL = "https://github.com/drimpossible/RanDumb.git"
OFFICIAL_DIR = Path(__file__).resolve().parent / "_official"
# You can pin a specific commit hash here once determined, e.g. "HEAD" or exact SHA
PINNED_COMMIT = "HEAD"


def clone_or_update_official_repo():
    print(f"Target directory: {OFFICIAL_DIR}")
    
    if OFFICIAL_DIR.exists() and (OFFICIAL_DIR / ".git").exists():
        print("Official repository already exists. Fetching latest updates...")
        try:
            subprocess.run(["git", "fetch", "origin"], cwd=OFFICIAL_DIR, check=True)
        except Exception as e:
            print(f"Error fetching repo: {e}")
            return False
    else:
        print(f"Cloning official repository from {OFFICIAL_REPO_URL}...")
        OFFICIAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", OFFICIAL_REPO_URL, str(OFFICIAL_DIR)],
                check=True
            )
        except Exception as e:
            print(f"Error cloning repo: {e}")
            return False

    # Checkout pinned commit
    print(f"Checking out commit: {PINNED_COMMIT}...")
    try:
        subprocess.run(["git", "checkout", PINNED_COMMIT], cwd=OFFICIAL_DIR, check=True)
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=OFFICIAL_DIR,
            check=True,
            capture_output=True,
            text=True
        )
        commit_sha = commit_res.stdout.strip()
        print(f"Successfully checked out official RanDumb at commit: {commit_sha}")
        
        # Save commit SHA to a text file for traceability in W&B runs
        commit_file = Path(__file__).resolve().parent / "official_commit_sha.txt"
        commit_file.write_text(commit_sha + "\n", encoding="utf-8")
        print(f"Recorded commit SHA in {commit_file.name}")
        return True
    except Exception as e:
        print(f"Error during checkout or getting commit SHA: {e}")
        return False


if __name__ == "__main__":
    success = clone_or_update_official_repo()
    if not success:
        sys.exit(1)
    print("Verification environment setup complete.")

