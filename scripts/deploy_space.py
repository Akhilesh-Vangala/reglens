"""Deploy the committed tree (HEAD) to a Hugging Face Docker Space.

    python scripts/deploy_space.py <hf-username>/reglens

Uses the token from `hf auth login`. Only tracked files are uploaded (git archive), so .env and local
caches never leave the machine. Spaces read their config from YAML front matter in README.md; it is
added to the uploaded copy only.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

FRONT_MATTER = """---
title: RegLens
emoji: 📜
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Governed RAG over banking regulations with verified citations
---

"""


def main(space: str) -> None:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=root)
    with tempfile.TemporaryDirectory() as tmp:
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(tmp, filter="data")
        readme = Path(tmp) / "README.md"
        readme.write_text(FRONT_MATTER + readme.read_text())
        api = HfApi()
        api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message=f"Deploy {commit}")
    print(f"https://huggingface.co/spaces/{space}  (building; first build takes a few minutes)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    main(sys.argv[1])
