from __future__ import annotations

import subprocess
from pathlib import Path

DATASET_REPOSITORY_URL = "https://github.com/forecastingresearch/forecastbench-datasets.git"


def download_dataset(target: Path, revision: str, refresh: bool) -> str:
    """Download or refresh a ForecastBench source revision."""
    target = target.resolve()
    git_directory = target / ".git"

    if target.exists():
        if not git_directory.is_dir():
            raise FileExistsError(f"Refusing to overwrite non-git directory: {target}")
        if refresh:
            _run_git(target, "fetch", "--depth", "1", "origin", revision)
            _run_git(target, "checkout", "--detach", "FETCH_HEAD")
        return dataset_revision(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    _run_git(target, "init")
    _run_git(target, "remote", "add", "origin", DATASET_REPOSITORY_URL)
    _run_git(target, "fetch", "--depth", "1", "origin", revision)
    _run_git(target, "checkout", "--detach", "FETCH_HEAD")
    return dataset_revision(target)


def dataset_revision(dataset_root: Path) -> str | None:
    if not (dataset_root / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_git(target: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(target), *arguments], check=True)
