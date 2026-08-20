from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositorySource:
    name: str
    url: str
    revision: str


SOURCES = (
    RepositorySource(
        name="allenai-fermi",
        url="https://github.com/allenai/fermi.git",
        revision="dfd4ceec41ef5fa0fe63e24c6027f13730d39a36",
    ),
    RepositorySource(
        name="open-scioly-fermi",
        url="https://github.com/landy8697/open-scioly-fermi.git",
        revision="dea8a2595651160d4f247f8a47ad9ca4aa2ceeee",
    ),
)


def download_repositories(cache_directory: Path, refresh: bool = False) -> dict[str, str]:
    """Materialize the two pinned source repositories below ``cache_directory``."""
    revisions: dict[str, str] = {}
    for source in SOURCES:
        target = cache_directory / source.name
        _download_repository(target, source, refresh)
        revisions[source.name] = repository_revision(target)
    return revisions


def repository_revision(repository: Path) -> str:
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"Expected a Git repository at {repository}")
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_revisions(cache_directory: Path) -> dict[str, str]:
    return {source.name: repository_revision(cache_directory / source.name) for source in SOURCES}


def _download_repository(target: Path, source: RepositorySource, refresh: bool) -> None:
    target = target.resolve()
    git_directory = target / ".git"
    if target.exists():
        if not git_directory.is_dir():
            raise FileExistsError(f"Refusing to overwrite non-Git cache path: {target}")
        current_revision = repository_revision(target)
        if refresh or current_revision != source.revision:
            _run_git(target, "fetch", "--depth", "1", "origin", source.revision)
            _run_git(target, "checkout", "--detach", "FETCH_HEAD")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        _run_git(target, "init")
        _run_git(target, "remote", "add", "origin", source.url)
        _run_git(target, "fetch", "--depth", "1", "origin", source.revision)
        _run_git(target, "checkout", "--detach", "FETCH_HEAD")

    actual_revision = repository_revision(target)
    if actual_revision != source.revision:
        raise ValueError(
            f"{source.name} is at {actual_revision}, expected pinned revision {source.revision}. "
            "Run download with --refresh."
        )


def _run_git(target: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(target), *arguments], check=True)
