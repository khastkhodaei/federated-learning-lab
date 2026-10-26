"""Pytest configuration and repository-wide test fixtures."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


def _snapshot_reports_dir(reports_dir: Path) -> dict[str, tuple[int, int]]:
    """Captures a snapshot of all files in the reports directory.

    Returns:
        Mapping of relative file path string to (size_bytes, mtime_ns).
    """
    if not reports_dir.is_dir():
        return {}

    snapshot: dict[str, tuple[int, int]] = {}
    for p in reports_dir.rglob("*"):
        if p.is_file():
            stat = p.stat()
            rel_path = str(p.relative_to(reports_dir))
            snapshot[rel_path] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


@pytest.fixture(autouse=True)
def guard_reports_directory(request: pytest.FixtureRequest) -> Generator[None, Any, None]:
    """Guards the repository reports/ directory against unauthorized writes by tests.

    Snapshots the repository reports/ tree before each test and fails the test if any
    file is added, removed, or modified.
    """
    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"

    before_snapshot = _snapshot_reports_dir(reports_dir)

    yield

    after_snapshot = _snapshot_reports_dir(reports_dir)

    test_id = request.node.nodeid

    # Check for added files
    added = set(after_snapshot.keys()) - set(before_snapshot.keys())
    if added:
        changed_path = min(added)
        pytest.fail(
            f"Test '{test_id}' illegally added file under repo reports/: {changed_path}"
        )

    # Check for removed files
    removed = set(before_snapshot.keys()) - set(after_snapshot.keys())
    if removed:
        changed_path = min(removed)
        pytest.fail(
            f"Test '{test_id}' illegally removed file under repo reports/: {changed_path}"
        )

    # Check for modified files
    modified = [
        p
        for p in before_snapshot
        if p in after_snapshot and before_snapshot[p] != after_snapshot[p]
    ]
    if modified:
        changed_path = min(modified)
        pytest.fail(
            f"Test '{test_id}' illegally modified file under repo reports/: {changed_path}"
        )
