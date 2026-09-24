"""Inherited cricket owners and Markdown links must remain auditable."""

from pathlib import Path

from scripts.tools.support_matrix import EvidenceLevel, _load_task_name, build_support_rows

from .doc_checks import check_file_paths

ROOT = Path(__file__).resolve().parents[2]


def test_inherited_cricket_owner_is_configured_not_training_validated():
    path = ROOT / "src/unilab/conf/ppo/task/g1_cricket_impact_events_v1/mujoco.yaml"
    assert _load_task_name(path) == "G1CricketImpact"
    rows = [r for r in build_support_rows(ROOT) if r.task_slug == "g1_cricket_impact_events_v1"]
    assert len(rows) == 1
    assert rows[0].cells["mujoco"].level == EvidenceLevel.CONFIGURED


def test_path_checker_captures_markdown_destinations(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/existing.py").touch()
    doc = tmp_path / "README.md"
    assert check_file_paths("[source](src/existing.py)", doc, tmp_path) == []
    assert check_file_paths("[source](src/missing.py)", doc, tmp_path) == [
        f"{doc}: Path not found: src/missing.py"
    ]
    assert check_file_paths("`src/existing.py`", doc, tmp_path) == []
