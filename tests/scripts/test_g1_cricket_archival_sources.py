"""Historical migration must never become a general live-source bypass."""

import hashlib
from pathlib import Path

import pytest
from scripts.evaluate_g1_cricket_swing import check_hashes
from scripts.g1_cricket_archival_sources import MIGRATED, check_retained_hashes

ROOT = Path(__file__).resolve().parents[2]


def test_only_listed_path_digest_pairs_use_verified_archive():
    check_retained_hashes(ROOT, MIGRATED)
    name = next(iter(MIGRATED))
    with pytest.raises(AssertionError):
        check_retained_hashes(ROOT, {name: "0" * 64})
    with pytest.raises(AssertionError):
        check_hashes(MIGRATED)


def test_unlisted_data_is_checked_against_live_bytes(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    check_retained_hashes(tmp_path, {path.name: expected})
    path.write_bytes(b"modified")
    with pytest.raises(AssertionError):
        check_retained_hashes(tmp_path, {path.name: expected})
