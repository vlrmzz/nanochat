"""
Tests for the wandb-free file logger and the dashboard's data loading.

Run with:
    python -m pytest tests/test_logger.py -v
"""

import os
import json

import pytest

from nanochat.common import get_logger, DummyWandb, FileLogger, MultiLogger


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    return tmp_path


def test_dummy_when_run_is_dummy_or_non_master(base_dir, monkeypatch):
    monkeypatch.setenv("NANOCHAT_LOGGER", "file")
    assert isinstance(get_logger("proj", "dummy", {}), DummyWandb)
    assert isinstance(get_logger("proj", "myrun", {}, master_process=False), DummyWandb)
    assert not (base_dir / "metrics").exists()


def test_file_logger_writes_jsonl(base_dir, monkeypatch):
    monkeypatch.setenv("NANOCHAT_LOGGER", "file")
    run = get_logger("nanochat", "d12", {"depth": 12})
    assert isinstance(run, FileLogger)
    run.log({"step": 0, "train/loss": 3.5, "train/epoch": "0 pq: 1 rg: 2"})
    run.log({"step": 100, "val/bpb": 1.2, "centered_results": {"arc": 0.1}})
    run.finish()

    path = base_dir / "metrics" / "nanochat" / "d12.jsonl"
    assert path.exists()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["_config"] == {"depth": 12}
    assert rows[1]["step"] == 0 and rows[1]["train/loss"] == 3.5
    assert rows[2]["val/bpb"] == 1.2 and rows[2]["centered_results"] == {"arc": 0.1}
    assert all("_time" in r for r in rows)


def test_file_logger_appends_across_resumes(base_dir, monkeypatch):
    monkeypatch.setenv("NANOCHAT_LOGGER", "file")
    for i in range(2):
        run = get_logger("nanochat", "d12", None)
        run.log({"step": i})
        run.finish()
    path = base_dir / "metrics" / "nanochat" / "d12.jsonl"
    assert [json.loads(l)["step"] for l in path.read_text().splitlines()] == [0, 1]


def test_invalid_logger_mode(base_dir, monkeypatch):
    monkeypatch.setenv("NANOCHAT_LOGGER", "tensorboard")
    with pytest.raises(AssertionError):
        get_logger("proj", "myrun", {})


def test_multi_logger_fans_out():
    class Rec:
        def __init__(self):
            self.calls = []
        def log(self, d):
            self.calls.append(d)
        def finish(self):
            self.calls.append("finish")
    a, b = Rec(), Rec()
    m = MultiLogger([a, b])
    m.log({"x": 1})
    m.finish()
    assert a.calls == b.calls == [{"x": 1}, "finish"]


def test_dashboard_load_runs(base_dir, monkeypatch):
    from scripts.dashboard import load_runs
    monkeypatch.setenv("NANOCHAT_LOGGER", "file")
    run = get_logger("nanochat", "d12", {"depth": 12})
    run.log({"step": 0, "train/loss": 3.5, "train/epoch": "0 pq: 1 rg: 2"})
    run.log({"step": 100, "val/bpb": 1.2, "centered_results": {"arc": 0.1}})
    run.log({"step": 100, "core_metric": 0.05})
    # a half-written trailing line must not break loading
    with open(run.path, "a") as f:
        f.write('{"step": 200, "train/lo')
    run.finish()

    runs = load_runs(str(base_dir / "metrics"))
    assert list(runs) == ["nanochat/d12"]
    r = runs["nanochat/d12"]
    assert r["config"] == {"depth": 12}
    # strings and nested dicts are stripped, numeric scalars kept
    assert all(isinstance(row.pop("_time"), float) for row in r["rows"])
    assert r["rows"] == [
        {"step": 0, "train/loss": 3.5},
        {"step": 100, "val/bpb": 1.2},
        {"step": 100, "core_metric": 0.05},
    ]
