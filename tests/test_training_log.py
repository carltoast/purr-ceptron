"""Tests for parsing per-epoch summaries out of a train.py run's log,
and for stitching a resumed run's log fragments back into one history."""

import subprocess

from plant_toxicity_checker import training_log
from plant_toxicity_checker.training_log import (
    fetch_all_epochs,
    list_remote_logs,
    parse_epochs,
)

SAMPLE_LOG = """\
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth"
  batch 100/7621  running_loss=4.7951  elapsed=71s
  batch 200/7621  running_loss=3.9102  elapsed=142s
epoch 1/30  train_loss=2.0175  val_top1=0.6496  val_top5=0.8653  time=5182.1s
  batch 100/7621  running_loss=1.4704  elapsed=70s
epoch 2/30  train_loss=1.3559  val_top1=0.6866  val_top5=0.8919  time=5239.6s
"""


def test_parses_every_completed_epoch_in_order():
    records = parse_epochs(SAMPLE_LOG)
    assert [r["epoch"] for r in records] == [1, 2]
    assert records[0]["total_epochs"] == 30
    assert records[0]["train_loss"] == 2.0175
    assert records[0]["val_top1"] == 0.6496
    assert records[0]["val_top5"] == 0.8653
    assert records[0]["time_seconds"] == 5182.1
    assert records[1]["train_loss"] == 1.3559


def test_ignores_batch_progress_and_other_lines():
    only_progress = """\
  batch 100/7621  running_loss=4.7951  elapsed=71s
  batch 200/7621  running_loss=3.9102  elapsed=142s
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth"
"""
    assert parse_epochs(only_progress) == []


def test_empty_log_returns_no_epochs():
    assert parse_epochs("") == []


def test_no_completed_epoch_yet_returns_empty_list():
    line = "  batch 100/7621  running_loss=4.79  elapsed=71s"
    assert parse_epochs(line) == []


def test_parses_ema_fields_when_present():
    line = (
        "epoch 1/30  train_loss=2.87  val_top1=0.6271  val_top5=0.8522  "
        "ema_val_top1=0.6100  ema_val_top5=0.8400  time=10208.3s\n"
    )
    records = parse_epochs(line)
    assert records[0]["val_top1"] == 0.6271
    assert records[0]["ema_val_top1"] == 0.6100
    assert records[0]["ema_val_top5"] == 0.8400


def test_ema_fields_are_none_when_absent():
    records = parse_epochs(
        "epoch 1/30  train_loss=2.0  val_top1=0.60  val_top5=0.80  "
        "time=100.0s\n"
    )
    assert records[0]["ema_val_top1"] is None
    assert records[0]["ema_val_top5"] is None


# --- list_remote_logs / fetch_all_epochs --------------------------------
# (mock subprocess.run / the fetch layer rather than hit a real host, same
# offline-by-default convention as the rest of this project's network code)


def _fake_run(stdout):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
    return run


def test_list_remote_logs_glob_excludes_unrelated_longer_run_names(
    monkeypatch,
):
    # Regression test: asking for "plantnet_convnextv2_tiny_toxicity"
    # must not also match the unrelated run
    # "plantnet_convnextv2_tiny_toxicity_v2.log" - a real bug this
    # project hit (that "_v2" run's data silently overwrote the base
    # run's history for every epoch number they shared). The ssh
    # command itself must glob "<base>.log" and "<base>-*.log" only -
    # never a bare "<base>*.log" that a longer, unrelated name could
    # also satisfy.
    captured = {}

    def fake_run(args, **kwargs):
        captured["command"] = args[-1]
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    list_remote_logs("plantnet_convnextv2_tiny_toxicity")

    command = captured["command"]
    assert "plantnet_convnextv2_tiny_toxicity.log" in command
    assert "plantnet_convnextv2_tiny_toxicity-*.log" in command
    assert "plantnet_convnextv2_tiny_toxicity*.log" not in command


def test_list_remote_logs_strips_directory_and_extension(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run(
            "/home/carl_toast/projects/cat-project/logs/plantnet-baseline.log\n"
            "/home/carl_toast/projects/cat-project/logs/"
            "plantnet-baseline-resume.log\n"
        ),
    )
    names = list_remote_logs("plantnet-baseline")
    assert names == ["plantnet-baseline", "plantnet-baseline-resume"]


def test_list_remote_logs_handles_no_matches(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(""))
    assert list_remote_logs("plantnet-baseline") == []


def test_fetch_all_epochs_merges_and_sorts_across_log_fragments(
    monkeypatch,
):
    first_log = (
        "epoch 1/30  train_loss=2.0  val_top1=0.60  val_top5=0.80  "
        "time=100.0s\n"
        "epoch 2/30  train_loss=1.5  val_top1=0.65  val_top5=0.85  "
        "time=100.0s\n"
    )
    second_log = (
        "epoch 3/30  train_loss=1.2  val_top1=0.70  val_top5=0.90  "
        "time=100.0s\n"
    )
    logs = {"plantnet-baseline": first_log, "plantnet-baseline-resume":
             second_log}

    monkeypatch.setattr(
        training_log, "list_remote_logs",
        lambda base, *a, **kw: ["plantnet-baseline",
                                 "plantnet-baseline-resume"],
    )
    monkeypatch.setattr(
        training_log, "fetch_remote_log",
        lambda name, *a, **kw: logs[name],
    )

    records = fetch_all_epochs("plantnet-baseline")
    assert [r["epoch"] for r in records] == [1, 2, 3]
    assert records[2]["train_loss"] == 1.2


def test_fetch_all_epochs_last_write_wins_on_duplicate_epoch(monkeypatch):
    # Defensive case: if a resume somehow re-logs an epoch already seen,
    # the result must still have exactly one record for that epoch.
    first_log = (
        "epoch 5/30  train_loss=1.0  val_top1=0.70  val_top5=0.90  "
        "time=100.0s\n"
    )
    second_log = (
        "epoch 5/30  train_loss=0.9  val_top1=0.71  val_top5=0.91  "
        "time=100.0s\n"
    )
    logs = {"a": first_log, "b": second_log}

    monkeypatch.setattr(
        training_log, "list_remote_logs",
        lambda base, *a, **kw: ["a", "b"],
    )
    monkeypatch.setattr(
        training_log, "fetch_remote_log", lambda name, *a, **kw: logs[name],
    )

    records = fetch_all_epochs("whatever")
    assert len(records) == 1
    assert records[0]["train_loss"] == 0.9
