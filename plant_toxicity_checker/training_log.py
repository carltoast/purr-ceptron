"""Read a train.py run's progress: fetch its log from the remote training
host and parse the per-epoch summary lines out of it.

Kept separate from ``train.py`` on purpose: this is read-only reporting
code for notebooks/dashboards, not part of the training loop itself.
"""

import re

from plant_toxicity_checker.config import REPO_ROOT

DEFAULT_HOST = "homepc"

_EPOCH_LINE = re.compile(
    r"^epoch (?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
    r"train_loss=(?P<train_loss>[\d.]+)\s+"
    r"val_top1=(?P<val_top1>[\d.]+)\s+"
    r"val_top5=(?P<val_top5>[\d.]+)\s+"
    r"(?:ema_val_top1=(?P<ema_val_top1>[\d.]+)\s+"
    r"ema_val_top5=(?P<ema_val_top5>[\d.]+)\s+)?"
    r"time=(?P<time_seconds>[\d.]+)s"
)


def parse_epochs(log_text):
    """Structured per-epoch records from a ``train.py`` run's log text.

    Only matches the one-line-per-completed-epoch summaries (not the
    more frequent in-epoch batch-progress lines); returns a list of
    dicts in log order, empty if no epoch has completed yet.

    ``ema_val_top1``/``ema_val_top5`` are only present (as floats) for
    runs trained with ``ema_decay`` set - ``None`` for every other run,
    rather than the key being missing, so callers can build a
    DataFrame from a mix of EMA and non-EMA runs without a KeyError.
    """
    records = []
    for line in log_text.splitlines():
        match = _EPOCH_LINE.match(line.strip())
        if match is None:
            continue
        fields = match.groupdict()
        records.append({
            "epoch": int(fields["epoch"]),
            "total_epochs": int(fields["total_epochs"]),
            "train_loss": float(fields["train_loss"]),
            "val_top1": float(fields["val_top1"]),
            "val_top5": float(fields["val_top5"]),
            "ema_val_top1": (
                float(fields["ema_val_top1"])
                if fields["ema_val_top1"] else None
            ),
            "ema_val_top5": (
                float(fields["ema_val_top5"])
                if fields["ema_val_top5"] else None
            ),
            "time_seconds": float(fields["time_seconds"]),
        })
    return records


def fetch_remote_log(run_name, host=DEFAULT_HOST, remote_dir=None):
    """Fetch ``logs/<run_name>.log`` from the remote training host over
    ssh; returns the raw text.

    ``remote_dir`` defaults to the same ``~/projects/<repo folder name>``
    path ``scripts/remote/`` uses (mirrors ``_common.sh``'s
    ``REMOTE_DIR``). Raises ``subprocess.CalledProcessError`` if the ssh
    call fails (host unreachable, run doesn't exist yet, ...) - never
    silently returns partial/stale data.
    """
    import subprocess

    remote_dir = remote_dir or f"~/projects/{REPO_ROOT.name}"
    result = subprocess.run(
        ["ssh", host, f"cat {remote_dir}/logs/{run_name}.log"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def list_remote_logs(base_run_name, host=DEFAULT_HOST, remote_dir=None):
    """Every ``logs/<base_run_name>.log`` or ``logs/<base_run_name>-
    <anything>.log`` file on the remote host, as bare run names (no
    directory, no ``.log``).

    Exists because of how ``--resume`` works: ``train.sh``'s tmux
    session/log is named after its own run-name *argument*, not the
    ``--resume`` target, so resuming ``plantnet-baseline`` under a
    different run-name (e.g. ``plantnet-baseline-resume``) continues the
    same run's checkpoints but writes to a SEPARATE log file. This finds
    every log fragment a base run has accumulated across such resumes,
    so ``fetch_all_epochs`` can stitch them back into one history.

    Deliberately NOT a bare ``<base_run_name>*.log`` glob: that would
    also match an unrelated run whose name happens to start with this
    one (e.g. asking for ``plantnet_convnextv2_tiny_toxicity`` would
    also match ``plantnet_convnextv2_tiny_toxicity_v2.log`` - a
    completely different run, not a resume of this one) - a real bug
    this project hit twice (the `convnextv2_tiny_sanity*` runs, then
    the `_v2` run name), silently corrupting the base run's history
    with the unrelated run's data for every epoch number they share.
    Resume suffixes always start with ``-`` (see the example above),
    so only that separator counts as "the same run, continued."
    """
    import subprocess

    remote_dir = remote_dir or f"~/projects/{REPO_ROOT.name}"
    result = subprocess.run(
        ["ssh", host,
         f"ls {remote_dir}/logs/{base_run_name}.log "
         f"{remote_dir}/logs/{base_run_name}-*.log 2>/dev/null || true"],
        capture_output=True, text=True, check=True,
    )
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        filename = line.rsplit("/", 1)[-1]
        if filename.endswith(".log"):
            filename = filename[: -len(".log")]
        names.append(filename)
    return names


def fetch_all_epochs(base_run_name, host=DEFAULT_HOST, remote_dir=None):
    """Every completed-epoch record for ``base_run_name``, merged across
    it and any of its ``--resume``d continuations, sorted by epoch
    number.

    If a resume somehow re-logged an epoch already present elsewhere,
    the last one encountered wins - this never duplicates an epoch
    number in the result.
    """
    records = {}
    for log_name in list_remote_logs(base_run_name, host, remote_dir):
        text = fetch_remote_log(log_name, host, remote_dir)
        for record in parse_epochs(text):
            records[record["epoch"]] = record
    return [records[epoch] for epoch in sorted(records)]
