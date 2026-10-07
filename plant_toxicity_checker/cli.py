"""Command-line interface: ``python -m plant_toxicity_checker <command>``."""

import argparse
import sys
from pathlib import Path

from plant_toxicity_checker.config import ASPCA_DETAILS_JSON_PATH
from plant_toxicity_checker.verdict import SCOPE_NOTE, Status

_HEADLINES = {
    Status.TOXIC: "TOXIC TO CATS - keep away from cats",
    Status.CAUTION: "CAUTION - treat as toxic, do not rely on this",
    Status.LIKELY_SAFE: "LIKELY SAFE for the main plant only "
    "(verified non-toxic, confident match)",
}


def verdict_to_dict(verdict, top_k=None):
    """JSON-serialisable form of a ``Verdict``.

    ``top_k`` only trims the listed predictions; the verdict itself was
    already computed from the full distribution.
    """
    entry = verdict.entry
    return {
        "status": verdict.status.value,
        "species": verdict.species,
        "confidence": verdict.confidence,
        "reasons": list(verdict.reasons),
        "scope_note": SCOPE_NOTE,
        "toxicity": None if entry is None else {
            "scientific_name": entry.scientific_name,
            "toxic_to_cats": entry.toxic_to_cats,
            "severity": entry.severity,
            "symptoms": list(entry.symptoms),
            "sources": list(entry.sources),
            "verified": entry.verified,
        },
        "predictions": [
            {"species": p.species, "confidence": p.confidence}
            for p in verdict.predictions[:top_k]
        ],
    }


def format_verdict(verdict, top_k=5):
    """Human-readable multi-line report (``top_k`` candidates shown)."""
    lines = [_HEADLINES[verdict.status]]
    if verdict.species is not None:
        lines.append(
            f"Best match: {verdict.species} ({verdict.confidence:.1%})"
        )
    lines.extend(f"- {reason}" for reason in verdict.reasons)
    entry = verdict.entry
    if entry is not None and entry.symptoms:
        lines.append("Symptoms: " + ", ".join(entry.symptoms))
    lines.append("Top candidates:")
    lines.extend(
        f"  {p.confidence:6.1%}  {p.species}"
        for p in verdict.predictions[:top_k]
    )
    lines.append(f"Note: {SCOPE_NOTE}")
    return "\n".join(lines)


def _run_scrape_aspca_details(args):
    from plant_toxicity_checker import aspca_detail

    entries = aspca_detail.load_entries()

    def progress(n, total, ok, failed):
        print(f"  {n}/{total} done ({ok} ok, {failed} failed)", flush=True)

    doc = aspca_detail.scrape_all(
        entries, out=args.out, refresh=args.refresh, limit=args.limit,
        on_progress=progress,
    )
    counts = doc["counts"]
    print(
        f"{counts['total']} plants "
        f"({counts['ok']} ok, {counts['failed']} failed) -> {args.out}"
    )
    print("First-draft data only: not verified. See README.")
    return 0


def _run_train_plantnet(args):
    from plant_toxicity_checker import train

    config = train.load_config(args.config)
    # Resuming always continues the same run's checkpoint/output
    # directory - --resume implies the run name, it isn't a separate
    # choice from --run-name.
    run_name = args.resume or args.run_name or Path(args.config).stem
    train.run_training(
        config, run_name,
        epochs_override=args.epochs, resume=args.resume is not None,
    )
    print(f"done: checkpoints/{run_name}/, outputs/{run_name}/")
    return 0


def _run_evaluate_plantnet(args):
    from plant_toxicity_checker import train

    result = train.run_evaluation(args.run, args.split)
    for k in (1, 3, 5, 10):
        print(
            f"top-{k}: {result[f'top_{k}']:.1%}   "
            f"avg-{k}: {result[f'avg_{k}']:.1%}"
        )
    return 0


def _run_species_breakdown(args):
    from plant_toxicity_checker import train

    result = train.run_species_breakdown(args.run, args.split)
    print(f"{len(result['species'])} species, split={args.split}")
    for name, stats in result["buckets"].items():
        if stats["n_species"] == 0:
            continue
        print(
            f"  train_count {name:>7}: {stats['n_species']:4d} species  "
            f"top1={stats['top1_accuracy']:.1%}  "
            f"top5={stats['top5_accuracy']:.1%}  "
            f"(n={stats['n']})"
        )
    out_path = f"outputs/{args.run}/species_breakdown_{args.split}.json"
    print(f"full per-species breakdown written to {out_path}")
    return 0


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser():
    """Build the argparse parser."""
    parser = argparse.ArgumentParser(prog="plant-toxicity-checker")
    sub = parser.add_subparsers(dest="command", required=True)

    scrape_details = sub.add_parser(
        "scrape-aspca-details",
        help="(Re-)scrape each ASPCA plant's own detail page, resuming "
             "from the cached pages on disk (one request per plant, ~10s "
             "apart; 983 plants takes ~2.7 hours)",
    )
    scrape_details.add_argument("--out", default=ASPCA_DETAILS_JSON_PATH)
    scrape_details.add_argument(
        "--refresh", action="store_true",
        help="re-download every page instead of using cached copies",
    )
    scrape_details.add_argument(
        "--limit", type=_positive_int, default=None,
        help="only scrape the first N plants (for a quick test run)",
    )
    scrape_details.set_defaults(func=_run_scrape_aspca_details)

    train_plantnet = sub.add_parser(
        "train-plantnet",
        help="Train the PlantNet-300K baseline species classifier "
             "(Roadmap Phase 1) - run this on the home PC's GPU, not "
             "locally (see scripts/remote/train.sh)",
    )
    train_plantnet.add_argument(
        "--config", required=True,
        help="training config JSON, e.g. configs/plantnet_baseline.json",
    )
    train_plantnet.add_argument(
        "--resume", default=None,
        help="resume this run name from its last checkpoint",
    )
    train_plantnet.add_argument(
        "--epochs", type=_positive_int, default=None,
        help="override the config's epoch count (e.g. --epochs 1 for a "
             "timing sanity check before the full run)",
    )
    train_plantnet.add_argument(
        "--run-name", default=None,
        help="ignored if --resume is given; otherwise defaults to the "
             "config file's stem",
    )
    train_plantnet.set_defaults(func=_run_train_plantnet)

    evaluate_plantnet = sub.add_parser(
        "evaluate-plantnet",
        help="Evaluate a trained run's checkpoint: top-1/3/5/10 and "
             "average-k accuracy",
    )
    evaluate_plantnet.add_argument(
        "--run", required=True, help="run name (checkpoints/<run>/)",
    )
    evaluate_plantnet.add_argument(
        "--split", choices=["val", "test"], default="test",
    )
    evaluate_plantnet.set_defaults(func=_run_evaluate_plantnet)

    species_breakdown = sub.add_parser(
        "species-breakdown",
        help="Per-species accuracy for a trained run, bucketed by "
             "training-image count - checks whether low-data species "
             "are dragging accuracy down, no retraining needed",
    )
    species_breakdown.add_argument(
        "--run", required=True, help="run name (checkpoints/<run>/)",
    )
    species_breakdown.add_argument(
        "--split", choices=["val", "test"], default="val",
    )
    species_breakdown.set_defaults(func=_run_species_breakdown)
    return parser


def main(argv=None):
    """Entry point; returns a process exit code."""
    from plant_toxicity_checker.train import ConfigError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ConfigError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
