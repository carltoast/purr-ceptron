"""Command-line interface: ``python -m plant_toxicity_checker <command>``."""

import argparse
import sys
from pathlib import Path

from plant_toxicity_checker import species_match
from plant_toxicity_checker.config import (
    ASPCA_DETAILS_JSON_PATH,
    DEFAULT_BACKBONE_RUN,
    DEFAULT_RUN,
)
from plant_toxicity_checker.verdict import SCOPE_NOTE, Status

_HEADLINES = {
    Status.TOXIC: "TOXIC TO CATS - keep away from cats",
    Status.CAUTION: "CAUTION - treat as toxic, do not rely on this",
    Status.LIKELY_SAFE: "LIKELY SAFE for the main plant only "
    "(verified non-toxic, confident match)",
}

# What the toxicity table says about one candidate, in reading order of
# how much it actually tells you about *that* species. Genus-level
# wordings deliberately say the species itself is not listed - the
# reader is meant to see the gap, not read a relative's entry as this
# plant's.
_CANDIDATE_LABELS = {
    "exact - toxic": "TOXIC to cats",
    "exact - non-toxic": "Safe for cats",
    "exact - unknown": "In the table, toxicity not recorded",
    "genus - toxic (direct)": "TOXIC to cats (whole genus listed)",
    "genus - toxic (inferred)": "A relative is toxic to cats",
    "genus - non-toxic (direct)":
        "Genus listed non-toxic, this species not listed",
    "genus - non-toxic (inferred)":
        "A relative is non-toxic, this species not listed",
    "genus - unknown": "Genus in the table, toxicity not recorded",
    "no match": "Not in the toxicity table",
}


def _candidate_toxicity(species, index):
    """What the toxicity table knows about one predicted species, so a
    reader can weigh the alternatives themselves instead of trusting
    the headline alone: ``{"toxicity_category", "toxicity_note",
    "verified"}``, ready to merge into a prediction dict.

    An exact row that is not ``verified`` never reads as a plain
    "Safe for cats" - unverified is said out loud, because the whole
    verdict logic treats unverified as not-safe.
    """
    result = species_match.lookup_toxicity(species, index)
    category = species_match.toxicity_category(result)
    note = _CANDIDATE_LABELS[category]
    verified = None if result.status != "exact" else result.exact_row.verified
    if category in ("exact - toxic", "exact - non-toxic") and not verified:
        note += ", unverified"
    return {
        "toxicity_category": category,
        "toxicity_note": note,
        "verified": verified,
    }


def verdict_to_dict(verdict, table, top_k=None):
    """JSON-serialisable form of a ``Verdict``.

    ``top_k`` only trims the listed predictions; the verdict itself was
    already computed from the full distribution.
    """
    entry = verdict.entry
    index = species_match.build_toxicity_index(table)
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
            {
                "species": p.species,
                "confidence": p.confidence,
                **_candidate_toxicity(p.species, index),
            }
            for p in verdict.predictions[:top_k]
        ],
    }


def format_verdict(verdict, table, top_k=5):
    """Human-readable multi-line report (``top_k`` candidates shown,
    each with what the toxicity table says about it)."""
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
    index = species_match.build_toxicity_index(table)
    lines.extend(
        f"  {p.confidence:6.1%}  {p.species}: "
        f"{_candidate_toxicity(p.species, index)['toxicity_note']}"
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

    result = train.run_evaluation(
        args.run, args.split, epoch=args.epoch, use_ema=args.ema,
    )
    print(
        f"{args.run}  epoch {result['trained_epoch']}  "
        f"({result['weights']} weights)  split={args.split}"
    )
    for k in (1, 3, 5, 10):
        print(
            f"top-{k}: {result[f'top_{k}']:.1%}   "
            f"avg-{k}: {result[f'avg_{k}']:.1%}"
        )
    print(
        f"macro-avg top-1: {result['macro_top_1']:.1%}   "
        f"macro-avg top-5: {result['macro_top_5']:.1%}"
        f"   (per-species mean - the paper's headline metric)"
    )
    confusion = result["toxic_confusion"]
    if confusion["confusion_rate"] is not None:
        print(
            f"toxic/non-toxic confusion: "
            f"{confusion['confusion_rate']:.2%}   "
            f"ambiguous: {confusion['ambiguous_rate']:.2%}   "
            f"(n={confusion['n_known_side']})"
        )
    return 0


def _run_species_breakdown(args):
    from plant_toxicity_checker import train

    result = train.run_species_breakdown(
        args.run, args.split, epoch=args.epoch,
    )
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
    suffix = f"_epoch{args.epoch}" if args.epoch is not None else ""
    out_path = (
        f"outputs/{args.run}/species_breakdown_{args.split}{suffix}.json"
    )
    print(f"full per-species breakdown written to {out_path}")
    return 0


def _run_extract_features(args):
    from plant_toxicity_checker import decoupled

    out_path = decoupled.extract_features(args.run, args.split)
    print(f"done: {out_path}")
    return 0


def _run_train_classifier(args):
    from plant_toxicity_checker import decoupled

    config = decoupled.load_config(args.config)
    features_file = decoupled.features_path(args.features_run, "train")
    run_name = args.run_name or Path(args.config).stem
    head_path = decoupled.train_classifier_head(
        features_file, config, run_name, args.features_run,
    )
    print(f"done: {head_path}")
    return 0


def _run_evaluate_classifier(args):
    from plant_toxicity_checker import decoupled

    result = decoupled.evaluate_classifier_head(args.run, args.split)
    for k in (1, 3, 5, 10):
        print(
            f"top-{k}: {result[f'top_{k}']:.1%}   "
            f"avg-{k}: {result[f'avg_{k}']:.1%}"
        )
    confusion = result["toxic_confusion"]
    if confusion["n_known_side"]:
        print(
            f"toxic confusion: {confusion['confusion_rate']:.1%}  "
            f"ambiguous: {confusion['ambiguous_rate']:.1%}  "
            f"(n={confusion['n_known_side']})"
        )
    else:
        print("toxic confusion: no samples with a known side")
    return 0


def _run_classifier_species_breakdown(args):
    from plant_toxicity_checker import decoupled

    result = decoupled.classifier_head_species_breakdown(args.run, args.split)
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


def _run_check(args):
    import json

    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.pipeline import check_image
    from plant_toxicity_checker.plantnet_classifier import PlantNetClassifier
    from plant_toxicity_checker.toxicity import load_table

    classifier = PlantNetClassifier(
        args.run, head_run=args.head_run, epoch=args.epoch,
        use_ema=args.ema,
    )
    table = load_table(TOXICITY_TABLE_PATH)
    verdict = check_image(args.image, classifier, table)

    if args.json:
        print(json.dumps(verdict_to_dict(verdict, table, args.top_k)))
    else:
        print(format_verdict(verdict, table, args.top_k or 5))
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
        help="Evaluate a trained run's checkpoint: top-1/3/5/10, "
             "average-k, and toxic/non-toxic confusion rate",
    )
    evaluate_plantnet.add_argument(
        "--run", required=True, help="run name (checkpoints/<run>/)",
    )
    evaluate_plantnet.add_argument(
        "--split", choices=["val", "test"], default="test",
    )
    evaluate_plantnet.add_argument(
        "--epoch", type=_positive_int, default=None,
        help="evaluate checkpoints/<run>/epoch_<N>.pt instead of latest.pt "
             "- for comparing candidate checkpoints",
    )
    evaluate_plantnet.add_argument(
        "--ema", action="store_true",
        help="evaluate the EMA shadow weights instead of the live ones "
             "(only for runs trained with an 'ema_decay' config key)",
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
    species_breakdown.add_argument(
        "--epoch", type=_positive_int, default=None,
        help="load checkpoints/<run>/epoch_<N>.pt instead of latest.pt "
             "- e.g. to compare bucket accuracy across several already"
             "-trained epochs",
    )
    species_breakdown.set_defaults(func=_run_species_breakdown)

    extract_features = sub.add_parser(
        "extract-features",
        help="Cache a trained backbone's penultimate-layer features for "
             "one split (Roadmap Phase 2) - run this on the home PC's "
             "GPU, not locally",
    )
    extract_features.add_argument(
        "--run", required=True,
        help="backbone run name (checkpoints/<run>/)",
    )
    extract_features.add_argument(
        "--split", choices=["train", "val", "test"], required=True,
    )
    extract_features.set_defaults(func=_run_extract_features)

    train_classifier = sub.add_parser(
        "train-classifier",
        help="Train a classifier head on cached backbone features "
             "(Roadmap Phase 2/3) - cheap enough for CPU, no GPU needed",
    )
    train_classifier.add_argument(
        "--config", required=True,
        help="decoupled-training config JSON, e.g. "
             "configs/decoupled_cb_loss.json",
    )
    train_classifier.add_argument(
        "--features-run", required=True,
        help="backbone run whose cached train-split features to use "
             "(see extract-features)",
    )
    train_classifier.add_argument(
        "--run-name", default=None,
        help="defaults to the config file's stem",
    )
    train_classifier.set_defaults(func=_run_train_classifier)

    evaluate_classifier = sub.add_parser(
        "evaluate-classifier",
        help="Evaluate a trained classifier head: top-1/3/5/10, "
             "average-k, and toxic/non-toxic confusion rate",
    )
    evaluate_classifier.add_argument(
        "--run", required=True, help="run name (checkpoints/<run>/)",
    )
    evaluate_classifier.add_argument(
        "--split", choices=["val", "test"], default="test",
    )
    evaluate_classifier.set_defaults(func=_run_evaluate_classifier)

    classifier_species_breakdown = sub.add_parser(
        "classifier-species-breakdown",
        help="Per-species accuracy for a trained classifier head, "
             "bucketed by training-image count - the Phase 2/3 "
             "equivalent of species-breakdown, from cached features",
    )
    classifier_species_breakdown.add_argument(
        "--run", required=True, help="run name (checkpoints/<run>/)",
    )
    classifier_species_breakdown.add_argument(
        "--split", choices=["val", "test"], default="val",
    )
    classifier_species_breakdown.set_defaults(
        func=_run_classifier_species_breakdown,
    )

    check = sub.add_parser(
        "check",
        help="Identify a plant photo and report whether it's toxic to "
             "cats",
    )
    check.add_argument("--image", required=True, help="path to a photo")
    check.add_argument(
        "--run", default=DEFAULT_RUN,
        help="species-ID run to identify with (checkpoints/<run>/)",
    )
    check.add_argument(
        "--head-run", default=None,
        help="use --run as a frozen backbone and bolt this separately "
             "trained classifier head on top (the decoupled-training "
             f"setup; pair it with --run {DEFAULT_BACKBONE_RUN})",
    )
    check.add_argument(
        "--epoch", type=_positive_int, default=None,
        help="evaluate a specific epoch_<N>.pt instead of latest.pt",
    )
    check.add_argument(
        "--ema", action="store_true",
        help="use the checkpoint's EMA shadow weights",
    )
    check.add_argument(
        "--top-k", type=_positive_int, default=None,
        help="candidates to show (default: 5 for text, all for --json)",
    )
    check.add_argument(
        "--json", action="store_true",
        help="machine-readable output instead of the text report",
    )
    check.set_defaults(func=_run_check)
    return parser


def main(argv=None):
    """Entry point; returns a process exit code."""
    from plant_toxicity_checker.pipeline import ImageLoadError
    from plant_toxicity_checker.train import ConfigError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ConfigError, ImageLoadError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
