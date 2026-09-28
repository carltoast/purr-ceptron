"""Command-line interface: ``python -m bouquet_checker <command>``."""

import argparse
import sys

from bouquet_checker.config import ASPCA_DETAILS_JSON_PATH
from bouquet_checker.verdict import SCOPE_NOTE, Status

_HEADLINES = {
    Status.TOXIC: "TOXIC TO CATS - keep away from cats",
    Status.CAUTION: "CAUTION - treat as toxic, do not rely on this",
    Status.LIKELY_SAFE: "LIKELY SAFE for the main flower only "
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
    from bouquet_checker import aspca_detail

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


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser():
    """Build the argparse parser.

    This is intentionally thin right now: the previous Oxford-102
    ``check``/``demo``/``download``/``evaluate`` commands were removed when
    the project moved to PlantNet-300K + ASPCA as its species/toxicity
    source (see README/CLAUDE.md); their replacements land with the
    PlantNet training pipeline. ``verdict_to_dict``/``format_verdict``
    above are kept - they are already species-space-agnostic and will back
    the new commands directly.
    """
    parser = argparse.ArgumentParser(prog="bouquet-checker")
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
    return parser


def main(argv=None):
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except OSError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
