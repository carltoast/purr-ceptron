"""Regenerate the figures embedded in README.md.

    python scripts/make_readme_figures.py

Writes PNGs into ``docs/images/``. The results figures read only the
JSON in ``outputs/``, so they work on any clone that has pulled a run.
The two demo cards additionally run the real classifier, so they need a
checkpoint; without one they are skipped with a message rather than
failing the whole script.

Re-run this whenever the numbers quoted in the README change.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from plant_toxicity_checker.config import (  # noqa: E402
    CANDIDATE_FLOOR,
    DEFAULT_BACKBONE_RUN,
    DEFAULT_RUN,
    MIN_CONFIDENCE,
    TOXICITY_TABLE_PATH,
)

OUTPUTS = REPO_ROOT / "outputs"
IMAGES = REPO_ROOT / "docs" / "images"
EXAMPLES = IMAGES / "examples"

BENCHMARK_LABEL = "ResNet-18 (benchmark)"
SHIPPED_LABEL = "ConvNeXt V2-Tiny (shipped)"
BENCHMARK_COLOR, SHIPPED_COLOR = "#8c8c8c", "#2d5f8a"
TIER_COLOR = {"toxic": "#b3202c", "caution": "#c77700",
              "likely_safe": "#2e7d32"}
TIER_HEADLINE = {"toxic": "TOXIC TO CATS",
                 "caution": "CAUTION",
                 "likely_safe": "LIKELY SAFE"}
# What the toxicity table says about a candidate, as a bar colour. Only a
# verified exact non-toxic entry is green, matching the verdict logic.
CATEGORY_COLOR = {
    "exact - toxic": "#b3202c",
    "genus - toxic (direct)": "#b3202c",
    "genus - toxic (inferred)": "#e08a7d",
    "exact - non-toxic": "#2e7d32",
    "genus - non-toxic (direct)": "#9e9e9e",
    "genus - non-toxic (inferred)": "#9e9e9e",
    "exact - unknown": "#9e9e9e",
    "genus - unknown": "#9e9e9e",
    "no match": "#d0d0d0",
}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(figure, name):
    path = IMAGES / name
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def load(path):
    return json.loads((OUTPUTS / path).read_text(encoding="utf-8"))


def pipeline_diagram():
    """The two-stage pipeline. Same figure as notebook 00's, which is
    the simplified view: accurate about which verdict comes out, drawn
    as a chain of tests. The exact procedure is in notebook 03."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stage1, stage2 = "#2d5f8a", "#6b4c9a"
    architecture = json.loads(
        (REPO_ROOT / "configs"
         / "plantnet_convnextv2_tiny_toxicity_v2.json").read_text(
            encoding="utf-8"))["model"]
    architecture = {"convnextv2_tiny": "ConvNeXt V2-Tiny"}.get(
        architecture, architecture)

    def box(xy, w, h, text, edge, fill="#ffffff", fontsize=10.5,
            weight="normal", color=None):
        x, y = xy
        axis.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4",
            linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2))
        axis.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                  color=color or "#1a1a1a", weight=weight, zorder=3)

    def arrow(start, end, color="#555555", label=None, dx=0, dy=1.8):
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.5,
            color=color, zorder=1, shrinkA=0, shrinkB=0))
        if label:
            axis.text((start[0] + end[0]) / 2 + dx,
                      (start[1] + end[1]) / 2 + dy, label, ha="center",
                      va="center", fontsize=9.5, color=color,
                      weight="bold", zorder=3)

    figure, axis = plt.subplots(figsize=(11, 8.6))
    axis.set_xlim(0, 100)
    axis.set_ylim(-11, 100)
    axis.axis("off")

    axis.add_patch(FancyBboxPatch(
        (3, 75), 94, 21, boxstyle="round,pad=0.6", linewidth=1.2,
        edgecolor=stage1, facecolor="#eef4fa", linestyle="--", zorder=0))
    axis.text(5, 93.3,
              "STAGE 1   -   species identification (deep neural network)",
              fontsize=11.5, weight="bold", color=stage1, va="center")
    box((16, 83), 22, 11, "photograph", stage1, fontsize=11)
    box((50, 83), 30, 11,
        f"{architecture} classifier\n1,000 Pl@ntNet-300K v2 species",
        stage1, fontsize=11)
    box((84, 83), 24, 11, "ranked species\n+ confidences", stage1,
        fontsize=11)
    arrow((27, 83), (35, 83))
    arrow((65, 83), (72, 83))
    arrow((84, 77.5), (84, 68.5), color=stage2)

    axis.add_patch(FancyBboxPatch(
        (3, -9), 94, 75, boxstyle="round,pad=0.6", linewidth=1.2,
        edgecolor=stage2, facecolor="#f6f2fb", linestyle="--", zorder=0))
    axis.text(5, 63.5,
              "STAGE 2   -   safety verdict (curated table + fixed rules)",
              fontsize=11.5, weight="bold", color=stage2, va="center")
    box((34, 56), 58, 10,
        "match each candidate species against the ASPCA\n"
        "cat-toxic / non-toxic plant lists (hand-curated)", stage2,
        fontsize=11)
    arrow((34, 50.5), (34, 47.5), color=stage2)

    decisions = [
        (42, "is the top match toxic?", "TOXIC", "toxic"),
        (27, f"is confidence at least {MIN_CONFIDENCE:.0%}?", "CAUTION",
         "caution"),
        (12, f"does every candidate above {CANDIDATE_FLOOR:.0%}\n"
             "have a VERIFIED non-toxic entry?", "CAUTION", "caution"),
    ]
    for index, (y, question, outcome, tier) in enumerate(decisions):
        box((34, y), 44, 9.5, question, "#555555", fontsize=11)
        exits_on_yes = index == 0   # only the toxic test leaves on "yes"
        arrow((56, y), (72, y), color=TIER_COLOR[tier],
              label="yes" if exits_on_yes else "no", dy=2.2)
        box((84, y), 21, 8, outcome, TIER_COLOR[tier],
            fill=TIER_COLOR[tier], fontsize=11.5, weight="bold",
            color="white")
        if index < len(decisions) - 1:
            arrow((34, y - 5.0), (34, decisions[index + 1][0] + 5.0),
                  label="no" if exits_on_yes else "yes", dx=-3.5, dy=0)

    arrow((34, 7.0), (34, -3), label="yes", dx=-3.5, dy=3.5,
          color=TIER_COLOR["likely_safe"])
    arrow((34, -3), (72, -3), color=TIER_COLOR["likely_safe"])
    box((84, -3), 21, 8, "LIKELY SAFE", TIER_COLOR["likely_safe"],
        fill=TIER_COLOR["likely_safe"], fontsize=11.5, weight="bold",
        color="white")
    save(figure, "pipeline.png")


def results_metrics():
    """Plain top-1 beside macro-average top-1, for both models."""
    evals = {
        BENCHMARK_LABEL: load(f"{DEFAULT_BACKBONE_RUN}/eval_test.json"),
        SHIPPED_LABEL: load(f"{DEFAULT_RUN}/eval_test_epoch30_ema.json"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, (title, key) in zip(
            axes, (("plain top-1", "top_1"),
                   ("macro-average top-1", "macro_top_1"))):
        names = list(evals)
        values = [evals[name][key] for name in names]
        bars = axis.bar(range(len(names)), values, width=0.55,
                        color=[BENCHMARK_COLOR, SHIPPED_COLOR])
        axis.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3,
                       fontsize=11, fontweight="bold")
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(["ResNet-18\nbenchmark",
                              "ConvNeXt V2-Tiny\nshipped"], fontsize=9.5)
        axis.set_ylim(0, 1.0)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    delta = (evals[SHIPPED_LABEL]["macro_top_1"]
             - evals[BENCHMARK_LABEL]["macro_top_1"])
    plain = evals[SHIPPED_LABEL]["top_1"] - evals[BENCHMARK_LABEL]["top_1"]
    figure.suptitle(
        f"Same two models, two ways of averaging: {plain:+.1%} one way, "
        f"{delta:+.1%} the other", fontsize=11, y=1.02)
    save(figure, "results_metrics.png")


def results_buckets():
    """Top-1 against how many training photos a species had."""
    breakdowns = {
        BENCHMARK_LABEL: load(
            f"{DEFAULT_BACKBONE_RUN}/species_breakdown_test.json"),
        SHIPPED_LABEL: load(
            f"{DEFAULT_RUN}/species_breakdown_test_epoch30.json"),
    }
    order = ["<20", "20-49", "50-199", "200+"]
    labels = ["under 20", "20-49", "50-199", "200 or more"]
    figure, axis = plt.subplots(figsize=(9, 4))
    width, positions = 0.38, np.arange(len(order))
    for offset, (name, color) in zip(
            (-width / 2, width / 2),
            ((BENCHMARK_LABEL, BENCHMARK_COLOR),
             (SHIPPED_LABEL, SHIPPED_COLOR))):
        values = [breakdowns[name]["buckets"][b]["top1_accuracy"]
                  for b in order]
        bars = axis.bar(positions + offset, values, width, color=color,
                        label=name)
        axis.bar_label(bars, labels=[f"{v:.0%}" for v in values], padding=2,
                       fontsize=9.5, fontweight="bold")
    counts = [breakdowns[SHIPPED_LABEL]["buckets"][b]["n_species"]
              for b in order]
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{label}\n({n} species)"
                          for label, n in zip(labels, counts)], fontsize=9.5)
    axis.set_xlabel("training photos for that species", fontsize=10)
    axis.set_ylim(0, 1.05)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    axis.set_title("accuracy is not spread evenly across species",
                   fontsize=12, fontweight="bold")
    axis.legend(fontsize=9.5, frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    save(figure, "results_buckets.png")


def demo_card(filename, output_name):
    """One example card: the photo, the top-5 candidates coloured by what
    the toxicity table says about each, and the verdict banner."""
    from plant_toxicity_checker import cli, plantnet_labels
    from plant_toxicity_checker.pipeline import check_image, load_image
    from plant_toxicity_checker.plantnet_classifier import PlantNetClassifier
    from plant_toxicity_checker.toxicity import load_table

    table = load_table(TOXICITY_TABLE_PATH)
    classifier = PlantNetClassifier(DEFAULT_RUN)
    short = {row["full_species"]: row["species"]
             for row in plantnet_labels.load_species_rows()}

    # Show only candidates the model gives real weight to. The shipped
    # model is confident enough that ranks 3-5 are often under 0.01%,
    # and a row reading "0.0%" is noise rather than information.
    VISIBLE_FLOOR = 0.001

    path = EXAMPLES / filename
    result = cli.verdict_to_dict(
        check_image(path, classifier, table), table, top_k=5)
    result["predictions"] = [p for p in result["predictions"]
                             if p["confidence"] >= VISIBLE_FLOOR]
    tier = result["status"]

    predictions = result["predictions"]

    # Read top to bottom: photo, then what the classifier thought, then
    # the verdict, with an arrow at each hand-off.
    figure = plt.figure(figsize=(5.2, 9.2))
    grid = figure.add_gridspec(
        5, 1, height_ratios=[3.3, 0.95, 2.4, 0.95, 0.6], hspace=0.3)

    photo = figure.add_subplot(grid[0])
    image = load_image(path)
    side = min(image.size)
    image = image.crop(((image.width - side) // 2, (image.height - side) // 2,
                        (image.width - side) // 2 + side,
                        (image.height - side) // 2 + side))
    photo.imshow(image)
    photo.axis("off")

    def down_arrow(row):
        axis = figure.add_subplot(grid[row])
        axis.axis("off")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.annotate("", xy=(0.5, 0.1), xytext=(0.5, 0.9),
                      arrowprops={"arrowstyle": "-|>", "color": "#666666",
                                  "linewidth": 2.6, "mutation_scale": 26})

    down_arrow(1)

    bars = figure.add_subplot(grid[2])
    # Names sit above their bars rather than in a left margin: in a
    # narrow card a margin wide enough for "Pelargonium x hybridum"
    # would leave almost no room for the bars themselves.
    slots = max(len(predictions), 3)
    positions = [slots - 1 - index for index in range(len(predictions))]
    colors = [
        "#9e9e9e" if (p["toxicity_category"] == "exact - non-toxic"
                      and not p["verified"])
        else CATEGORY_COLOR[p["toxicity_category"]]
        for p in predictions
    ]
    bars.barh(positions, [p["confidence"] for p in predictions],
              color=colors, height=0.42)
    bars.set_ylim(-0.55, slots - 0.25)
    bars.set_yticks([])
    bars.set_xlim(0, 1)
    bars.set_xticks([CANDIDATE_FLOOR, 0.5, MIN_CONFIDENCE, 1.0])
    bars.set_xticklabels([f"{CANDIDATE_FLOOR:.0%}", "50%",
                          f"{MIN_CONFIDENCE:.0%}", "100%"], fontsize=8)
    bars.spines["left"].set_visible(False)
    for value, style in ((CANDIDATE_FLOOR, ":"), (MIN_CONFIDENCE, "--")):
        bars.axvline(value, color="#999999", linewidth=0.9, linestyle=style,
                     zorder=0)
    for y, prediction in zip(positions, predictions):
        bars.text(0.008, y + 0.3, short.get(prediction["species"],
                                            prediction["species"]),
                  va="bottom", ha="left", fontsize=8.5, style="italic",
                  color="#1a1a1a")
        inside = prediction["confidence"] > 0.75
        bars.text(prediction["confidence"] - 0.015 if inside
                  else prediction["confidence"] + 0.015, y,
                  f"{prediction['confidence']:.1%}", va="center",
                  ha="right" if inside else "left", fontsize=8.5,
                  color="white" if inside else "#333333",
                  fontweight="bold" if inside else "normal")

    down_arrow(3)

    banner = figure.add_subplot(grid[4])
    banner.axis("off")
    banner.add_patch(plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[tier]))
    banner.text(0.5, 0.5, TIER_HEADLINE[tier], ha="center", va="center",
                color="white", fontsize=16, fontweight="bold")
    save(figure, output_name)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    pipeline_diagram()
    results_metrics()
    results_buckets()
    try:
        demo_card("pelargonium_zonale.jpg", "demo_toxic.png")
        demo_card("calendula_confident.jpg", "demo_safe.png")
    except (OSError, ImportError) as error:
        print(f"skipped the demo cards: {error}")
        print("they need a checkpoint; see scripts/remote/pull.sh")


if __name__ == "__main__":
    main()
