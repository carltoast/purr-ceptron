"""Train a PlantNet-300K species classifier (Roadmap Phase 1 baseline).

Replicates the official PlantNet-300K paper's ResNet-18 baseline setup:
ImageNet-pretrained backbone, SGD with Nesterov momentum, a step learning
-rate schedule, and plain cross-entropy - no class-imbalance handling
(that's Phase 2) and no toxicity-aware loss weighting (Phase 3). The
model only ever predicts a species; nothing here touches toxicity.

``torch``/``torchvision`` are imported inside functions, not at module
level, so importing this module (e.g. from ``cli.py`` for argument
parsing) never requires torch to be installed.
"""

import json
import random
import time
from pathlib import Path

from plant_toxicity_checker.config import REPO_ROOT

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
OUTPUTS_DIR = REPO_ROOT / "outputs"

REQUIRED_CONFIG_KEYS = ("lr", "weight_decay", "batch_size", "epochs")
# Only SGD (the default) needs these; AdamW configs can omit them.
_SGD_REQUIRED_CONFIG_KEYS = ("momentum", "nesterov")
# Only the "multistep" scheduler (the default) needs these; "cosine"
# derives its own shape from the config's own "epochs".
_MULTISTEP_REQUIRED_CONFIG_KEYS = ("lr_decay_epochs", "lr_decay_factor")

# DataLoader performance knobs - not paper hyperparameters (those are all
# in configs/*.json), just how fast data gets to the GPU, so they live
# here as fixed constants rather than in the training config.
# 8 of the remote PC's 12 cores: leaves headroom for the main process/OS
# rather than claiming every core.
NUM_WORKERS = 8

# Buckets for rolling per-species accuracy up by training support, to
# check whether PlantNet-300K's long-tailed species distribution is
# dragging accuracy down specifically for low-data species. (name, lo,
# hi) - hi=None means unbounded.
TRAIN_COUNT_BUCKETS = (
    ("<20", 0, 20),
    ("20-49", 20, 50),
    ("50-199", 50, 200),
    ("200+", 200, None),
)


class ConfigError(ValueError):
    """A training config is missing a required key or has a bad value."""


def load_config(path):
    """Load and lightly validate a training config JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in data]
    if data.get("optimizer", "sgd") == "sgd":
        missing += [
            k for k in _SGD_REQUIRED_CONFIG_KEYS if k not in data
        ]
    if data.get("scheduler", "multistep") == "multistep":
        missing += [
            k for k in _MULTISTEP_REQUIRED_CONFIG_KEYS if k not in data
        ]
    if missing:
        raise ConfigError(f"config missing keys: {missing}")
    return data


def toxicity_priority_species_ids(config, species_metadata_path=None,
                                   toxicity_table_path=None):
    """``species_id``\\s to prioritize when subsetting species (see
    ``plantnet_dataset.filter_species_subset``), if the config asks for
    it via ``"prioritize_toxicity_species": true`` - the PlantNet
    species with an EXACT match in the real toxicity table, so a small
    ``max_species`` dev subset doesn't accidentally end up with none of
    them. Returns ``None`` if the config doesn't ask for this (the
    common case - the paper-replica baseline config never sets it).

    Matches against ``data/toxicity.json`` (``species_match
    .match_against_toxicity_table``) - the one place this project
    decides how PlantNet species relate to the toxicity table, not the
    raw ASPCA scrape directly. Name matching only, never reads
    ``toxic_to_cats`` here, so this stays outside the toxicity-verdict
    safety path entirely.
    """
    if not config.get("prioritize_toxicity_species"):
        return None
    from plant_toxicity_checker import species_match
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.plantnet_labels import load_species_rows
    from plant_toxicity_checker.toxicity import load_table

    table = load_table(toxicity_table_path or TOXICITY_TABLE_PATH)
    matches = species_match.match_against_toxicity_table(
        load_species_rows(species_metadata_path), table
    )
    return {
        species_id for species_id, result in matches.items()
        if result.status == "exact"
    }


def _bucket_for(train_count, buckets=TRAIN_COUNT_BUCKETS):
    """Which bucket name ``train_count`` falls into."""
    for name, lo, hi in buckets:
        if train_count >= lo and (hi is None or train_count < hi):
            return name
    raise ValueError(f"no bucket for train_count={train_count}")


def aggregate_by_train_count_bucket(
    per_class_stats, train_counts, buckets=TRAIN_COUNT_BUCKETS,
):
    """Roll ``metrics.per_class_accuracy``'s per-class stats up into
    buckets by each species' training-image count.

    ``per_class_stats`` and ``train_counts`` are both keyed by
    ``species_id`` (not class index - callers join on that first). A
    species absent from ``train_counts`` (zero training images) counts
    as bucket ``"<20"``. Returns ``{bucket_name: {"n_species", "n",
    "top1_correct", "top5_correct", "top1_accuracy", "top5_accuracy"}}``
    for every bucket, even ones with zero species (accuracy ``None``
    then, not a divide-by-zero).
    """
    result = {
        name: {"n_species": 0, "n": 0, "top1_correct": 0, "top5_correct": 0}
        for name, _, _ in buckets
    }
    for species_id, stats in per_class_stats.items():
        bucket = _bucket_for(train_counts.get(species_id, 0), buckets)
        agg = result[bucket]
        agg["n_species"] += 1
        agg["n"] += stats["n"]
        agg["top1_correct"] += stats["top1_correct"]
        agg["top5_correct"] += stats["top5_correct"]
    for agg in result.values():
        agg["top1_accuracy"] = (
            agg["top1_correct"] / agg["n"] if agg["n"] else None
        )
        agg["top5_accuracy"] = (
            agg["top5_correct"] / agg["n"] if agg["n"] else None
        )
    return result


def build_model(num_classes, model="resnet18", pretrained=True):
    """The backbone named by ``model`` (the config's own ``"model"``
    key - default ``"resnet18"``, Phase 1's paper-replica baseline),
    ImageNet-pretrained, with its final classifier replaced for
    ``num_classes``.

    ``pretrained=False`` skips fetching those ImageNet weights, which
    is what every load-a-checkpoint path wants: the download is
    discarded by the ``load_state_dict`` that follows, and skipping it
    saves ~110MB and a network round trip on the first run.

    ``"convnextv2_tiny"`` loads ``convnextv2_tiny.fcmae_ft_in22k_in1k``
    via ``timm``: ConvNeXt V2's self-supervised (masked-autoencoder)
    pretraining on ImageNet-1k, with an ImageNet-22k-assisted
    fine-tune - both closed, pre-2021 corpora (see ``CLAUDE.md``'s
    backbone-provenance rule), scoring ~1.4 points higher on
    ImageNet-1k than the plain supervised torchvision weights at the
    same ~28.6M-parameter/4.5-GFLOP budget. ``timm.create_model``'s
    own ``num_classes`` argument replaces the classifier head in one
    step - there's no single ``.fc`` attribute to swap by hand the way
    ResNet-18 has.
    """
    if model == "resnet18":
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if model == "convnextv2_tiny":
        import timm

        return timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    raise ValueError(f"unknown model: {model!r}")


def build_optimizer(model, config):
    """SGD with Nesterov momentum (the config's ``"optimizer"`` key
    left unset or ``"sgd"`` - Phase 1's original recipe) or AdamW
    (``"adamw"`` - the literature's standard choice for ConvNeXt-
    family backbones, which SGD tends to undertrain)."""
    from torch import optim

    if config.get("optimizer", "sgd") == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    return optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
        weight_decay=config["weight_decay"],
    )


def build_scheduler(optimizer, config):
    """Step decay at ``lr_decay_epochs`` by ``lr_decay_factor`` (the
    config's ``"scheduler"`` key left unset or ``"multistep"`` -
    Phase 1's original recipe), or cosine decay across the full
    ``epochs`` (``"cosine"`` - ConvNeXt's own published recipe, Liu et
    al. 2022). Cosine needs no milestones: it derives its shape from
    ``config["epochs"]`` alone.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

    if config.get("scheduler", "multistep") == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["epochs"])
    return MultiStepLR(
        optimizer,
        milestones=config["lr_decay_epochs"],
        gamma=config["lr_decay_factor"],
    )


def make_loader(dataset, batch_size, shuffle, sampler=None):
    """A ``DataLoader`` with this project's standard performance knobs.

    ``persistent_workers=True`` keeps worker processes alive between
    epochs instead of respawning them each time (previously visible as
    new worker PIDs every epoch); ``pin_memory=True`` speeds up the
    CPU->GPU copy. Neither changes what's computed, only how fast data
    gets to the GPU - safe to change freely without affecting results.

    ``sampler``, if given (e.g. ``make_sqrt_freq_sampler``'s result),
    picks which images each batch draws instead of plain shuffling -
    ``shuffle`` is ignored in that case (``DataLoader`` itself rejects
    passing both).
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )


def make_sqrt_freq_sampler(dataset):
    """``WeightedRandomSampler`` with per-sample weight ``1 /
    sqrt(species_count)``, where ``species_count`` is that sample's
    species' image count within ``dataset.rows`` itself (so it still
    makes sense under ``max_species``/``max_images_per_species``
    subsetting).

    A gentler alternative to full inverse-frequency balancing: with
    PlantNet-300K's ~3600:1 imbalance ratio (2 images for the rarest
    species vs. 7202 for the commonest), full inverse weighting would
    oversample a 2-image species ~3600x more than the commonest one -
    mostly repeating the same couple of photos' augmented crops rather
    than teaching the backbone anything new. The square root compresses
    that ratio to ~60:1, giving the long tail meaningfully more
    gradient attention without letting a handful of near-duplicate
    images dominate batches - see ``CLAUDE.md``'s Phase 2/3 notes on
    why decoupled training didn't try this (it can't: this changes
    what the frozen backbone itself would see, which only a from
    -scratch retrain can test).
    """
    import math

    from torch.utils.data import WeightedRandomSampler

    from plant_toxicity_checker.plantnet_dataset import images_per_species

    counts = images_per_species(dataset.rows)
    weights = [
        1.0 / math.sqrt(counts[int(row["species_id"])])
        for row in dataset.rows
    ]
    return WeightedRandomSampler(weights, num_samples=len(dataset.rows))


def save_checkpoint(
    run_name, epoch, model, optimizer, scheduler, config, ema=None,
):
    """Write ``checkpoints/<run_name>/latest.pt`` and a per-epoch
    snapshot. ``ema`` (a ``timm.utils.ModelEmaV2``), if given, has its
    shadow weights saved alongside the live model's - so a
    ``--resume``d run picks EMA back up too, not just the raw weights.
    """
    import torch

    run_dir = CHECKPOINTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
        "config": config,
    }
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    torch.save(state, run_dir / f"epoch_{epoch}.pt")
    torch.save(state, run_dir / "latest.pt")


def load_checkpoint(run_name, model, optimizer, scheduler, ema=None):
    """Load ``checkpoints/<run_name>/latest.pt`` in place; returns the
    epoch it finished. ``ema``, if given, is restored from the
    checkpoint's saved shadow weights when present (older checkpoints
    saved before EMA was added simply have none to restore)."""
    import torch

    state = torch.load(
        CHECKPOINTS_DIR / run_name / "latest.pt", map_location="cpu"
    )
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if ema is not None and "ema_state_dict" in state:
        ema.load_state_dict(state["ema_state_dict"])
    torch.set_rng_state(state["rng_state"]["torch"])
    random.setstate(state["rng_state"]["python"])
    return state["epoch"]


def _predict_probas(model, loader, device):
    """Softmax probabilities and true labels for every item in
    ``loader``."""
    import numpy as np
    import torch

    model.eval()
    all_probas, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probas = torch.softmax(logits, dim=1)
            all_probas.append(probas.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probas), np.concatenate(all_labels)


def train_one_epoch(
    model, loader, optimizer, device, toxicity_cost_weight=0.0,
    species_sides=None, warmup_steps=0, base_lr=None,
    grad_accum_steps=1, ema=None, log_every=100,
):
    """One training epoch; returns the mean training loss.

    ``toxicity_cost_weight`` (0 = off, the default - Phase 1's plain
    cross-entropy) adds Phase 3's toxicity-aware term on top of CE,
    reusing ``decoupled.toxicity_cost_term`` exactly as the decoupled
    classifier-head retraining does - the same training-time signal
    (``species_sides``, from ``species_match.species_sides``), just
    applied jointly with the backbone's own gradients instead of on
    frozen cached features. ``species_sides`` is required whenever
    ``toxicity_cost_weight`` is nonzero.

    ``grad_accum_steps`` (1 = off, the default) accumulates gradients
    over that many micro-batches before each real optimizer step -
    approximates a larger effective batch (``batch_size *
    grad_accum_steps``) without the extra activation memory a truly
    larger batch would need, since only one micro-batch's activations
    are ever resident at once. Reduces gradient noise roughly by
    ``1/sqrt(grad_accum_steps)`` - see the conversation history for why
    this was added (validation accuracy swinging far more than the
    baseline ever did, batch 16 vs. the baseline's 32). The final,
    possibly-short group at the end of the loader still triggers a real
    step, so no batch's gradient is ever silently dropped.

    ``warmup_steps`` (0 = off, the default), together with ``base_lr``,
    linearly ramps the optimizer's LR from 0 up to ``base_lr`` over the
    first ``warmup_steps`` REAL OPTIMIZER STEPS of THIS call (not raw
    micro-batches - with accumulation, several micro-batches pass
    between each one), then holds it at ``base_lr`` for the rest.
    Callers should only pass a nonzero value for the run's true first
    training epoch (see ``run_training`` - a ``--resume``d run never
    re-warms-up). Guards against AdamW taking a large step on a strong
    pretrained backbone's weights before its per-parameter moment
    estimates have seen enough steps to be reliable - an un-warmed-up
    high LR collapsed this project's ConvNeXt V2 sanity check to a
    near-uniform prediction (loss stuck at ln(1000)); see the
    conversation history for the actual numbers.

    ``ema``, if given (a ``timm.utils.ModelEmaV2`` wrapping ``model``),
    is updated once per real optimizer step - same cadence as the
    optimizer itself, not per micro-batch.

    Prints running progress every ``log_every`` batches - an epoch over
    the full training set can take a long time (largely disk-I/O bound
    on the first, cold-cache pass over the dataset), and a run with zero
    in-epoch output is hard to tell apart from a hang when checking on it
    remotely.
    """
    import torch
    from torch import nn

    if toxicity_cost_weight:
        from plant_toxicity_checker.decoupled import toxicity_cost_term

    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_seen = 0
    n_batches = len(loader)
    optimizer_step = 0
    t0 = time.time()
    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        if toxicity_cost_weight:
            probas = torch.softmax(logits, dim=1)
            loss = loss + toxicity_cost_weight * toxicity_cost_term(
                probas, labels, species_sides,
            )
        (loss / grad_accum_steps).backward()
        total_loss += loss.item() * len(labels)
        n_seen += len(labels)

        is_last_batch = batch_idx == n_batches
        if batch_idx % grad_accum_steps == 0 or is_last_batch:
            if warmup_steps and optimizer_step < warmup_steps:
                warmup_lr = base_lr * (optimizer_step + 1) / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr
            elif warmup_steps and optimizer_step == warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = base_lr

            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            optimizer_step += 1

        if batch_idx % log_every == 0 or is_last_batch:
            elapsed = time.time() - t0
            print(
                f"  batch {batch_idx}/{n_batches}  "
                f"running_loss={total_loss / n_seen:.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
    return total_loss / max(n_seen, 1)


def run_training(
    config, run_name, epochs_override=None, resume=False,
):
    """The full training loop: builds everything, trains, checkpoints
    every epoch, writes final metrics.

    ``epochs_override`` lets a short "how long is one epoch really"
    sanity run happen without a second config file - see the
    ``train-plantnet --epochs`` CLI flag.
    """
    import torch

    from plant_toxicity_checker import metrics
    from plant_toxicity_checker.plantnet_dataset import PlantNetDataset
    from plant_toxicity_checker.plantnet_labels import SpeciesLabels

    labels = SpeciesLabels.from_metadata()
    max_species = config.get("max_species")
    max_images_per_species = config.get("max_images_per_species")
    priority_species_ids = toxicity_priority_species_ids(config)
    train_ds = PlantNetDataset(
        "train", labels,
        max_species=max_species,
        max_images_per_species=max_images_per_species,
        priority_species_ids=priority_species_ids,
    )
    val_ds = PlantNetDataset(
        "val", labels,
        max_species=max_species,
        max_images_per_species=max_images_per_species,
        priority_species_ids=priority_species_ids,
    )
    sampler_kind = config.get("sampler")
    train_sampler = None
    if sampler_kind == "sqrt_freq":
        train_sampler = make_sqrt_freq_sampler(train_ds)
    elif sampler_kind is not None:
        raise ConfigError(f"unknown sampler: {sampler_kind!r}")

    toxicity_cost_weight = float(config.get("toxicity_cost_weight", 0.0))
    species_sides = None
    if toxicity_cost_weight:
        from plant_toxicity_checker import species_match
        from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
        from plant_toxicity_checker.plantnet_labels import load_species_rows
        from plant_toxicity_checker.toxicity import load_table

        table = load_table(TOXICITY_TABLE_PATH)
        species_sides = species_match.species_sides(
            load_species_rows(), table, labels,
        )

    warmup_steps = int(config.get("warmup_steps", 0))
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    ema_decay = config.get("ema_decay")

    print(
        f"train={len(train_ds)} images  val={len(val_ds)} images"
        + (f"  max_species={max_species}" if max_species else "")
        + (
            f"  max_images_per_species={max_images_per_species}"
            if max_images_per_species else ""
        )
        + (
            f"  toxicity_priority_species={len(priority_species_ids)}"
            if priority_species_ids else ""
        )
        + (f"  sampler={sampler_kind}" if sampler_kind else "")
        + (
            f"  toxicity_cost_weight={toxicity_cost_weight}"
            if toxicity_cost_weight else ""
        )
        + (f"  warmup_steps={warmup_steps}" if warmup_steps else "")
        + (
            f"  grad_accum_steps={grad_accum_steps}"
            if grad_accum_steps != 1 else ""
        )
        + (f"  ema_decay={ema_decay}" if ema_decay else ""),
        flush=True,
    )

    batch_size = config["batch_size"]
    train_loader = make_loader(
        train_ds, batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get("model", "resnet18")
    model = build_model(len(labels), model_name).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    ema = None
    if ema_decay:
        from timm.utils import ModelEmaV2

        ema = ModelEmaV2(model, decay=ema_decay, device=device)

    start_epoch = 1
    if resume:
        finished_epoch = load_checkpoint(
            run_name, model, optimizer, scheduler, ema=ema,
        )
        start_epoch = finished_epoch + 1

    total_epochs = epochs_override or config["epochs"]
    if start_epoch > total_epochs:
        print(
            f"run '{run_name}' already has {start_epoch - 1} epoch(s); "
            f"nothing to do for epochs={total_epochs}",
            flush=True,
        )
        return

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            toxicity_cost_weight=toxicity_cost_weight,
            species_sides=species_sides,
            warmup_steps=warmup_steps if epoch == 1 else 0,
            base_lr=config["lr"],
            grad_accum_steps=grad_accum_steps, ema=ema,
        )
        scheduler.step()

        val_probas, val_labels = _predict_probas(model, val_loader, device)
        val_top1 = metrics.top_k_accuracy(val_probas, val_labels, 1)
        val_top5 = metrics.top_k_accuracy(val_probas, val_labels, 5)
        elapsed = time.time() - t0

        # Reuses this epoch's own val predictions - no extra GPU work -
        # so every run trained from here on gets a per-epoch bucket
        # breakdown "for free", rather than needing a separate manual
        # pass afterwards (as the first convnextv2 run did).
        _write_species_breakdown(
            run_name, "val", epoch,
            _species_breakdown_result(
                run_name, "val", epoch, val_probas, val_labels, labels,
            ),
        )

        line = (
            f"epoch {epoch}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_top1={val_top1:.4f}  val_top5={val_top5:.4f}  "
        )
        if ema is not None:
            ema_probas, ema_labels = _predict_probas(
                ema.module, val_loader, device,
            )
            ema_val_top1 = metrics.top_k_accuracy(ema_probas, ema_labels, 1)
            ema_val_top5 = metrics.top_k_accuracy(ema_probas, ema_labels, 5)
            line += (
                f"ema_val_top1={ema_val_top1:.4f}  "
                f"ema_val_top5={ema_val_top5:.4f}  "
            )
        line += f"time={elapsed:.1f}s"
        print(line, flush=True)
        save_checkpoint(
            run_name, epoch, model, optimizer, scheduler, config, ema=ema,
        )

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    final_metrics = {
        "epochs_run": total_epochs,
        "val_top1": val_top1,
        "val_top5": val_top5,
    }
    if ema is not None:
        final_metrics["ema_val_top1"] = ema_val_top1
        final_metrics["ema_val_top5"] = ema_val_top5
    (out_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8",
    )


def load_trained_model(run_name, num_classes, epoch=None, use_ema=False):
    """``(model, config, epoch)`` for a finished ``train-plantnet`` run,
    ready for inference - the one place that turns a saved checkpoint
    back into a usable model, shared by batch evaluation
    (``run_evaluation``) and single-image inference
    (``plantnet_classifier``) so the two can never drift apart.

    ``epoch`` picks an ``epoch_<N>.pt`` snapshot instead of
    ``latest.pt``; ``use_ema`` loads the EMA shadow weights saved
    alongside the live ones. A checkpoint trimmed for shipping may hold
    only the weights that were chosen, in which case its EMA weights
    are already in ``model_state_dict`` and ``use_ema`` is not needed.
    """
    import torch

    checkpoint_name = f"epoch_{epoch}.pt" if epoch is not None else "latest.pt"
    path = CHECKPOINTS_DIR / run_name / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path} - train it with `train-plantnet "
            f"--config ...` or pull it from the remote PC with "
            f"scripts/remote/pull.sh checkpoints"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes, config.get("model", "resnet18"),
                        pretrained=False)
    model = model.to(device)
    if use_ema:
        if "ema_state_dict" not in state:
            raise ValueError(
                f"{run_name} epoch {state['epoch']} has no EMA weights - it "
                f"was trained without an 'ema_decay' config key"
            )
        # ModelEmaV2 stores the shadow model under a 'module.' prefix.
        model.load_state_dict({
            key[len("module."):]: value
            for key, value in state["ema_state_dict"].items()
            if key.startswith("module.")
        })
    else:
        model.load_state_dict(state["model_state_dict"])
    return model, config, state["epoch"]


def run_evaluation(run_name, split, epoch=None, use_ema=False):
    """Load a run's checkpoint and report top-1/3/5/10, average-k, and
    the toxic/non-toxic confusion rate on ``split``.

    ``epoch`` picks a specific ``epoch_<N>.pt`` snapshot instead of
    ``latest.pt``; ``use_ema`` evaluates the EMA shadow weights saved
    alongside the live ones instead of the live weights themselves.
    Together these are what checkpoint selection needs: the honest way
    to choose is to compare candidates on ``val`` and then report the
    chosen one on ``test``, so a number picked as the maximum of a
    noisy curve doesn't get reported as if it were unbiased.

    The confusion rate (``metrics.toxic_confusion_rate``) is included
    because top-1/top-5 alone can't distinguish a model that makes its
    mistakes WITHIN a toxicity side from one that makes them ACROSS
    it, and only the latter is dangerous here - so a checkpoint should
    never be chosen on accuracy alone in this project.
    """

    from plant_toxicity_checker import metrics, species_match
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.plantnet_dataset import PlantNetDataset
    from plant_toxicity_checker.plantnet_labels import (
        SpeciesLabels,
        load_species_rows,
    )
    from plant_toxicity_checker.toxicity import load_table

    labels = SpeciesLabels.from_metadata()
    model, config, trained_epoch = load_trained_model(
        run_name, len(labels), epoch=epoch, use_ema=use_ema,
    )
    device = next(model.parameters()).device

    max_species = config.get("max_species")
    max_images_per_species = config.get("max_images_per_species")
    priority_species_ids = toxicity_priority_species_ids(config)
    val_ds = PlantNetDataset(
        "val", labels,
        max_species=max_species,
        max_images_per_species=max_images_per_species,
        priority_species_ids=priority_species_ids,
    )
    eval_ds = PlantNetDataset(
        split, labels,
        max_species=max_species,
        max_images_per_species=max_images_per_species,
        priority_species_ids=priority_species_ids,
    )
    val_loader = make_loader(val_ds, config["batch_size"], shuffle=False)
    eval_loader = make_loader(eval_ds, config["batch_size"], shuffle=False)

    val_probas, val_labels = _predict_probas(model, val_loader, device)
    eval_probas, eval_labels = _predict_probas(model, eval_loader, device)

    result = {
        "run": run_name,
        "split": split,
        "trained_epoch": trained_epoch,
        "weights": "ema" if use_ema else "live",
    }
    for k in (1, 3, 5, 10):
        result[f"top_{k}"] = metrics.top_k_accuracy(
            eval_probas, eval_labels, k
        )
        result[f"avg_{k}"] = metrics.average_k_accuracy(
            val_probas, val_labels, eval_probas, eval_labels, k
        )
    # The paper's headline metric: plain top-k is dominated by the few
    # species holding most of the images, so it hides what happens in
    # the long tail. See metrics.macro_top_k_accuracy.
    for k in (1, 5):
        result[f"macro_top_{k}"] = metrics.macro_top_k_accuracy(
            eval_probas, eval_labels, k
        )

    sides = species_match.species_sides(
        load_species_rows(), load_table(TOXICITY_TABLE_PATH), labels,
    )
    result["toxic_confusion"] = metrics.toxic_confusion_rate(
        eval_probas, eval_labels, sides,
    )

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    suffix += "_ema" if use_ema else ""
    (out_dir / f"eval_{split}{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _species_breakdown_result(run_name, split, trained_epoch, probas,
                               labels_idx, labels_obj):
    """The per-species/per-bucket aggregation step itself, given a
    (probas, true-label-index) pair for ``split`` that ALREADY exists -
    shared by ``run_species_breakdown`` (loads a checkpoint and reruns
    inference to get that pair) and ``run_training`` (reuses the val
    predictions it just computed this epoch for ``val_top1``/
    ``val_top5`` - no extra GPU work needed for this).
    """
    from plant_toxicity_checker import metrics
    from plant_toxicity_checker.plantnet_dataset import (
        images_per_species,
        load_metadata,
        rows_for_split,
    )

    per_class_stats = metrics.per_class_accuracy(probas, labels_idx)

    train_rows = rows_for_split(load_metadata(), "train")
    train_counts = images_per_species(train_rows)

    stats_by_species_id = {}
    species_rows = []
    for index, stats in per_class_stats.items():
        species_id = int(labels_obj.species_id_of(index))
        train_count = train_counts.get(species_id, 0)
        stats_by_species_id[species_id] = stats
        species_rows.append({
            "species_id": species_id,
            "name": labels_obj.name_of(index),
            "train_count": train_count,
            "n_eval": stats["n"],
            "top1_accuracy": stats["top1_correct"] / stats["n"],
            "top5_accuracy": stats["top5_correct"] / stats["n"],
        })
    species_rows.sort(key=lambda row: row["train_count"])

    buckets = aggregate_by_train_count_bucket(
        stats_by_species_id, train_counts
    )

    return {
        "run": run_name,
        "split": split,
        "trained_epoch": trained_epoch,
        "buckets": buckets,
        "species": species_rows,
    }


def _write_species_breakdown(run_name, split, epoch, result):
    """Write a ``_species_breakdown_result`` to
    ``outputs/<run_name>/species_breakdown_<split>[_epoch<N>].json`` -
    the ``_epoch<N>`` suffix (when ``epoch`` is given) keeps a
    per-epoch snapshot from colliding with the plain (``latest.pt``)
    result."""
    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    path = out_dir / f"species_breakdown_{split}{suffix}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def run_species_breakdown(run_name, split="val", epoch=None):
    """Per-species accuracy for a trained run's checkpoint, bucketed by
    each species' training-image count.

    No retraining involved - just a different slice through the same
    checkpoint and eval split ``run_evaluation`` already uses. Exists to
    check, cheaply, whether PlantNet-300K's long-tailed species
    distribution (hundreds of species with under 20 training images) is
    dragging accuracy down specifically for those low-data species,
    before committing to an actual imbalance-aware retrain (Phase 2).

    ``epoch``, if given, loads that specific ``epoch_<N>.pt`` snapshot
    instead of ``latest.pt`` - lets this be run once per epoch of an
    already-finished stretch of training, to see whether a bucket's
    accuracy is itself volatile across epochs (vs. just the aggregate
    top-1 number), without needing to re-run training. Written output
    is named ``species_breakdown_<split>_epoch<N>.json`` in that case,
    so it never collides with the plain (``latest.pt``) result. (A run
    trained after this was added gets these written automatically,
    every epoch, by ``run_training`` itself - this function stays
    useful for backfilling runs trained before that, e.g. the
    `plantnet-baseline` and `plantnet_convnextv2_tiny_toxicity` runs
    already in this project's history.)
    """
    import torch

    from plant_toxicity_checker.plantnet_dataset import PlantNetDataset
    from plant_toxicity_checker.plantnet_labels import SpeciesLabels

    labels = SpeciesLabels.from_metadata()
    checkpoint_name = f"epoch_{epoch}.pt" if epoch is not None else "latest.pt"
    state = torch.load(
        CHECKPOINTS_DIR / run_name / checkpoint_name, map_location="cpu"
    )
    config = state["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels), config.get("model", "resnet18"),
                        pretrained=False)
    model = model.to(device)
    model.load_state_dict(state["model_state_dict"])

    max_species = config.get("max_species")
    max_images_per_species = config.get("max_images_per_species")
    priority_species_ids = toxicity_priority_species_ids(config)
    eval_ds = PlantNetDataset(
        split, labels,
        max_species=max_species,
        max_images_per_species=max_images_per_species,
        priority_species_ids=priority_species_ids,
    )
    eval_loader = make_loader(eval_ds, config["batch_size"], shuffle=False)
    eval_probas, eval_labels = _predict_probas(model, eval_loader, device)

    result = _species_breakdown_result(
        run_name, split, state["epoch"], eval_probas, eval_labels, labels,
    )
    _write_species_breakdown(run_name, split, epoch, result)
    return result


def predict_for_species(run_name, species_ids, split="val", out_name=None):
    """Per-image predictions for a specific, named handful of species -
    not just the per-species aggregate ``run_species_breakdown`` gives.

    For each image: its on-disk path (relative to the images dir), the
    true species, the predicted top-5 (name + probability each), and
    top1/top5 correctness. Meant for picking out a few illustrative
    examples (a clean hit, a top-5-but-not-top-1 near miss, a complete
    miss) and for a small confusion matrix restricted to ``species_ids``
    - not a full per-class evaluation, which ``run_species_breakdown``
    already covers more cheaply in aggregate.
    """
    import numpy as np
    import torch

    from plant_toxicity_checker.plantnet_dataset import (
        PlantNetDataset,
        filter_species_ids,
        load_metadata,
        species_dir,
    )
    from plant_toxicity_checker.plantnet_labels import SpeciesLabels

    labels = SpeciesLabels.from_metadata()
    state = torch.load(
        CHECKPOINTS_DIR / run_name / "latest.pt", map_location="cpu"
    )
    config = state["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels), config.get("model", "resnet18"),
                        pretrained=False)
    model = model.to(device)
    model.load_state_dict(state["model_state_dict"])

    rows = filter_species_ids(load_metadata(), species_ids)
    eval_ds = PlantNetDataset(split, labels, rows=rows)
    eval_loader = make_loader(eval_ds, config["batch_size"], shuffle=False)
    eval_probas, eval_labels = _predict_probas(model, eval_loader, device)

    predictions = []
    # eval_ds.rows is in the same order the (shuffle=False) loader
    # iterated it in, so this zip lines each row up with its own result.
    for row, proba, true_index in zip(
        eval_ds.rows, eval_probas, eval_labels,
    ):
        top5_idx = np.argsort(-proba)[:5]
        true_index = int(true_index)
        predictions.append({
            "species_id": int(row["species_id"]),
            "true_name": labels.name_of(true_index),
            "split": row["split"],
            "image_path": (
                f"{row['split']}/{species_dir(row['species_id'])}/"
                f"{row['PN_hash']}.jpg"
            ),
            "top1_correct": bool(top5_idx[0] == true_index),
            "top5_correct": bool(true_index in top5_idx),
            "top5": [
                {"name": labels.name_of(int(i)), "proba": float(proba[i])}
                for i in top5_idx
            ],
        })

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = out_name or f"predictions_{split}"
    (out_dir / f"{out_name}.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    return predictions
