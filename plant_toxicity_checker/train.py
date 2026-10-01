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

REQUIRED_CONFIG_KEYS = (
    "lr", "momentum", "nesterov", "weight_decay", "batch_size", "epochs",
    "lr_decay_epochs", "lr_decay_factor",
)

# DataLoader performance knobs - not paper hyperparameters (those are all
# in configs/*.json), just how fast data gets to the GPU, so they live
# here as fixed constants rather than in the training config.
# 8 of the remote PC's 12 cores: leaves headroom for the main process/OS
# rather than claiming every core.
NUM_WORKERS = 8


class ConfigError(ValueError):
    """A training config is missing a required key or has a bad value."""


def load_config(path):
    """Load and lightly validate a training config JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in data]
    if missing:
        raise ConfigError(f"config missing keys: {missing}")
    return data


def aspca_priority_species_ids(
    config, species_metadata_path=None, aspca_json_path=None,
):
    """``species_id``\\s to prioritize when subsetting species (see
    ``plantnet_dataset.filter_species_subset``), if the config asks for
    it via ``"prioritize_aspca_species": true`` - the PlantNet species
    ASPCA has a scraped page for, so a small ``max_species`` dev subset
    doesn't accidentally end up with none of them. Returns ``None`` if
    the config doesn't ask for this (the common case - the paper-replica
    baseline config never sets it).

    Name matching only, via ``species_match.aspca_matched_species_ids``
    - this never reads ``toxic_to``/``non_toxic_to``, so it stays outside
    the toxicity-verdict safety path entirely.
    """
    if not config.get("prioritize_aspca_species"):
        return None
    from plant_toxicity_checker import species_match
    from plant_toxicity_checker.config import ASPCA_DETAILS_JSON_PATH
    from plant_toxicity_checker.plantnet_labels import load_species_rows

    aspca_path = Path(aspca_json_path or ASPCA_DETAILS_JSON_PATH)
    aspca_doc = json.loads(aspca_path.read_text(encoding="utf-8"))
    return species_match.aspca_matched_species_ids(
        load_species_rows(species_metadata_path), aspca_doc["plants"]
    )


def build_model(num_classes):
    """ResNet-18, ImageNet-pretrained, final layer replaced for
    ``num_classes``."""
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(model, config):
    """SGD with Nesterov momentum, per the config."""
    from torch import optim

    return optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
        weight_decay=config["weight_decay"],
    )


def build_scheduler(optimizer, config):
    """Step decay at ``lr_decay_epochs`` by ``lr_decay_factor``."""
    from torch.optim.lr_scheduler import MultiStepLR

    return MultiStepLR(
        optimizer,
        milestones=config["lr_decay_epochs"],
        gamma=config["lr_decay_factor"],
    )


def make_loader(dataset, batch_size, shuffle):
    """A ``DataLoader`` with this project's standard performance knobs.

    ``persistent_workers=True`` keeps worker processes alive between
    epochs instead of respawning them each time (previously visible as
    new worker PIDs every epoch); ``pin_memory=True`` speeds up the
    CPU->GPU copy. Neither changes what's computed, only how fast data
    gets to the GPU - safe to change freely without affecting results.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )


def save_checkpoint(run_name, epoch, model, optimizer, scheduler, config):
    """Write ``checkpoints/<run_name>/latest.pt`` and a per-epoch
    snapshot."""
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
    torch.save(state, run_dir / f"epoch_{epoch}.pt")
    torch.save(state, run_dir / "latest.pt")


def load_checkpoint(run_name, model, optimizer, scheduler):
    """Load ``checkpoints/<run_name>/latest.pt`` in place; returns the
    epoch it finished."""
    import torch

    state = torch.load(
        CHECKPOINTS_DIR / run_name / "latest.pt", map_location="cpu"
    )
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
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


def train_one_epoch(model, loader, optimizer, device, log_every=100):
    """One training epoch; returns the mean training loss.

    Prints running progress every ``log_every`` batches - an epoch over
    the full training set can take a long time (largely disk-I/O bound
    on the first, cold-cache pass over the dataset), and a run with zero
    in-epoch output is hard to tell apart from a hang when checking on it
    remotely.
    """
    from torch import nn

    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_seen = 0
    n_batches = len(loader)
    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n_seen += len(labels)
        if batch_idx % log_every == 0 or batch_idx == n_batches:
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
    priority_species_ids = aspca_priority_species_ids(config)
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
    print(
        f"train={len(train_ds)} images  val={len(val_ds)} images"
        + (f"  max_species={max_species}" if max_species else "")
        + (
            f"  max_images_per_species={max_images_per_species}"
            if max_images_per_species else ""
        )
        + (
            f"  aspca_priority_species={len(priority_species_ids)}"
            if priority_species_ids else ""
        ),
        flush=True,
    )

    batch_size = config["batch_size"]
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels)).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    start_epoch = 1
    if resume:
        finished_epoch = load_checkpoint(
            run_name, model, optimizer, scheduler,
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_probas, val_labels = _predict_probas(model, val_loader, device)
        val_top1 = metrics.top_k_accuracy(val_probas, val_labels, 1)
        val_top5 = metrics.top_k_accuracy(val_probas, val_labels, 5)
        elapsed = time.time() - t0

        print(
            f"epoch {epoch}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_top1={val_top1:.4f}  val_top5={val_top5:.4f}  "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        save_checkpoint(run_name, epoch, model, optimizer, scheduler, config)

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "epochs_run": total_epochs,
                "val_top1": val_top1,
                "val_top5": val_top5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_evaluation(run_name, split):
    """Load a run's checkpoint and report top-1/3/5/10 + average-k on
    ``split``."""
    import torch

    from plant_toxicity_checker import metrics
    from plant_toxicity_checker.plantnet_dataset import PlantNetDataset
    from plant_toxicity_checker.plantnet_labels import SpeciesLabels

    labels = SpeciesLabels.from_metadata()
    state = torch.load(
        CHECKPOINTS_DIR / run_name / "latest.pt", map_location="cpu"
    )
    config = state["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels)).to(device)
    model.load_state_dict(state["model_state_dict"])

    max_species = config.get("max_species")
    max_images_per_species = config.get("max_images_per_species")
    priority_species_ids = aspca_priority_species_ids(config)
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
        "trained_epoch": state["epoch"],
    }
    for k in (1, 3, 5, 10):
        result[f"top_{k}"] = metrics.top_k_accuracy(
            eval_probas, eval_labels, k
        )
        result[f"avg_{k}"] = metrics.average_k_accuracy(
            val_probas, val_labels, eval_probas, eval_labels, k
        )

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
