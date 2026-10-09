"""Decoupled classifier-head training (Phase 2/3 proof-of-concept).

Freeze the Phase 1 backbone, cache its features once, then train only
a fresh linear classifier head on those cached vectors - cheap enough
(minutes, not hours) to compare several loss variants side by side,
instead of retraining the whole backbone from scratch for each one.
See ``CLAUDE.md``'s "Phase 2/3 approach" section for the full
reasoning (Kang et al., ICLR 2020, arXiv:1910.09217).

``torch``/``torchvision`` are imported inside functions, not at module
level, matching ``train.py``'s convention - importing this module for
``load_config``/``class_balanced_weights``-style pure logic never
needs torch installed.
"""

import json
from pathlib import Path

import numpy as np

REQUIRED_CONFIG_KEYS = ("loss", "epochs")
_VALID_LOSSES = ("ce", "cb", "balanced_softmax")


def load_config(path):
    """Load and lightly validate a decoupled-training config JSON
    file. Reuses ``train.ConfigError`` so both training paths' CLI
    error handling looks identical."""
    from plant_toxicity_checker.train import ConfigError

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in data]
    if missing:
        raise ConfigError(f"config missing keys: {missing}")
    if data["loss"] not in _VALID_LOSSES:
        raise ConfigError(f"unknown loss: {data['loss']!r}")
    return data


def features_path(backbone_run, split):
    """Where ``extract_features`` saves / ``evaluate_classifier_head``
    loads cached features for one (backbone checkpoint, split) pair."""
    from plant_toxicity_checker.train import OUTPUTS_DIR

    return OUTPUTS_DIR / backbone_run / f"features_{split}.npz"


def extract_features(backbone_run, split, batch_size=64, log_every=100):
    """Cache the Phase 1 backbone's penultimate-layer (512-dim)
    features for every image in ``split``, keyed by the same class
    index ``PlantNetDataset`` already uses. Needs the real dataset and
    a GPU - remote PC only, same as Phase 1 training itself.

    Deterministic eval-style transforms (``CenterCrop``, no
    augmentation) regardless of ``split`` - the whole point is one
    reusable, fixed vector per image, not a fresh augmented view every
    epoch (that's what avoids the I/O-bound cost of full retraining).

    Prints progress every ``log_every`` batches, same reasoning as
    ``train.train_one_epoch``: a single forward pass over the full
    train split is still I/O-bound and can take comparable wall-clock
    to a training epoch - silent for that whole stretch is hard to
    tell apart from a hang when checking on it remotely.
    """
    import time

    import torch
    from torch import nn

    from plant_toxicity_checker.plantnet_dataset import (
        PlantNetDataset,
        build_transforms,
    )
    from plant_toxicity_checker.plantnet_labels import SpeciesLabels
    from plant_toxicity_checker.train import (
        CHECKPOINTS_DIR,
        build_model,
        make_loader,
    )

    labels_obj = SpeciesLabels.from_metadata()
    state = torch.load(
        CHECKPOINTS_DIR / backbone_run / "latest.pt", map_location="cpu",
    )
    model = build_model(len(labels_obj))
    model.load_state_dict(state["model_state_dict"])
    model.fc = nn.Identity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = PlantNetDataset(
        split, labels_obj, transform=build_transforms(train=False),
    )
    loader = make_loader(dataset, batch_size, shuffle=False)
    n_batches = len(loader)
    t0 = time.time()

    all_features, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(loader, start=1):
            feats = model(images.to(device))
            all_features.append(feats.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            if batch_idx % log_every == 0 or batch_idx == n_batches:
                elapsed = time.time() - t0
                print(
                    f"  batch {batch_idx}/{n_batches}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    features = np.concatenate(all_features)
    targets = np.concatenate(all_labels)

    out_path = features_path(backbone_run, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=features, labels=targets)
    return out_path


def _load_backbone_fc(backbone_run):
    """Load the Phase 1 backbone checkpoint's original, untouched
    model (``fc`` included) - the one place that checkpoint is loaded,
    reused by every "score cached features through the original
    model" function below.
    """
    import torch

    from plant_toxicity_checker.plantnet_labels import SpeciesLabels
    from plant_toxicity_checker.train import CHECKPOINTS_DIR, build_model

    labels_obj = SpeciesLabels.from_metadata()
    state = torch.load(
        CHECKPOINTS_DIR / backbone_run / "latest.pt", map_location="cpu",
    )
    model = build_model(len(labels_obj))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, labels_obj


def _backbone_predict(model, backbone_run, split):
    """Run a loaded backbone model over cached features for one
    split."""
    import torch

    data = np.load(features_path(backbone_run, split))
    features = torch.from_numpy(data["features"]).float()
    with torch.no_grad():
        probas = torch.softmax(model.fc(features), dim=1).numpy()
    return probas, data["labels"]


def sanity_check_extraction(backbone_run, split="val"):
    """Confirm ``extract_features``'s pipeline is correct: run the
    cached features back through the ORIGINAL (untouched) backbone's
    ``fc`` layer weights and check top-1/top-5 against what that
    checkpoint already reported during Phase 1 training. If they don't
    match, something in extraction is broken (wrong transform, wrong
    layer, a normalization mismatch) - fix before trusting any
    classifier-head variant trained on these features.
    """
    from plant_toxicity_checker import metrics

    model, _labels_obj = _load_backbone_fc(backbone_run)
    probas, labels = _backbone_predict(model, backbone_run, split)

    return {
        "top_1": metrics.top_k_accuracy(probas, labels, 1),
        "top_5": metrics.top_k_accuracy(probas, labels, 5),
    }


def class_balanced_weights(class_counts, beta=0.9999):
    """Per-class loss weights via the "effective number of samples"
    formula (Cui et al. 2019, *Class-Balanced Loss Based on Effective
    Number of Samples*): ``weight_c = (1-beta) / (1-beta**n_c)``,
    normalized to mean 1 across classes actually present in training
    (``n_c > 0``) - classes with zero training examples get weight 0
    (never selected as a true label anyway; kept at a defined value
    rather than dividing by zero).

    ``class_counts`` is a tensor/array of per-class training-image
    counts, in class-index order. Returns a ``torch.float`` tensor of
    the same length.
    """
    import torch

    class_counts = torch.as_tensor(class_counts, dtype=torch.float64)
    present = class_counts > 0
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = torch.where(
        present, (1.0 - beta) / effective_num, torch.zeros_like(class_counts)
    )
    weights = weights / weights[present].mean()
    return weights.float()


def balanced_softmax_logits(logits, class_counts, tau=1.0):
    """Training-time logit adjustment (Ren et al. 2020, *Balanced
    Meta-Softmax for Long-Tailed Visual Recognition*): add
    ``tau * log(n_c / N)`` to each class's logit before the softmax/CE,
    so the model is encouraged to learn logits naturally calibrated for
    a balanced inference distribution. Only ever applied at TRAINING
    time - the trained model's raw logits need no further adjustment
    at inference (unlike the verdict path's matching, nothing here
    touches toxicity).

    ``class_counts`` as in ``class_balanced_weights``; classes with
    zero training examples are clamped to count 1 before the log
    (avoids ``-inf``) - they still get a strongly negative adjustment.
    ``tau`` is the one tunable knob here (the original paper fixes it
    at 1.0 - this is the generalized "logit adjustment" form, Menon et
    al. 2021): ``tau > 1`` pushes harder toward balancing rare classes
    (at some cost to common-class accuracy), ``tau < 1`` is gentler,
    ``tau = 0`` disables the adjustment entirely (reduces to plain CE).
    """
    import torch

    class_counts = torch.as_tensor(
        class_counts, dtype=torch.float64, device=logits.device,
    )
    total = class_counts.sum()
    adjustment = tau * torch.log(torch.clamp(class_counts, min=1.0) / total)
    return logits + adjustment.float()


def toxicity_cost_term(probas, labels, species_sides, eps=1e-6):
    """Auxiliary loss penalizing predicted probability mass on the
    OPPOSITE toxicity side from the true species, for samples whose
    true side is known - added to the primary loss as
    ``weight * toxicity_cost_term(...)``.

    ``probas`` are softmax probabilities (not logits), shape
    ``(batch, num_classes)``. ``species_sides`` as in
    ``metrics.toxic_confusion_rate``. Samples whose true class has no
    known side contribute nothing (never guessed at - matches the
    project's "exclude, don't infer" rule for unverified toxicity
    signal). Returns a 0-d tensor (0 if no sample in the batch has a
    known side).
    """
    import torch

    sides = torch.as_tensor(species_sides, device=probas.device)
    true_sides = sides[labels]
    known_mask = true_sides != -1
    if not bool(known_mask.any()):
        return probas.new_zeros(())

    opposite_side_mask = (sides.unsqueeze(0) != -1) & (
        sides.unsqueeze(0) != true_sides.unsqueeze(1)
    )
    wrong_side_mass = (probas * opposite_side_mask).sum(dim=1)
    wrong_side_mass = wrong_side_mass[known_mask]
    return -torch.log(1.0 - wrong_side_mass + eps).mean()


def _class_counts_by_index(labels_obj):
    """Dense per-class-index array of TRAINING image counts, for the
    loss functions that need them."""
    from plant_toxicity_checker.plantnet_dataset import (
        images_per_species,
        load_metadata,
        rows_for_split,
    )

    train_rows = rows_for_split(load_metadata(), "train")
    counts_by_species_id = images_per_species(train_rows)
    counts = [0] * len(labels_obj)
    for species_id, count in counts_by_species_id.items():
        try:
            index = labels_obj.index_of_species_id(species_id)
        except KeyError:
            continue
        counts[index] = count
    return counts


def train_classifier_head(features_file, config, run_name, backbone_run):
    """Train a fresh linear classifier head on cached, frozen-backbone
    features - loss per ``config``, no image I/O at all. Cheap enough
    to default to CPU.

    ``config`` keys: ``loss`` (``"ce"``/``"cb"``/``"balanced_softmax"``),
    ``cb_beta`` (if ``loss == "cb"``), ``balanced_softmax_tau`` (if
    ``loss == "balanced_softmax"``), ``toxicity_cost_weight`` (0 =
    off, Phase 2 only; >0 = Phase 2+3 combined), ``epochs``,
    ``batch_size``. ``optimizer`` (``"adam"``, the default, or
    ``"sgd"``) picks how the head is trained: ``"adam"`` just needs
    ``lr`` (default ``1e-3``); ``"sgd"`` reuses
    ``train.build_optimizer``/``build_scheduler`` as-is, so it needs
    the same keys the backbone's own config does - ``lr``,
    ``momentum``, ``nesterov``, ``weight_decay``, and, if a decay
    schedule is wanted, ``lr_decay_epochs``/``lr_decay_factor``. This
    matters because the original Phase 1 head was trained *jointly*
    with SGD/momentum/decay, not Adam - plain Adam retraining of just
    the head does not reliably reproduce it, even under the identical
    loss (see ``CLAUDE.md``'s "Phase 2/3 approach" section and the
    04 notebook's take-aways on the decoupling/baseline gap this
    option exists to close). ``seed``, if given, seeds torch before
    the head's weight init and the per-epoch shuffle order - lets a
    config be re-run with a different seed to check whether a small
    result difference from another config is real or just run-to-run
    noise (every earlier run left this unset, i.e. whatever the
    process's ambient RNG state happened to be - not reproducible by
    itself, but fine as "one arbitrary seed" to compare a second one
    against). Saves ``checkpoints/<run_name>/head.pt``.
    """
    import torch
    from torch import nn, optim

    from plant_toxicity_checker import species_match
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.plantnet_labels import (
        SpeciesLabels,
        load_species_rows,
    )
    from plant_toxicity_checker.toxicity import load_table
    from plant_toxicity_checker.train import (
        CHECKPOINTS_DIR,
        build_optimizer,
        build_scheduler,
    )

    if "seed" in config:
        torch.manual_seed(config["seed"])

    labels_obj = SpeciesLabels.from_metadata()
    num_classes = len(labels_obj)

    data = np.load(features_file)
    features = torch.from_numpy(data["features"]).float()
    targets = torch.from_numpy(data["labels"]).long()

    loss_kind = config["loss"]
    class_counts = None
    if loss_kind in ("cb", "balanced_softmax"):
        class_counts = torch.tensor(
            _class_counts_by_index(labels_obj), dtype=torch.float64,
        )

    cb_weights = None
    if loss_kind == "cb":
        cb_weights = class_balanced_weights(
            class_counts, beta=config.get("cb_beta", 0.9999),
        )

    toxicity_cost_weight = float(config.get("toxicity_cost_weight", 0.0))
    sides = None
    if toxicity_cost_weight:
        table = load_table(TOXICITY_TABLE_PATH)
        species_rows = load_species_rows()
        sides = species_match.species_sides(species_rows, table, labels_obj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = nn.Linear(features.shape[1], num_classes).to(device)

    optimizer_kind = config.get("optimizer", "adam")
    scheduler = None
    if optimizer_kind == "sgd":
        optimizer = build_optimizer(head, config)
        if "lr_decay_epochs" in config:
            scheduler = build_scheduler(optimizer, config)
    else:
        optimizer = optim.Adam(head.parameters(), lr=config.get("lr", 1e-3))

    features = features.to(device)
    targets = targets.to(device)
    if class_counts is not None:
        class_counts = class_counts.to(device)
    if cb_weights is not None:
        cb_weights = cb_weights.to(device)

    batch_size = config.get("batch_size", 512)
    n = features.shape[0]
    epochs = config["epochs"]

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_features = features[idx]
            batch_targets = targets[idx]

            logits = head(batch_features)
            if loss_kind == "balanced_softmax":
                logits = balanced_softmax_logits(
                    logits, class_counts,
                    tau=config.get("balanced_softmax_tau", 1.0),
                )

            weight = cb_weights if loss_kind == "cb" else None
            loss = nn.functional.cross_entropy(
                logits, batch_targets, weight=weight,
            )
            if toxicity_cost_weight:
                probas = torch.softmax(logits, dim=1)
                loss = loss + toxicity_cost_weight * toxicity_cost_term(
                    probas, batch_targets, sides,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if scheduler is not None:
            scheduler.step()
        print(f"epoch {epoch}/{epochs}  loss={total_loss / n:.4f}", flush=True)

    run_dir = CHECKPOINTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "backbone_run": backbone_run,
            "config": config,
        },
        run_dir / "head.pt",
    )
    return run_dir / "head.pt"


def _full_metrics(
    val_probas, val_labels, eval_probas, eval_labels, labels_obj,
):
    """top-1/3/5/10 + avg-k + ``toxic_confusion_rate`` for one (val,
    eval) probability pair - the one metric suite every row of the
    Phase 1/2/3 comparison is computed with, whether the probabilities
    came from a retrained head (``evaluate_classifier_head``) or the
    original backbone's own ``fc`` layer (``baseline_eval_from_features``),
    so the comparison is apples-to-apples rather than mixing this
    project's old training-time metrics with new ones.
    """
    from plant_toxicity_checker import metrics, species_match
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.plantnet_labels import load_species_rows
    from plant_toxicity_checker.toxicity import load_table

    result = {}
    for k in (1, 3, 5, 10):
        result[f"top_{k}"] = metrics.top_k_accuracy(
            eval_probas, eval_labels, k
        )
        result[f"avg_{k}"] = metrics.average_k_accuracy(
            val_probas, val_labels, eval_probas, eval_labels, k
        )

    table = load_table(TOXICITY_TABLE_PATH)
    species_rows = load_species_rows()
    sides = species_match.species_sides(species_rows, table, labels_obj)
    result["toxic_confusion"] = metrics.toxic_confusion_rate(
        eval_probas, eval_labels, sides
    )
    return result


def baseline_eval_from_features(backbone_run, split):
    """The Phase 1 baseline's full metric suite - top-1/3/5/10, avg-k,
    and ``toxic_confusion_rate`` - computed from CACHED features run
    back through the backbone's own original ``fc`` layer, not a
    retrained head. Lets the Phase 1 baseline sit as a row in the same
    comparison table as the Phase 2/3 classifier-head variants, scored
    by the exact same code path, rather than reusing the two
    training-time numbers already in ``outputs/<run>/metrics.json``
    (which predate ``toxic_confusion_rate`` and used a different eval
    pipeline). Needs ``extract_features`` already run for both
    ``split`` and ``"val"``.
    """
    from plant_toxicity_checker.train import OUTPUTS_DIR

    model, labels_obj = _load_backbone_fc(backbone_run)
    val_probas, val_labels = _backbone_predict(model, backbone_run, "val")
    eval_probas, eval_labels = _backbone_predict(model, backbone_run, split)

    result = {"run": backbone_run, "split": split}
    result.update(
        _full_metrics(
            val_probas, val_labels, eval_probas, eval_labels, labels_obj,
        )
    )

    out_dir = OUTPUTS_DIR / backbone_run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_from_features_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _load_classifier_head(run_name):
    """Load a trained classifier head, its labels, and the
    backbone_run/config it was trained against - the one place a head
    checkpoint is loaded, reused by every classifier-head analysis
    function below.
    """
    import torch
    from torch import nn

    from plant_toxicity_checker.plantnet_labels import SpeciesLabels
    from plant_toxicity_checker.train import CHECKPOINTS_DIR

    head_state = torch.load(
        CHECKPOINTS_DIR / run_name / "head.pt", map_location="cpu",
    )
    labels_obj = SpeciesLabels.from_metadata()
    head = nn.Linear(
        head_state["state_dict"]["weight"].shape[1], len(labels_obj),
    )
    head.load_state_dict(head_state["state_dict"])
    head.eval()
    return head, labels_obj, head_state["backbone_run"], head_state["config"]


def _classifier_head_predict(head, backbone_run, split):
    """Run a loaded classifier head over cached features for one
    split. Errors clearly if the cache is missing rather than silently
    re-extracting (a 70-90min remote-only step shouldn't happen
    implicitly).
    """
    import torch

    path = features_path(backbone_run, split)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached features at {path} - run `extract-features "
            f"--run {backbone_run} --split {split}` first"
        )
    data = np.load(path)
    features = torch.from_numpy(data["features"]).float()
    with torch.no_grad():
        probas = torch.softmax(head(features), dim=1).numpy()
    return probas, data["labels"]


def evaluate_classifier_head(run_name, split):
    """Evaluate a trained classifier head on cached features for
    ``split``.

    Computes top-1/3/5/10 + average-k (reusing cached val features for
    the average-k lambda calibration, same pattern as
    ``train.run_evaluation``) and ``toxic_confusion_rate``. Writes
    ``outputs/<run_name>/eval_<split>.json``.
    """
    from plant_toxicity_checker.train import OUTPUTS_DIR

    head, labels_obj, backbone_run, config = _load_classifier_head(run_name)
    val_probas, val_labels = _classifier_head_predict(
        head, backbone_run, "val",
    )
    eval_probas, eval_labels = _classifier_head_predict(
        head, backbone_run, split,
    )

    result = {
        "run": run_name, "backbone_run": backbone_run, "split": split,
        "config": config,
    }
    result.update(
        _full_metrics(
            val_probas, val_labels, eval_probas, eval_labels, labels_obj,
        )
    )

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _species_breakdown(probas, eval_labels, labels_obj):
    """Per-species top-1/top-5 accuracy, bucketed by each species'
    training-image count - the same slice ``train.run_species_breakdown``
    provides for a full backbone checkpoint, computed here from
    whatever probabilities the caller already has. No image I/O or GPU
    needed since the caller already has probabilities from cached
    features.
    """
    from plant_toxicity_checker import metrics
    from plant_toxicity_checker.plantnet_dataset import (
        images_per_species,
        load_metadata,
        rows_for_split,
    )
    from plant_toxicity_checker.train import aggregate_by_train_count_bucket

    per_class_stats = metrics.per_class_accuracy(probas, eval_labels)

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
        stats_by_species_id, train_counts,
    )
    return buckets, species_rows


def classifier_head_species_breakdown(run_name, split="val"):
    """Per-species accuracy for a trained classifier head, bucketed by
    training-image count - the Phase 2/3 equivalent of
    ``train.run_species_breakdown``, computed from cached features (no
    image I/O, no GPU needed) instead of re-running the full backbone.
    Needs ``extract_features`` already run for ``split``. Writes
    ``outputs/<run_name>/species_breakdown_<split>.json``.
    """
    from plant_toxicity_checker.train import OUTPUTS_DIR

    head, labels_obj, backbone_run, _config = _load_classifier_head(run_name)
    probas, eval_labels = _classifier_head_predict(head, backbone_run, split)

    buckets, species_rows = _species_breakdown(
        probas, eval_labels, labels_obj,
    )

    result = {
        "run": run_name, "split": split, "buckets": buckets,
        "species": species_rows,
    }
    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"species_breakdown_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def toxic_confusion_detail(probas, eval_labels, labels_obj):
    """Named per-species detail behind ``metrics.toxic_confusion_rate``
    - which specific toxic/non-toxic species pairs a model's top-1
    predictions cross, and how often. Diagnostic only, not written to
    a results file like ``evaluate_classifier_head`` - meant for
    notebook-side exploration of one settled candidate (e.g. is this
    config confusing the SAME species the Phase 1 baseline already
    struggled with, or different ones).
    """
    from plant_toxicity_checker import metrics, species_match
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.plantnet_labels import load_species_rows
    from plant_toxicity_checker.toxicity import load_table

    table = load_table(TOXICITY_TABLE_PATH)
    species_rows = load_species_rows()
    sides = species_match.species_sides(species_rows, table, labels_obj)

    pairs = metrics.toxic_confusion_pairs(probas, eval_labels, sides)
    return [
        {
            "true_species": labels_obj.name_of(t),
            "predicted_species": labels_obj.name_of(p),
            "count": c,
        }
        for t, p, c in pairs
    ]


def classifier_head_toxic_confusion_detail(run_name, split="val"):
    """``toxic_confusion_detail`` for a trained classifier head."""
    head, labels_obj, backbone_run, _config = _load_classifier_head(run_name)
    probas, eval_labels = _classifier_head_predict(head, backbone_run, split)
    return toxic_confusion_detail(probas, eval_labels, labels_obj)


def baseline_toxic_confusion_detail(backbone_run, split="val"):
    """``toxic_confusion_detail`` for the Phase 1 baseline, scored
    through its original ``fc`` layer on cached features - the same
    apples-to-apples reasoning as ``baseline_eval_from_features``.
    """
    model, labels_obj = _load_backbone_fc(backbone_run)
    probas, eval_labels = _backbone_predict(model, backbone_run, split)
    return toxic_confusion_detail(probas, eval_labels, labels_obj)


def load_combined_model(backbone_run, head_run):
    """Assemble the frozen backbone + a trained classifier head into
    one model ready for single-image inference - the deployed-model
    equivalent of what ``extract_features`` + ``evaluate_classifier_head``
    do as two separate cached-feature stages, combined here since
    there's no batch of images to cache features for.

    Returns ``(model, labels_obj)``; ``model(image_tensor_batch)``
    gives raw logits straight from the trained head.
    """
    from plant_toxicity_checker.train import CHECKPOINTS_DIR

    backbone_path = CHECKPOINTS_DIR / backbone_run / "latest.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(
            f"no backbone checkpoint at {backbone_path} - train it "
            f"with `train-plantnet --config ...` or pull it from the "
            f"remote PC with scripts/remote/pull.sh checkpoints"
        )
    head_path = CHECKPOINTS_DIR / head_run / "head.pt"
    if not head_path.exists():
        raise FileNotFoundError(
            f"no classifier head checkpoint at {head_path} - train it "
            f"with `train-classifier --config ...` or pull it from the "
            f"remote PC with scripts/remote/pull.sh checkpoints"
        )

    model, labels_obj = _load_backbone_fc(backbone_run)
    head, _head_labels_obj, head_backbone_run, _config = (
        _load_classifier_head(head_run)
    )
    if head_backbone_run != backbone_run:
        raise ValueError(
            f"head {head_run!r} was trained on backbone_run "
            f"{head_backbone_run!r}, not {backbone_run!r}"
        )
    model.fc = head
    model.eval()
    return model, labels_obj


def verdict_tier_coverage(probas, labels_obj, table):
    """Empirical TOXIC/CAUTION/LIKELY_SAFE distribution for one
    model's predictions, run through the real ``verdict.assess()``
    safety path - the "verdict-tier coverage" row notebook 00's Key
    Metrics table carried as ``TBD`` until there was a real classifier
    (Phase 4) to measure it with. Pure measurement - never used to
    change ``assess()``'s own behavior, only to report on it.
    """
    import collections

    from plant_toxicity_checker.verdict import Prediction, assess

    counts = collections.Counter()
    for row in probas:
        predictions = [
            Prediction(labels_obj.name_of(i), float(confidence))
            for i, confidence in enumerate(row)
        ]
        counts[assess(predictions, table).status.value] += 1
    total = sum(counts.values())
    return {status: count / total for status, count in counts.items()}


def classifier_head_verdict_tiers(run_name, split="val"):
    """``verdict_tier_coverage`` for a trained classifier head. Writes
    ``outputs/<run_name>/verdict_tiers_<split>.json``.
    """
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.toxicity import load_table
    from plant_toxicity_checker.train import OUTPUTS_DIR

    head, labels_obj, backbone_run, _config = _load_classifier_head(run_name)
    probas, _eval_labels = _classifier_head_predict(head, backbone_run, split)
    table = load_table(TOXICITY_TABLE_PATH)
    tiers = verdict_tier_coverage(probas, labels_obj, table)

    result = {"run": run_name, "split": split, "tiers": tiers}
    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"verdict_tiers_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def baseline_verdict_tiers(backbone_run, split="val"):
    """``verdict_tier_coverage`` for the Phase 1 baseline, scored
    through its original ``fc`` layer on cached features - the same
    apples-to-apples reasoning as ``baseline_eval_from_features``.
    """
    from plant_toxicity_checker.config import TOXICITY_TABLE_PATH
    from plant_toxicity_checker.toxicity import load_table
    from plant_toxicity_checker.train import OUTPUTS_DIR

    model, labels_obj = _load_backbone_fc(backbone_run)
    probas, _eval_labels = _backbone_predict(model, backbone_run, split)
    table = load_table(TOXICITY_TABLE_PATH)
    tiers = verdict_tier_coverage(probas, labels_obj, table)

    result = {"run": backbone_run, "split": split, "tiers": tiers}
    out_dir = OUTPUTS_DIR / backbone_run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"verdict_tiers_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
