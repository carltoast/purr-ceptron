"""Real species classifier for ``pipeline.check_image``.

``pipeline.py`` is species-model-agnostic by design - it only needs an
object with a ``.predict(image, top_k=None)`` method returning
``verdict.Prediction`` objects (see ``tests/conftest.py``'s
``FakeClassifier`` for the exact contract this mirrors). This module
is the real implementation, backed by the Phase 1 PlantNet-300K
backbone plus the Phase 2/3 classifier head (see
``notebooks/04_decoupled_training_comparison.ipynb`` for how that head
was chosen).

``torch`` is imported lazily, matching ``train.py``/``decoupled.py``'s
convention.
"""


class PlantNetClassifier:
    """Wraps a trained backbone+head pair for single-image inference."""

    def __init__(self, backbone_run, head_run):
        from plant_toxicity_checker import decoupled
        from plant_toxicity_checker.plantnet_dataset import build_transforms

        self._model, self._labels = decoupled.load_combined_model(
            backbone_run, head_run,
        )
        self._transform = build_transforms(train=False)

    def predict(self, image, top_k=None):
        """``image`` is an already-loaded PIL image (upright, RGB) -
        see ``pipeline.load_image`` for the path/EXIF/HEIC handling
        that happens before this is ever called. Returns every class
        as a ``Prediction``, sorted by confidence descending, sliced
        to ``top_k`` (``None`` returns the full distribution, matching
        ``FakeClassifier``'s behavior).
        """
        import torch

        from plant_toxicity_checker.verdict import Prediction

        tensor = self._transform(image).unsqueeze(0)
        with torch.no_grad():
            probas = torch.softmax(self._model(tensor), dim=1)[0]

        predictions = [
            Prediction(self._labels.name_of(index), float(confidence))
            for index, confidence in enumerate(probas)
        ]
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:top_k]
