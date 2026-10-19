"""Real species classifier for ``pipeline.check_image``.

``pipeline.py`` is species-model-agnostic by design - it only needs an
object with a ``.predict(image, top_k=None)`` method returning
``verdict.Prediction`` objects (see ``tests/conftest.py``'s
``FakeClassifier`` for the exact contract this mirrors). This module
is the real implementation, backed by the shipped ConvNeXt V2-Tiny
run, or by the Phase 1 backbone plus a Phase 2/3 classifier head.

``torch`` is imported lazily, matching ``train.py``/``decoupled.py``'s
convention.
"""


class PlantNetClassifier:
    """A trained species model, wrapped for single-image inference.

    Two shapes of trained model exist in this project and both are
    loadable here:

    - ``PlantNetClassifier(run)`` - a whole network trained end to end
      by ``train-plantnet`` (the shipped ConvNeXt V2-Tiny model).
    - ``PlantNetClassifier(run, head_run=...)`` - the Phase 1 backbone
      with a separately trained classifier head bolted on, from the
      decoupled-training experiments.
    """

    def __init__(self, run, head_run=None, epoch=None, use_ema=False):
        from plant_toxicity_checker.plantnet_dataset import build_transforms
        from plant_toxicity_checker.plantnet_labels import SpeciesLabels

        if head_run is not None:
            from plant_toxicity_checker import decoupled

            self._model, self._labels = decoupled.load_combined_model(
                run, head_run,
            )
        else:
            from plant_toxicity_checker.train import load_trained_model

            self._labels = SpeciesLabels.from_metadata()
            self._model, _config, _epoch = load_trained_model(
                run, len(self._labels), epoch=epoch, use_ema=use_ema,
            )
            self._model.eval()
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
