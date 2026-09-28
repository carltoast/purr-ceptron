"""Tests for embedding caching and normalisation (the model is faked)."""

import numpy as np
import pytest
import torch
from PIL import Image

from bouquet_checker import embeddings


def make_images(tmp_path, n):
    paths = []
    for i in range(n):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (16, 16), (i * 20 % 255, 0, 0)).save(path)
        paths.append(str(path))
    return paths


class FakeModel:
    calls = 0

    def __call__(self, pixel_values):
        FakeModel.calls += 1
        n = pixel_values.shape[0]
        features = torch.arange(1, n * 4 + 1, dtype=torch.float32)
        features = features.reshape(n, 4)
        return type("Out", (), {"pooler_output": features})()


def fake_processor(images, return_tensors):
    return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


@pytest.fixture(autouse=True)
def fake_backbone(monkeypatch):
    FakeModel.calls = 0
    monkeypatch.setattr(
        embeddings, "_load_model",
        lambda model_id: (fake_processor, FakeModel()),
    )


def test_cache_key_depends_on_model_paths_and_order():
    base = embeddings.cache_key("m", ["a", "b"])
    assert base == embeddings.cache_key("m", ["a", "b"])
    assert base != embeddings.cache_key("other", ["a", "b"])
    assert base != embeddings.cache_key("m", ["b", "a"])
    assert base != embeddings.cache_key("m", ["a"])


def test_l2_normalise_gives_unit_rows_and_keeps_zero_rows():
    out = embeddings.l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])


def test_embed_paths_returns_unit_length_float32_rows(tmp_path):
    out = embeddings.embed_paths(
        make_images(tmp_path, 5), batch_size=2, show_progress=False
    )
    assert out.shape == (5, 4) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert FakeModel.calls == 3                      # ceil(5 / 2) batches


def test_embed_paths_reuses_a_matching_cache(tmp_path):
    paths = make_images(tmp_path, 3)
    cache = tmp_path / "sub" / "emb.npz"
    first = embeddings.embed_paths(paths, cache_path=cache,
                                   show_progress=False)
    calls = FakeModel.calls
    second = embeddings.embed_paths(paths, cache_path=cache,
                                    show_progress=False)
    assert FakeModel.calls == calls                  # no recompute
    assert np.array_equal(first, second)


@pytest.mark.parametrize("change", ["order", "model", "subset"])
def test_a_stale_cache_is_recomputed(tmp_path, change):
    paths = make_images(tmp_path, 3)
    cache = tmp_path / "emb.npz"
    embeddings.embed_paths(paths, cache_path=cache, show_progress=False)
    calls = FakeModel.calls
    kwargs = {"model_id": embeddings.EMBED_MODEL_ID}
    if change == "order":
        paths = paths[::-1]
    elif change == "model":
        kwargs["model_id"] = "some/other-model"
    else:
        paths = paths[:2]
    out = embeddings.embed_paths(paths, cache_path=cache,
                                 show_progress=False, **kwargs)
    assert FakeModel.calls > calls
    assert len(out) == len(paths)


def test_falls_back_to_cls_token_without_a_pooler(tmp_path, monkeypatch):
    class NoPooler:
        def __call__(self, pixel_values):
            n = pixel_values.shape[0]
            hidden = torch.ones(n, 3, 4)
            return type("Out", (), {"pooler_output": None,
                                    "last_hidden_state": hidden})()

    monkeypatch.setattr(
        embeddings, "_load_model",
        lambda model_id: (fake_processor, NoPooler()),
    )
    paths = make_images(tmp_path, 2)
    out = embeddings.embed_paths(paths, show_progress=False)
    assert out.shape == (2, 4)
