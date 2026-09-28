"""Image embeddings from a neutral pretrained backbone (for exploration).

Deliberately NOT the flower classifier: that checkpoint was probably trained
on this very dataset, so its features would look unrealistically clean. A
general-purpose backbone (DINOv2 by default) shows how separable the classes
are on their own. Embeddings are L2-normalised, so a dot product is the
cosine similarity, and cached on disk keyed by model id and image list.
"""

import hashlib
from pathlib import Path

import numpy as np

from bouquet_checker.config import HF_CACHE_DIR
from bouquet_checker.pipeline import load_image

EMBED_MODEL_ID = "facebook/dinov2-small"


def cache_key(model_id, paths):
    """Fingerprint of what an embedding file was computed from."""
    digest = hashlib.sha1(model_id.encode())
    for path in paths:
        digest.update(b"\0" + str(path).encode())
    return digest.hexdigest()


def l2_normalise(matrix):
    """Scale each row to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _load_model(model_id, cache_dir=HF_CACHE_DIR):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        model_id, cache_dir=cache_dir
    )
    model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).eval()
    return processor, model


def _read_cache(cache_path, key):
    if cache_path is None or not Path(cache_path).exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        if str(data["key"]) == key:
            return data["embeddings"]
    return None


def embed_paths(paths, model_id=EMBED_MODEL_ID, batch_size=32,
                cache_path=None, show_progress=True):
    """Return an ``(n_images, dim)`` float32 array of unit-length embeddings.

    Reuses ``cache_path`` if it was computed from the same model and the
    same ordered list of images; otherwise recomputes and rewrites it.
    """
    paths = [str(p) for p in paths]
    key = cache_key(model_id, paths)
    cached = _read_cache(cache_path, key)
    if cached is not None:
        return cached

    import torch

    processor, model = _load_model(model_id)
    starts = range(0, len(paths), batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        starts = tqdm(starts, desc=f"embedding ({model_id})", unit="batch")
    chunks = []
    for start in starts:
        images = [load_image(p) for p in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            output = model(**inputs)
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        chunks.append(features.float().cpu().numpy())
    embeddings = l2_normalise(np.concatenate(chunks)).astype(np.float32)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, key=key)
    return embeddings
