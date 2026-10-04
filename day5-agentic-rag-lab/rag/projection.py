"""
rag/projection.py — squash 384-dimensional vectors down to 2D so we can plot them.

Concept taught: PCA (Principal Component Analysis) finds the two directions in
which the chunk vectors vary the most. Projecting onto those directions gives
x, y coordinates where nearby points ≈ similar meaning. It is only an
approximation — 382 dimensions of detail are thrown away.

Implemented with NumPy's SVD (no scikit-learn):
    X_centered = X - mean
    U, S, Vt   = svd(X_centered)
    components = Vt[:2]             # top-2 directions
    xy         = X_centered @ components.T
The SAME mean and components are reused to project a query, so the query
lands in the same 2D space as the chunks.
"""
from __future__ import annotations

import numpy as np

# collection name -> {"mean": np.ndarray, "components": np.ndarray}
_MODELS: dict[str, dict] = {}


def fit(collection: str, embeddings: list[list[float]]) -> list[tuple[float, float]]:
    """Fit PCA for a collection and return the 2D points of its embeddings."""
    X = np.asarray(embeddings, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean
    if len(X) < 2:
        components = np.eye(X.shape[1])[:2]
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2]
        if components.shape[0] < 2:  # only one direction available
            components = np.vstack([components, np.zeros_like(components[0])])
    _MODELS[collection] = {"mean": mean, "components": components}
    return [(float(x), float(y)) for x, y in centered @ components.T]


def has_model(collection: str) -> bool:
    return collection in _MODELS


def project(collection: str, vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Project new vectors (e.g. a query) with the components fitted for `collection`."""
    model = _MODELS[collection]
    xy = (np.asarray(vectors, dtype=np.float64) - model["mean"]) @ model["components"].T
    return [(float(x), float(y)) for x, y in xy]


def forget(collection: str) -> None:
    _MODELS.pop(collection, None)
