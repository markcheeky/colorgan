import itertools
from pathlib import Path
from typing import Any, Generator, Iterable

import tensorflow as tf
from IPython.display import SVG
from PIL import Image

PROJ_ROOT = Path(__file__).parent.parent

IMG_EXTENSIONS = {
    ex.strip(".") for ex, f in Image.registered_extensions().items() if f in Image.OPEN
}


def plot_model(model: tf.keras.Model, dpi: int = 64, **kwargs: Any) -> SVG:
    return SVG(
        tf.keras.utils.model_to_dot(model, dpi=dpi, **kwargs).create(prog="dot", format="svg")
    )


def chunkify(iterable: Iterable, chunk_size: int) -> Generator[Any, None, None]:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            return
        yield chunk
