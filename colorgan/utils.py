import itertools
from pathlib import Path
from typing import Any, Generator, Iterable

import tensorflow as tf
import wandb
from dataset import postprocess
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


class LogPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        every_n_batch: int,
        name: str = "visualization",
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.every_n_batch = every_n_batch

    def on_train_batch_end(self, batch_idx: int, logs=None) -> None:

        if batch_idx % self.every_n_batch != 0:
            return

        table = wandb.Table(columns=["x", "pred", "y"])
        preds = self.model.predict(self.dataset)

        for pred, (x, y) in zip(preds, self.dataset.unbatch()):
            table.add_data(
                wandb.Image(postprocess(x.numpy())),
                wandb.Image(postprocess(pred)),
                wandb.Image(postprocess(y.numpy())),
            )

        wandb.log({self.name: table})
