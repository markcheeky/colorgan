from pathlib import Path

import tensorflow as tf
from IPython.display import SVG

PROJ_ROOT = Path(__file__).parent.parent


def plot_model(model, expand_nested=False):
    return SVG(
        tf.keras.utils.model_to_dot(model, dpi=60, expand_nested=expand_nested).create(
            prog="dot", format="svg"
        )
    )
