import numpy as np
import skimage.color
import tensorflow as tf
import wandb
from dataset import postprocess, postprocess_lab

from models import ColorGan


class LogPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        vis_every_n_batch: int,
        loss_every_n_batch: int,
        param_hist_every_n_batch: int,
        name: str = "visualization",
        use_lab: bool = False,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.loss_every_n_batch = loss_every_n_batch
        self.vis_every_n_batch = vis_every_n_batch
        self.param_hist_every_n_batch = param_hist_every_n_batch
        self.use_lab = use_lab

    def on_train_batch_end(self, batch_idx: int, logs=None) -> None:

        if batch_idx % self.loss_every_n_batch == 0:
            wandb.log(logs, step=batch_idx)

        if batch_idx % self.param_hist_every_n_batch == 0:
            if isinstance(self.model, ColorGan):
                g_params = self.model.g.trainable_weights
                d_params = self.model.d.trainable_weights

                wandb.log(
                    {
                        "d_params": wandb.Histogram(np.concatenate([w.numpy().ravel() for w in d_params])),
                        "g_params": wandb.Histogram(np.concatenate([w.numpy().ravel() for w in g_params])),
                    },
                    step=batch_idx,
                )
                wandb.log({f"d_params_{w.name}": wandb.Histogram(w) for w in d_params}, step=batch_idx)
                wandb.log({f"g_params_{w.name}": wandb.Histogram(w) for w in g_params}, step=batch_idx)
            else:
                wandb.log(
                    {f"params_{w.name}": wandb.Histogram(w) for w in self.model.trainable_weights},
                    step=batch_idx,
                )

        if batch_idx % self.vis_every_n_batch != 0:
            return

        outputs = self.model.predict(self.dataset)
        if isinstance(outputs, tuple):
            preds, scores = outputs
            scores = scores.squeeze()
        else:
            preds = outputs
            scores = None

        columns = ["x", "pred", "y"]
        if scores is not None:
            columns.append("discriminator_score")

        table = wandb.Table(columns)

        for i, (pred, (x, y)) in enumerate(zip(preds, self.dataset.unbatch())):
            if self.use_lab:
                row_cells = [
                    wandb.Image((x.numpy() * 255).astype(np.uint8)),
                    wandb.Image(postprocess_lab(x, pred)),
                    wandb.Image(postprocess_lab(x, y)),
                ]
            else:
                row_cells = [
                    wandb.Image(postprocess(x.numpy())),
                    wandb.Image(postprocess(pred)),
                    wandb.Image(postprocess(y.numpy())),
                ]
            if scores is not None:
                row_cells.append(scores[i])
            table.add_data(*row_cells)

        wandb.log({self.name: table}, step=batch_idx)
