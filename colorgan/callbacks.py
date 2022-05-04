import tensorflow as tf
import wandb
from dataset import postprocess


class LogPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        vis_every_n_batch: int,
        loss_every_n_batch: int,
        name: str = "visualization",
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.loss_every_n_batch = loss_every_n_batch
        self.vis_every_n_batch = vis_every_n_batch

    def on_train_batch_end(self, batch_idx: int, logs=None) -> None:

        if batch_idx % self.loss_every_n_batch == 0:
            wandb.log(logs)

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
            row_cells = [
                wandb.Image(postprocess(x.numpy())),
                wandb.Image(postprocess(pred)),
                wandb.Image(postprocess(y.numpy())),
            ]
            if scores is not None:
                row_cells.append(scores[i])
            table.add_data(*row_cells)

        wandb.log({self.name: table})
