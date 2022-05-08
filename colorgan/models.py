from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Layer,
    LeakyReLU,
)
from tensorflow.keras.models import Model, Sequential


def get_unet_encoder_block(
    inputs: Layer,
    kernel_num: int,
    kernel_size: Tuple[int, int],
    stride: int,
    init: Initializer,
    batch_norm: bool,
    leakiness: float = 0.2,
) -> Sequential:

    block = Sequential()

    block.add(Conv2D(kernel_num, kernel_size, padding="same", kernel_initializer=init, strides=stride))
    if batch_norm:
        block.add(BatchNormalization())
    block.add(LeakyReLU(alpha=leakiness))

    return block(inputs)


def get_unet_decoder_block(
    inputs: Layer,
    kernel_num: int,
    kernel_size: Tuple[int, int],
    stride: int,
    dropout: bool,
    init: Initializer,
    dropout_rate: float = 0.5,
    activation: str = "relu",
) -> Sequential:

    block = Sequential()

    if isinstance(inputs, list):
        inputs = Concatenate(axis=3)(inputs)

    block.add(
        Conv2DTranspose(kernel_num, kernel_size, padding="same", kernel_initializer=init, strides=stride)
    )
    block.add(BatchNormalization())
    if dropout:
        block.add(Dropout(dropout_rate))
    block.add(Activation(activation))

    return block(inputs)


def get_unet_generator(
    encoder_kernel_nums: List[int] = [64, 128, 256, 512, 512, 512, 512, 512],
    decoder_kernel_nums: List[int] = [512, 512, 512, 512, 256, 128, 64],
    kernel_size: Tuple[int, int] = (4, 4),
    stride: int = 2,
    use_lab: bool = False,
) -> Model:

    if use_lab:
        input_layer = Input(shape=[None, None, 1])
        decoder_kernel_nums = decoder_kernel_nums + [2]
    else:
        input_layer = Input(shape=[None, None, 3])
        decoder_kernel_nums = decoder_kernel_nums + [3]

    if len(decoder_kernel_nums) != len(encoder_kernel_nums):
        raise ValueError("number of encoder and decoder layers must match")

    encoder_blocks: List[Layer] = []
    decoder_blocks: List[Layer] = []

    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=0)

    for kernel_num in encoder_kernel_nums:

        if len(encoder_blocks) == 0:
            inputs = input_layer
            batch_norm = False
        else:
            inputs = encoder_blocks[-1]
            batch_norm = True

        encoder_blocks.append(
            get_unet_encoder_block(inputs, kernel_num, kernel_size, stride, init, batch_norm)
        )

    for skip_src, kernel_num in zip(encoder_blocks[::-1], decoder_kernel_nums):
        if len(decoder_blocks) == 0:
            inputs = skip_src
        else:
            inputs = [skip_src, decoder_blocks[-1]]

        dropout = len(decoder_blocks) < 3
        is_last = len(decoder_blocks) == len(decoder_kernel_nums) - 1
        activation = "tanh" if is_last else "relu"

        decoder_blocks.append(
            get_unet_decoder_block(
                inputs,
                kernel_num,
                kernel_size,
                stride,
                dropout,
                init,
                activation=activation,
            )
        )

    return Model(inputs=input_layer, outputs=decoder_blocks[-1])


def get_discriminator_block(
    kernel_num: int,
    kernel_size: Tuple[int, ...],
    stride: int,
    init: Initializer,
    batch_norm: bool,
) -> Sequential:

    return Sequential(
        [
            Conv2D(kernel_num, kernel_size, padding="same", kernel_initializer=init, strides=stride),
            BatchNormalization() if batch_norm else Layer(),
            LeakyReLU(alpha=0.2),
        ]
    )


def get_discriminator(
    kernel_nums: Tuple[int, ...] = (64, 128, 256, 256, 512),
    kernel_size: Tuple[int, ...] = (4, 4),
    stride: int = 2,
    use_lab: bool = False,
) -> Model:

    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=42)

    if use_lab:
        input_l = Input((None, None, 1), name="lightness")
        input_ab = Input((None, None, 2), name="ab")
        inputs = [input_l, input_ab]
    else:
        input_bw = Input((None, None, 3), name="bw")
        input_colorized = Input((None, None, 3), name="colorized")
        inputs = [input_bw, input_colorized]

    x = Concatenate()(inputs)

    for i, kernel_num in enumerate(kernel_nums):
        not_first = i != 0
        block = get_discriminator_block(kernel_num, kernel_size, stride, init=init, batch_norm=not_first)
        x = block(x)
    x = Conv2D(1, kernel_size=kernel_size)(x)
    return Model(inputs=inputs, outputs=x)


class ColorGan(Model):
    def __init__(self, g: Model, d: Model, weight_mae_loss: float = 50.0, use_mae=True):
        super().__init__()

        self.g = g
        self.d = d

        self.weight_mae_loss = weight_mae_loss
        self.use_mae = use_mae

    def compile(self, d_optimizer, g_optimizer, end_loss):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.end_loss = end_loss

    def g_loss(
        self,
        fakes: tf.Tensor,
        reals: tf.Tensor,
        d_preds_on_fakes: tf.Tensor,
    ) -> tf.Tensor:

        # images are from -1 to 1, but the function needs them from 0 to 1
        # lab_fakes = tf.clip_by_value(rgb_to_lab(fakes * 0.5 + 0.5)[:, :, :, 1:], -127, 127) / 255
        # lab_reals = tf.clip_by_value(rgb_to_lab(reals * 0.5 + 0.5)[:, :, :, 1:], -127, 127) / 255

        if self.use_mae:
            # from_mae = tf.reduce_mean(tf.abs(lab_fakes - lab_reals))
            from_mae = tf.reduce_mean(tf.abs(fakes - reals))
        else:
            from_mae = tf.zeros(1)
        from_gan = self.end_loss(tf.ones_like(d_preds_on_fakes), d_preds_on_fakes)
        loss = from_mae * self.weight_mae_loss + from_gan
        return loss, from_mae, from_gan

    def d_loss(
        self,
        preds_on_fakes: tf.Tensor,
        preds_on_reals: tf.Tensor,
    ) -> tf.Tensor:
        loss_on_fakes = self.end_loss(tf.zeros_like(preds_on_fakes), preds_on_fakes)
        loss_on_reals = self.end_loss(tf.ones_like(preds_on_reals), preds_on_reals)
        loss = loss_on_fakes + loss_on_reals
        return loss, loss_on_fakes, loss_on_reals

    def train_step(
        self,
        data: Tuple[tf.Tensor, tf.Tensor],
    ) -> Dict[str, Any]:

        bw, reals = data

        # Imo it backpropagates over the discriminator twice, while I think it
        # could be done using one pass, but discriminator is around 15x smaller
        # than generator so it's not worth optimizing

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fakes = self.g(bw, training=True)
            d_preds_fakes = self.d([bw, fakes], training=True)
            d_preds_reals = self.d([bw, reals], training=True)

            g_loss, g_loss_from_mae, g_loss_from_gan = self.g_loss(fakes, reals, d_preds_fakes)
            d_loss, d_loss_on_fakes, d_loss_on_reals = self.d_loss(d_preds_fakes, d_preds_reals)

            g_grads = g_tape.gradient(g_loss, self.g.trainable_variables)
            d_grads = d_tape.gradient(d_loss, self.d.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_grads, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_grads, self.d.trainable_variables))

        fakes_scores = tf.reduce_mean(tf.math.sigmoid(d_preds_fakes), axis=(1, 2, 3))
        reals_scores = tf.reduce_mean(tf.math.sigmoid(d_preds_reals), axis=(1, 2, 3))

        d_acc_on_fakes = tf.reduce_mean(tf.cast(fakes_scores < 0.5, tf.float32))
        d_acc_on_reals = tf.reduce_mean(tf.cast(reals_scores > 0.5, tf.float32))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "d_avg_pred_on_fakes": tf.reduce_mean(fakes_scores),
            "d_avg_pred_on_reals": tf.reduce_mean(reals_scores),
            "d_accuracy": (d_acc_on_fakes + d_acc_on_reals) / 2,
            "g_loss_from_mae": g_loss_from_mae,
            "g_loss_from_gan": g_loss_from_gan,
            "d_loss_on_fakes": d_loss_on_fakes,
            "d_loss_on_reals": d_loss_on_reals,
            "d_grad_glob_norm": tf.linalg.global_norm(d_grads),
            "g_grad_glob_norm": tf.linalg.global_norm(g_grads),
        }

    def call(self, bw: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        colorized = self.g(bw)
        scores = tf.math.sigmoid(self.d([bw, colorized]))
        scores = tf.reduce_mean(scores, axis=(1, 2, 3))
        return colorized, scores
