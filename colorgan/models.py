from typing import List, Optional, Tuple

import tensorflow as tf
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
from tensorflow.keras.losses import MeanAbsoluteError, Reduction
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import SpectralNormalization


def get_unet_encoder_block(
    inputs: Layer,
    kernel_num: int,
    kernel_size: Tuple[int, int],
    stride: int,
    batch_norm: bool,
    leakiness: float = 0.2,
) -> Sequential:

    block = Sequential()

    block.add(Conv2D(kernel_num, kernel_size, padding="same", strides=stride))
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
    dropout_rate: float = 0.5,
    activation: str = "relu",
) -> Sequential:

    block = Sequential()

    if isinstance(inputs, list):
        inputs = Concatenate(axis=3)(inputs)

    block.add(Conv2DTranspose(kernel_num, kernel_size, padding="same", strides=stride))
    block.add(BatchNormalization())
    if dropout:
        block.add(Dropout(dropout_rate))
    block.add(Activation(activation))

    return block(inputs)


def get_unet_generator(
    input_shape: List[Optional[int]] = [None, None, 3],
    encoder_kernel_nums: List[int] = [64, 128, 256, 512, 512, 512, 512, 512],
    decoder_kernel_nums: List[int] = [512, 512, 512, 512, 256, 128, 64, 3],
    kernel_size: Tuple[int, int] = (4, 4),
    stride: int = 2,
) -> Model:

    if len(decoder_kernel_nums) != len(encoder_kernel_nums):
        raise ValueError("number of encoder and decoder layers must match")

    input_layer = Input(shape=input_shape)
    encoder_blocks: List[Layer] = []
    decoder_blocks: List[Layer] = []

    for kernel_num in encoder_kernel_nums:

        if len(encoder_blocks) == 0:
            inputs = input_layer
            batch_norm = False
        else:
            inputs = encoder_blocks[-1]
            batch_norm = True

        encoder_blocks.append(
            get_unet_encoder_block(inputs, kernel_num, kernel_size, stride, batch_norm)
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
                inputs, kernel_num, kernel_size, stride, dropout, activation=activation
            )
        )

    return Model(inputs=input_layer, outputs=decoder_blocks[-1])


def get_discriminator_block(
    kernel_num: int,
    kernel_size: Tuple[int, ...],
    stride: int,
    batch_norm: bool,
) -> Sequential:

    return Sequential(
        [
            SpectralNormalization(Conv2D(kernel_num, kernel_size, padding="same", strides=stride)),
            BatchNormalization() if batch_norm else Layer(),
            LeakyReLU(alpha=0.2),
        ]
    )


def get_discriminator(
    input_shape: Tuple[Optional[int], ...] = (None, None, 3),
    kernel_nums: Tuple[int, ...] = (64, 64, 128, 128, 256, 256, 512),
    kernel_size: Tuple[int, ...] = (4, 4),
    stride: int = 2,
) -> Model:

    input_bw = Input(input_shape, name="bw")
    input_colorized = Input(input_shape, name="colorized")

    x = Concatenate()([input_bw, input_colorized])

    for i, kernel_num in enumerate(kernel_nums):
        not_first = i != 0
        block = get_discriminator_block(kernel_num, kernel_size, stride, batch_norm=not_first)
        x = block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(inputs=[input_bw, input_colorized], outputs=x)


class ColorGan(Model):
    def __init__(self, g: Model, d: Model, weight_mae_loss: float = 50.0):
        super().__init__()

        self.g = g
        self.d = d

        self.weight_mae_loss = weight_mae_loss

    def compile(self, d_optimizer, g_optimizer, end_loss):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.end_loss = end_loss

    def g_loss(
        self,
        fakes: tf.Tensor,
        origs: tf.Tensor,
        d_preds: tf.Tensor,
    ) -> tf.Tensor:
        from_mae = tf.reduce_mean(tf.abs(origs - fakes))
        from_gan = self.end_loss(tf.ones_like(d_preds), d_preds)
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
    ) -> None:
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

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_loss_from_mae": g_loss_from_mae,
            "g_loss_from_gan": g_loss_from_gan,
            "d_loss_on_fakes": d_loss_on_fakes,
            "d_loss_on_reals": d_loss_on_reals,
        }

    def call(self, bw: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        colorized = self.g(bw)
        scores = self.d([bw, colorized])
        return colorized, scores
