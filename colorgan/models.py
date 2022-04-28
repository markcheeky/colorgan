from typing import List, Optional, Tuple

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
            Conv2D(kernel_num, kernel_size, padding="same", strides=stride),
            BatchNormalization() if batch_norm else Layer(),
            LeakyReLU(alpha=0.2),
        ]
    )


def get_discriminator(
    input_shape: Tuple[Optional[int], ...] = (None, None, 3),
    kernel_nums: Tuple[int, ...] = (64, 128, 256, 512),
    kernel_size: Tuple[int, ...] = (4, 4),
    stride: int = 2,
) -> Model:

    layers = Sequential()
    layers.add(Input(input_shape))

    for i, kernel_num in enumerate(kernel_nums):
        not_first = i != 0
        block = get_discriminator_block(kernel_num, kernel_size, stride, batch_norm=not_first)
        layers.add(block)

    layers.add(GlobalAveragePooling2D())
    layers.add(Dense(1, activation="sigmoid"))

    return layers
