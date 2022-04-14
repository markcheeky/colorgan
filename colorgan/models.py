from typing import Optional, Tuple

from tensorflow.keras.layers import (
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
    ReLU,
)
from tensorflow.keras.models import Model, Sequential


def get_unet_generator(
    input_shape: Tuple[Optional[int]] = (None, None, 3),
    encoder_kernel_nums: Tuple[int] = (64, 128, 256, 512, 512, 512, 512, 512),
    decoder_kernel_nums: Optional[Tuple[int]] = None,
    kernel_size: Tuple[int] = (4, 4),
    stride: int = 2,
) -> Model:

    if decoder_kernel_nums is None:
        decoder_kernel_nums = encoder_kernel_nums[::-1]

    if len(decoder_kernel_nums) != len(encoder_kernel_nums):
        raise ValueError("number of encoder and decoder layers must match")

    input_layer = Input(shape=input_shape)
    encoder_blocks = []
    decoder_blocks = []

    for kernel_num in encoder_kernel_nums:
        if len(encoder_blocks) == 0:
            inputs = input_layer
        else:
            inputs = encoder_blocks[-1]

        block = Sequential(
            [
                Conv2D(kernel_num, kernel_size, padding="same", strides=stride),
                BatchNormalization() if len(encoder_blocks) != 0 else Layer(),
                LeakyReLU(alpha=0.2),
            ]
        )

        encoder_blocks.append(block(inputs))

    for skip_src, kernel_num in zip(encoder_blocks[::-1], decoder_kernel_nums):
        if len(decoder_blocks) == 0:
            inputs = skip_src
        else:
            inputs = Concatenate(axis=3)([skip_src, decoder_blocks[-1]])

        block = Sequential(
            [
                Conv2DTranspose(kernel_num, kernel_size, padding="same", strides=2),
                BatchNormalization(),
                Dropout(0.5) if len(decoder_blocks) < 3 else Layer(),
                ReLU(),
            ]
        )

        decoder_blocks.append(block(inputs))

    output_layer = Conv2DTranspose(3, kernel_size, activation="tanh")(decoder_blocks[-1])
    return Model(inputs=input_layer, outputs=output_layer)


def get_discriminator_block(
    kernel_num: int,
    kernel_size: Tuple[int],
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
    input_shape: Tuple[Optional[int]] = (None, None, 3),
    kernel_nums: Tuple[int] = (64, 128, 256, 512),
    kernel_size: Tuple[int] = (4, 4),
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
