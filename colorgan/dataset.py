import concurrent.futures
import io
import os
from typing import Collection, Generator, Optional, Tuple

import numpy as np
import PIL.Image
import requests
import tensorflow as tf
from requests import Response
from tensorflow.data import Dataset
from utils import IMG_EXTENSIONS, chunkify


def response_to_img(response: Response) -> PIL.Image.Image:

    if response.status_code != 200:
        err = f"Response from {response.url} has code {response.status_code}"
        raise ValueError(err, response)

    img = PIL.Image.open(io.BytesIO(response.content))
    img_array = np.asarray(img)

    if len(img_array.shape) != 3:
        err = f"image {response.url} with shape {img_array.shape} has invalid dimensions."
        raise ValueError(err, response, img)

    if img_array.shape[-1] not in (3, 4):
        err = f"image {response.url} with shape {img_array.shape} has wrong number of channels."
        raise ValueError(err, response, img)

    return img


def get_img(
    image_id: str,
    url: str,
    max_img_size: Tuple[int, int],
    use_cache: bool,
    cache_dir: Optional[str] = None,
    timeout: int = 10,
) -> Optional[np.ndarray]:

    possible_paths = [f"{cache_dir}/{image_id}.{ext}" for ext in IMG_EXTENSIONS]
    existings = list(filter(os.path.exists, possible_paths))

    if len(existings) != 0:
        img = PIL.Image.open(existings[0])
    else:
        try:
            response = requests.get(url, timeout=timeout)
            img = response_to_img(response)
            if use_cache:
                img.save(f"{cache_dir}/{image_id}.{img.format.lower()}")
        except Exception as e:
            print(e)
            return None

    img.thumbnail(max_img_size)
    array = np.asarray(img)[:, :, :3]
    return array


def download_imgs(
    image_ids: Collection[str],
    urls: Collection[str],
    workers: int,
    max_img_size: Tuple[int, int],
    use_cache: bool = False,
    cache_dir: Optional[str] = None,
) -> Generator[np.ndarray, None, None]:
    def getter(img_id: str, url: str) -> Optional[np.ndarray]:
        return get_img(img_id, url, max_img_size, use_cache, cache_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in chunkify(zip(image_ids, urls), workers):
            futures = [pool.submit(getter, *args) for args in chunk]
            for img in concurrent.futures.as_completed(futures):
                if img.result() is not None:
                    yield img.result()


def get_xy_for_G(img: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.image.rgb_to_grayscale(img)
    x = tf.repeat(x, 3, axis=-1)
    x = tf.cast(x, tf.float32) / 127.5 - 1
    y = tf.cast(img, tf.float32) / 127.5 - 1
    return x, y


def preprocess(inputs: np.ndarray) -> np.ndarray:
    return inputs.astype(np.float32) / 127.5 - 1


def postprocess(outputs: np.ndarray) -> np.ndarray:
    return ((outputs + 1) * 127.5).astype(np.uint8)


def extract_patches(
    img: tf.Tensor,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Dataset:

    img = tf.expand_dims(img, 0)
    patches = tf.image.extract_patches(
        images=img,
        sizes=[1, *patch_size, 1],
        strides=[1, *stride, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patches = tf.reshape(patches, (-1, *patch_size, 3))
    return Dataset.from_tensor_slices(patches)


def url_dataset_for_G(
    image_ids: Collection[str],
    urls: Collection[str],
    augment: bool = False,
    use_cache: bool = False,
    cache_dir: Optional[str] = None,
    patch_size: Tuple[int, int] = (512, 512),
    stride: Tuple[int, int] = (400, 400),
    max_img_size: Tuple[int, int] = (1350, 1350),
    download_workers: int = 25,
    prefetch: int = 0,
    repeats: int = 1,
    repeat_cycle: int = 5000,
    seed: int = 42,
) -> Dataset:

    tf.random.set_seed(seed)
    signature = tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)

    ds = Dataset.from_generator(
        lambda: download_imgs(
            image_ids, urls, download_workers, max_img_size, use_cache, cache_dir
        ),
        output_signature=signature,
    )

    ds = ds.prefetch(prefetch)
    ds = ds.flat_map(lambda img: extract_patches(img, patch_size, stride))
    ds = ds.interleave(lambda img: Dataset.from_tensors(img).repeat(repeats), repeat_cycle)

    if augment:
        ds = ds.map(lambda img: tf.image.random_flip_left_right(img))
        ds = ds.map(lambda img: tf.image.random_brightness(img, max_delta=0.15))
        ds = ds.map(lambda img: tf.image.random_saturation(img, lower=0.8, upper=1.2))

    ds = ds.map(get_xy_for_G)

    return ds
