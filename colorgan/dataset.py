import io
import logging
from typing import Collection, Iterable, List, Optional, Tuple

import grequests
import numpy as np
import PIL
import requests
import skimage.color
import tensorflow as tf


class ImgUrlDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        urls: Collection,
        batch_size: int = 32,
        max_width: int = 1080,
        max_height: int = 1080,
    ) -> None:

        self.urls = urls
        self.batch_size = batch_size
        self.max_width = max_width
        self.max_height = max_height

    def __len__(self) -> int:
        return len(self.urls) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        start = idx * self.batch_size
        end = start + self.batch_size
        rows = self.urls[start:end]
        imgs = self.fetch_imgs(rows)
        imgs = [self.resize_to_fit(img, self.max_height, self.max_width) for img in imgs]
        imgs = [
            self.pad_to_size(
                img, self.max_height, self.max_width, 0, np.random.default_rng(idx + i)
            )
            for i, img in enumerate(imgs)
        ]
        img_bw = [np.expand_dims(skimage.color.rgb2gray(img), -1) for img in imgs]
        x = (np.stack(img_bw).astype(np.float32) / 255.0 - 0.5) * 2
        y = (np.stack(imgs).astype(np.float32) / 255.0 - 0.5) * 2
        # todo return masks
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

    @staticmethod
    def resize_to_fit(
        img: np.ndarray,
        max_height: int,
        max_width: int,
    ) -> np.ndarray:
        pil_img = PIL.Image.fromarray(img)
        pil_img.thumbnail((max_width, max_height))
        return np.asarray(pil_img)

    @staticmethod
    def pad_to_size(
        img: np.ndarray,
        height: int,
        width: int,
        bg_val: int,
        random_gen: np.random.Generator,
    ) -> np.ndarray:

        t_padding = random_gen.integers(0, height - img.shape[0] + 1)
        b_padding = height - img.shape[0] - t_padding
        vertical = (t_padding, b_padding)

        l_padding = random_gen.integers(0, width - img.shape[1] + 1)
        r_padding = width - img.shape[1] - l_padding
        horizontal = (l_padding, r_padding)

        return np.pad(img, [vertical, horizontal, (0, 0)], constant_values=bg_val)

    @staticmethod
    def response_to_img_safe(response: requests.Response) -> Optional[np.ndarray]:
        try:
            img = skimage.io.imread(io.BytesIO(response.content))
            if len(img.shape) != 3:
                raise ValueError(
                    f"image {response.url} with shape {img.shape} has invalid dimensions."
                )
            if img.shape[-1] not in (3, 4):
                raise ValueError(
                    f"image {response.url} with shape {img.shape} has wrong number of channels."
                )
            img = img[:, :, :3]  # dropping potential alpha channel
            return img
        except Exception as e:
            logging.warning(response.url, e)
        return None

    @staticmethod
    def fetch_imgs(urls: Iterable[str]) -> List[np.ndarray]:
        reqs = (grequests.get(url, timeout=10) for url in urls)
        responses = grequests.map(reqs)
        imgs = list(map(ImgUrlDataset.response_to_img_safe, responses))
        return [img for img in imgs if img is not None]
