"""Backbone registry: everything that differs between pretrained encoders lives here.

Each entry says how to build the encoder (ImageNet weights, no classification top), how to
preprocess images for it, how to undo that preprocessing for display, and which layers to
unfreeze in phase 2 by default. Everything else (data pipeline, head, two-phase training,
OOD scores) is shared and just asks this registry.

To add an encoder later, write a build function, a preprocess function and a deprocess function,
then add one line to BACKBONES. Nothing else in the project needs to change.

TensorFlow is imported inside the functions so this file can be imported cheaply.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    build: Callable  # (img_size, weights) -> keras Model WITHOUT classification top
    preprocess: Callable  # float32 tensor in [0, 255] -> what the encoder expects (used inside tf.data)
    deprocess: Callable  # numpy array from preprocess -> uint8 RGB in [0, 255], for display only
    default_unfreeze: str  # phase-2 default: 'block5' (name prefixes) or 'last:N' (last N layers)


# ---------------------------------------------------------------------------
# VGG16
# ---------------------------------------------------------------------------
def _vgg16_build(img_size, weights):
    from tensorflow import keras
    return keras.applications.VGG16(
        include_top=False, weights=weights, input_shape=(img_size, img_size, 3))


def _vgg16_preprocess(x):
    """RGB -> BGR and subtract the ImageNet channel means (no rescaling to 0-1)."""
    import tensorflow as tf
    return tf.keras.applications.vgg16.preprocess_input(x)


def _vgg16_deprocess(x):
    mean_bgr = np.array([103.939, 116.779, 123.68])
    return np.clip((x + mean_bgr)[..., ::-1], 0, 255).astype("uint8")


BACKBONES = {
    "vgg16": BackboneSpec("vgg16", _vgg16_build, _vgg16_preprocess, _vgg16_deprocess,
                          default_unfreeze="block5"),
}


def get(name):
    """Look up a backbone by name, with a helpful error."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONES)}")
    return BACKBONES[name]
