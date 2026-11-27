"""Transfer-learning model: frozen pretrained encoder (see backbones.py) + small trainable head."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import backbones


def build_model(backbone, num_classes, img_size=224, head_units=256, dropout=0.5, weights="imagenet"):
    """Return (model, base).

    backbone: a name from backbones.BACKBONES (e.g. "vgg16").
    The model outputs LOGITS (no softmax); softmax happens inside the loss during training and in
    the analysis scripts afterwards. The 'features' layer (head_units-dim) is the embedding used
    by the class-distance OOD score.
    Input must already be preprocessed for this backbone (data.make_dataset does that).
    """
    spec = backbones.get(backbone)
    base = spec.build(img_size, weights)
    base.trainable = False  # PHASE 1 starts with the encoder frozen

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(head_units, activation="relu", name="features")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


def build_vgg16(num_classes, img_size=224, head_units=256, dropout=0.5, weights="imagenet"):
    """Kept so older code keeps working; same as build_model('vgg16', ...)."""
    return build_model("vgg16", num_classes, img_size, head_units, dropout, weights)


def compile_model(model, lr):
    """(Re)compile. Must be called again after changing which layers are trainable."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def unfreeze(base, spec):
    """PHASE 2: choose which encoder layers become trainable; everything else stays frozen.

    spec = 'block5' or 'block5,block4'  -> layers whose names start with these prefixes (VGG16 blocks)
    spec = 'last:30'                    -> the last 30 layers (BatchNorm layers are kept frozen,
                                           the usual safe choice when fine-tuning on small data)
    """
    base.trainable = True
    if spec.startswith("last:"):
        n = int(spec.split(":", 1)[1])
        cutoff = len(base.layers) - n
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    else:
        prefixes = tuple(s.strip() for s in spec.split(",") if s.strip())
        for layer in base.layers:
            layer.trainable = layer.name.startswith(prefixes)
    n_train = sum(int(l.trainable) for l in base.layers)
    print(f"Unfroze '{spec}': {n_train}/{len(base.layers)} encoder layers now trainable")


def unfreeze_blocks(base, blocks=("block5",)):
    """Kept so older code keeps working; same as unfreeze(base, 'block5,block4')."""
    unfreeze(base, ",".join(blocks))


def get_base(model):
    """Find the nested encoder inside a model (also works after load_model)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No nested backbone model found")
