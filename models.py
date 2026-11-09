"""VGG16 transfer-learning model: frozen ImageNet backbone + small trainable head."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_vgg16(num_classes, img_size=224, head_units=256, dropout=0.5, weights="imagenet"):
    """Return (model, base).

    The model outputs LOGITS (no softmax), which makes energy-based OOD scores easy later.
    The 'features' layer (head_units-dim) is a handy embedding for Mahalanobis / kNN OOD scores.
    Input must already be VGG16-preprocessed (data.preprocess does that in the pipeline).
    """
    base = keras.applications.VGG16(
        include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base.trainable = False  # PHASE 1 starts with the backbone frozen

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(head_units, activation="relu", name="features")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="vgg16_transfer"), base


def compile_model(model, lr):
    """(Re)compile. Must be called again after changing which layers are trainable."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def unfreeze_blocks(base, blocks=("block5",)):
    """PHASE 2: make only the listed VGG16 conv blocks trainable; keep earlier blocks frozen.
    VGG16 has no BatchNorm layers, so there is no BN-statistics gotcha when unfreezing."""
    base.trainable = True
    for layer in base.layers:
        layer.trainable = any(layer.name.startswith(b) for b in blocks)
    n_train = sum(int(l.trainable) for l in base.layers)
    print(f"Unfroze {blocks}: {n_train}/{len(base.layers)} backbone layers now trainable")


def get_base(model):
    """Find the nested VGG16 backbone inside a model (also works after load_model)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No nested backbone model found")
