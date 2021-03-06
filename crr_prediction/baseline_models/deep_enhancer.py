"""Model implementing Deep Enhancer."""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from extra_keras_metrics import get_standard_binary_metrics

__all__ = ["deep_enhancers"]


def deep_enhancers(
    window_size: int
) -> Model:
    """Return Deep Enhancers fixed model.

    References
    -------------
    https://www.nature.com/articles/nmeth.2987
    """
    model = Sequential([
        Input((window_size, 4)),
        Reshape((window_size, 4, 1)),
        Conv2D(filters=128, kernel_size=(8, 1), activation="relu"),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(8, 1), activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 1)),
        Conv2D(filters=64, kernel_size=(3, 1), activation="relu"),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 1), activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 1)),
        Flatten(),
        Dense(units=256, activation="relu"),
        Dropout(rate=0.1),
        Dense(units=128, activation="relu"),
        Dropout(rate=0.1),
        Dense(units=1, activation="sigmoid"),
    ], name="DeepEnhancer")

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        # We add all the most common binary metrics
        metrics=get_standard_binary_metrics()
    )

    return model
