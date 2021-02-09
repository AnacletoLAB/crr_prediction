"""Model implementing Deep Enhancer."""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, AveragePooling2D
from extra_keras_metrics import get_standard_binary_metrics

__all__ = ["fixed_cnn"]


def fixed_cnn(
    window_size: int
) -> Model:
    """Return fixed CNN model."""
    model = Sequential([
        Input((window_size, 4)),
        Reshape((window_size, 4, 1)),
        Conv2D(filters=64, kernel_size=(8, 1), activation="relu"),
        MaxPool2D(pool_size=(2, 1)),
        Conv2D(filters=128, kernel_size=(3, 1), activation="relu"),
        MaxPool2D(pool_size=(2, 1)),
        Conv2D(filters=128, kernel_size=(3, 1), activation="relu"),
        AveragePooling2D(pool_size=(2, 1)),
        Flatten(),
        Dropout(rate=0.5),
        Dense(units=10, activation="relu"),
        Dropout(rate=0.1),
        Dense(units=1, activation="sigmoid"),
    ], name="FixedCNN")

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        # We add all the most common binary metrics
        metrics=get_standard_binary_metrics()
    )

    return model
