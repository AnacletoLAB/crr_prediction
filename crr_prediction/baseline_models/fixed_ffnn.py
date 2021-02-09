"""Model implementing Deep Enhancer."""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from extra_keras_metrics import get_standard_binary_metrics

__all__ = ["fixed_ffnn"]


def fixed_ffnn(
    features_number: int
) -> Model:
    """Return fixed FFNN model."""
    model = Sequential([
        Input((features_number,)),
        Dense(units=16, activation="relu"),
        Dense(units=4, activation="relu"),
        Dense(units=2, activation="relu"),
        Dense(units=1, activation="sigmoid"),
    ], name="FixedFFNN")

    model.compile(
        optimizer=SGD(learning_rate=0.5, decay=0.1),
        loss="binary_crossentropy",
        # We add all the most common binary metrics
        metrics=get_standard_binary_metrics()
    )

    return model
