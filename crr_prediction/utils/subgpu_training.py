"""Methods relative to subgpu training."""
import tensorflow as tf


def enable_subgpu_training():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
