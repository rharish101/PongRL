"""DQN model."""
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from utils import IMG_SIZE, STATE_FRAMES


def get_model(output_dims: int) -> Model:
    """Get the DQN model."""
    model = Sequential()
    model.add(
        Conv2D(
            16,
            8,
            strides=4,
            activation="relu",
            kernel_initializer=VarianceScaling(2),
            input_shape=(*IMG_SIZE, STATE_FRAMES),
        )
    )
    model.add(
        Conv2D(
            32,
            4,
            strides=2,
            activation="relu",
            kernel_initializer=VarianceScaling(2),
        )
    )
    model.add(Flatten())
    model.add(
        Dense(256, activation="relu", kernel_initializer=VarianceScaling(2))
    )
    model.add(Dense(output_dims, kernel_initializer=VarianceScaling(2)))
    return model
