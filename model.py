"""DQN model."""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten


def get_model(input_shape, output_dims):
    """Get the DQN model."""
    model = Sequential()
    model.add(
        Conv2D(16, 8, strides=4, activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(32, 4, strides=2, activation="relu"))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(output_dims))
    return model
