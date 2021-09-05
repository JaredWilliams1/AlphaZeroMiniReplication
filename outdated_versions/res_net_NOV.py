from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


class Connect4Model:
    def __init__(self):
        self.model = None  # type: Model

    def build(self):
        in_x = x = Input((6,7,3))
        # (batch, channels, height, width)
        x = Conv2D(filters=128, kernel_size=8, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(10):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(1e-4))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        policy_out1 = Dense(7, kernel_regularizer=l2(1e-4), activation="softmax", name="policy_out1")(x)   # how many outputs

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(1e-4))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=l2(1e-4), activation="relu")(x)
        value_out1 = Dense(1, kernel_regularizer=l2(1e-4), activation="tanh", name="value_out1")(x)

        self.model = Model(in_x, outputs=[policy_out1, value_out1], name="connect4_model")

    def _build_residual_block(self, x):
        in_x = x
        x = Conv2D(filters=128, kernel_size=3, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=128, kernel_size=3, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    def save_model(self, model_path, weight_path):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weight_path)
        print("Saved model to disk")
        # SAVE NN
