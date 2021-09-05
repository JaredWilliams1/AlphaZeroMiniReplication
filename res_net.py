from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate
from tensorflow import shape
from tensorflow import int64
from tensorflow import concat
from tensorflow import stack
from tensorflow import RaggedTensor

import sys

class Connect4Model:
    def __init__(self):
        self.model = self.build()
        #self.loss_func = self.model_loss()

    #data_format = "channels_first"

    def build(self):
        in_x = x = Input((6,7,3))
        # (batch, channels, height, width)
        x = Conv2D(filters=128, kernel_size=8, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(10):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_last", kernel_regularizer=l2(1e-4))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        policy_out1 = Dense(7, kernel_regularizer=l2(1e-4), activation="softmax", name="policy_out1")(x)   # how many outputs

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_last", kernel_regularizer=l2(1e-4))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=l2(1e-4), activation="relu")(x)
        value_out1 = Dense(1, kernel_regularizer=l2(1e-4), activation="tanh", name="value_out1")(x)

        output = self._build_output_layer(value_out1, policy_out1)

        return Model(in_x, outputs=output, name="connect4_model")

    def _build_residual_block(self, x):
        in_x = x
        x = Conv2D(filters=128, kernel_size=3, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=128, kernel_size=3, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    def _build_output_layer(self, value_out1, policy_out1):
        value_shape = shape(value_out1, out_type=int64)[0]
        policy_shape = shape(policy_out1, out_type=int64)[0]
        values = concat([value_out1, policy_out1], axis=0)
        lens = stack([value_shape, policy_shape])
        return RaggedTensor.from_row_lengths(values, lens)

    #def compile(self):
    #    self.model.compile(optimizer=tf.keras.optimizers.Nadam(),
    #              loss=Connect4Model.model_loss,
    #              metrics=['kullback_leibler_divergence', 'mean_squared_error'])

    def save_model(self, model_path, weight_path):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weight_path)
        print("Saved model to disk")
        # SAVE NN
