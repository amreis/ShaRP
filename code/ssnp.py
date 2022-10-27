import gc
import os

import numpy as np
import tensorflow as tf
from sklearn.neighbors import DistanceMetric, NearestNeighbors
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import datasets as kdatasets
from tensorflow.keras import losses, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical

# os.environ["TF_DETERMINISTIC_OPS"] = "1"

from keras.losses import binary_crossentropy as k_bce


def vae_reconstruction_loss(*args, **kwargs):
    return k_bce(*args, **kwargs)


class SSNP:
    def __init__(
        self,
        init_labels="precomputed",
        epochs=100,
        input_l1=0.0,
        input_l2=0.0,
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
        verbose=1,
        opt="adam",
        bottleneck_activation="tanh",
        act="relu",
        init="glorot_uniform",
        bias=0.0001,
        patience=3,
        min_delta=0.01,
    ):
        self.init_labels = init_labels
        self.epochs = epochs
        self.verbose = verbose
        self.opt = opt
        self.act = act
        self.init = init
        self.bias = bias
        self.input_l1 = input_l1
        self.input_l2 = input_l2
        self.bottleneck_l1 = bottleneck_l1
        self.bottleneck_l2 = bottleneck_l2
        self.bottleneck_activation = bottleneck_activation
        self.patience = patience
        self.min_delta = min_delta

        self.label_bin = LabelBinarizer()

        self.fwd = None
        self.inv = None

        tf.random.set_seed(42)

        self.is_fitted = False
        K.clear_session()

    def fit(self, X, y=None):
        if y is None and self.init_labels == "precomputed":
            raise Exception("Must provide labels when using init_labels = precomputed")

        if y is None:
            y = self.init_labels.fit_predict(X)

        self.label_bin.fit(y)

        main_input = Input(shape=(X.shape[1],), name="main_input")
        x = Dense(
            512,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(main_input)
        x = Dense(
            128,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)
        x = Dense(
            32,
            activation=self.act,
            activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)
        encoded = Dense(
            2,
            activation=self.bottleneck_activation,
            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)

        x = Dense(
            32,
            activation=self.act,
            kernel_initializer=self.init,
            name="enc1",
            bias_initializer=Constant(self.bias),
        )(encoded)
        x = Dense(
            128,
            activation=self.act,
            kernel_initializer=self.init,
            name="enc2",
            bias_initializer=Constant(self.bias),
        )(x)
        x = Dense(
            512,
            activation=self.act,
            kernel_initializer=self.init,
            name="enc3",
            bias_initializer=Constant(self.bias),
        )(x)

        n_classes = len(np.unique(y))

        if n_classes == 2:
            n_units = 1
        else:
            n_units = n_classes

        main_output = Dense(
            n_units,
            activation="softmax",
            name="main_output",
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)

        decoder_output = Dense(
            X.shape[1],
            activation="sigmoid",
            name="decoder_output",
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)

        model = Model(inputs=main_input, outputs=[main_output, decoder_output])

        model.compile(
            optimizer=self.opt,
            loss={
                "main_output": "categorical_crossentropy",
                "decoder_output": "binary_crossentropy",
            },
            metrics=["accuracy"],
        )

        if self.patience > 0:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    min_delta=self.min_delta,
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            ]
        else:
            callbacks = []

        hist = model.fit(
            X,
            [self.label_bin.transform(y), X],
            batch_size=256,  # TODO: (UNCHANGE)
            epochs=self.epochs,
            shuffle=True,
            verbose=self.verbose,
            validation_split=0.05,
            callbacks=callbacks,
        )

        encoded_input = Input(shape=(2,))
        l = model.get_layer("enc1")(encoded_input)
        l = model.get_layer("enc2")(l)
        l = model.get_layer("enc3")(l)
        decoder_layer = model.get_layer("decoder_output")(l)
        classifier_layer = model.get_layer("main_output")(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded)
        self.clustering = Model(inputs=main_input, outputs=main_output)
        self.latent_clustering = Model(inputs=encoded_input, outputs=classifier_layer)
        self.is_fitted = True

        return hist

    def transform(self, X):
        if self._is_fit():
            return self.fwd.predict(X)

    def inverse_transform(self, X_2d):
        if self._is_fit():
            return self.inv.predict(X_2d)

    def predict(self, X):
        if self._is_fit():
            y_pred = self.clustering.predict(X)
            return self.label_bin.inverse_transform(y_pred)

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception("Model not trained. Call fit() before calling transform()")
