import os
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pylab
import tensorflow as tf
from sampling_layers import get_layer_builder
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras as tfk
from tensorflow.keras import callbacks
from tensorflow.keras import layers as tfkl
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import Constant


class Encoder(tfkl.Layer):
    def __init__(
        self,
        *,
        act="relu",
        init="glorot_uniform",
        bias=1e-4,
        input_l1=0.0,
        input_l2=0.0,
        name="encoder",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.act = act
        self.init = init
        self.bias = bias
        self.input_l1 = input_l1
        self.input_l2 = input_l2

        self.enc1 = tfkl.Dense(
            512,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )
        self.enc2 = tfkl.Dense(
            128,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )
        self.enc3 = tfkl.Dense(
            32,
            activation=self.act,
            activity_regularizer=regularizers.l1_l2(self.input_l1, self.input_l2),
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )

    def get_config(self):
        return {
            "act": self.act,
            "init": self.init,
            "bias": self.bias,
            "input_l1": self.input_l1,
            "input_l2": self.input_l2,
        }

    def call(self, inputs):
        x = self.enc1(inputs)
        x = self.enc2(x)
        x = self.enc3(x)
        return x


class Decoder(tfkl.Layer):
    def __init__(self, act="relu", init="glorot_uniform", bias=1e-4, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.act = act
        self.init = init
        self.bias = bias
        self.dec1 = tfkl.Dense(
            32,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )
        self.dec2 = tfkl.Dense(
            128,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )
        self.dec3 = tfkl.Dense(
            512,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )

    def get_config(self):
        return {"act": self.act, "init": self.init, "bias": self.bias}

    def call(self, inputs):
        x = self.dec1(inputs)
        x = self.dec2(x)
        x = self.dec3(x)
        return x


class ShaRP(tfk.Model):
    def __init__(
        self,
        original_dim,
        n_classes: int,
        variational_layer: Union[str, Callable[..., tfkl.Layer]],
        variational_layer_kwargs: dict = dict(),
        act="relu",
        opt="adam",
        bottleneck_activation="tanh",
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
        var_leaky_relu_alpha=-0.01,
        init="glorot_uniform",
        bias=1e-4,
        name="autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.n_classes = n_classes
        self.latent_dim = 2
        self.variational_layer = variational_layer
        self.act = act
        self.opt = opt
        self.init = init
        self.bias = bias
        self.bottleneck_activation = bottleneck_activation
        self.label_bin = LabelBinarizer()
        self.bottleneck_l1 = bottleneck_l1
        self.bottleneck_l2 = bottleneck_l2
        self.var_leaky_relu_alpha = var_leaky_relu_alpha
        self.variational_layer_kwargs = variational_layer_kwargs | {
            "act": self.bottleneck_activation,
            "init": self.init,
            "bias": self.bias,
            "l1_reg": self.bottleneck_l1,
            "l2_reg": self.bottleneck_l2,
        }

        self.encoder = Encoder(act=act, init=init, bias=bias)
        self.variational = self._build_variational_layer(self.variational_layer_kwargs)
        self.decoder = Decoder(act=act, init=init, bias=bias)

        self.reconstructor = tfkl.Dense(
            original_dim,
            name="reconstruction",
            activation="sigmoid",
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )
        self.classifier = tfkl.Dense(
            1 if n_classes == 2 else n_classes,
            activation="softmax",
            name="main_output",
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )

        self.compile(
            optimizer=self.opt,
            loss=["categorical_crossentropy", "binary_crossentropy"],
            loss_weights=[1.0, 1.0],  # TODO Changed.
            metrics=["accuracy"],
        )

        self.main_input = tfk.Input(self.original_dim)
        mu, log_var, encoded = self.variational(self.encoder(self.main_input))
        self.fwd = tfk.Model(inputs=self.main_input, outputs=encoded)

        self.encoded_input = tfk.Input(2)
        rev = self.decoder(self.encoded_input)
        classes = self.classifier(rev)
        rev = self.reconstructor(rev)
        self.inv = tfk.Model(inputs=self.encoded_input, outputs=rev)

        self.log_var_model = tfk.Model(inputs=self.main_input, outputs=log_var)
        self.mu_model = tfk.Model(inputs=self.main_input, outputs=mu)

        self.class_model = tfk.Model(inputs=self.encoded_input, outputs=classes)

    def _build_variational_layer(self, variational_kwargs):
        if isinstance(self.variational_layer, str):
            return get_layer_builder(self.variational_layer)(self.latent_dim, **variational_kwargs)
        else:
            return self.variational_layer(self.latent_dim, **variational_kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "original_dim": self.original_dim,
                "n_classes": self.n_classes,
                "variational_layer": self.variational_layer,
                "variational_layer_kwargs": self.variational_layer_kwargs,
                "act": self.act,
                "opt": self.opt,
                "bottleneck_activation": self.bottleneck_activation,
                "bottleneck_l1": self.bottleneck_l1,
                "bottleneck_l2": self.bottleneck_l2,
                "var_leaky_relu_alpha": self.var_leaky_relu_alpha,
                "init": self.init,
                "bias": self.bias,
                "name": self.name,
            }
        )
        return config

    def save_weights(self, export_path: str, *args, **kwargs):
        # Route `save_weights` to specific models.
        self.fwd.save_weights(os.path.join(export_path, "fwd"), *args, **kwargs)
        self.inv.save_weights(os.path.join(export_path, "inv"), *args, **kwargs)
        self.log_var_model.save_weights(os.path.join(export_path, "logvar"), *args, **kwargs)
        self.mu_model.save_weights(os.path.join(export_path, "mu"), *args, **kwargs)

    def load_weights(self, export_path: str, *args, **kwargs):
        # Same for `load_weights`
        self.fwd.load_weights(os.path.join(export_path, "fwd"), *args, **kwargs)
        self.inv.load_weights(os.path.join(export_path, "inv"), *args, **kwargs)
        self.log_var_model.load_weights(os.path.join(export_path, "logvar"), *args, **kwargs)
        self.mu_model.load_weights(os.path.join(export_path, "mu"), *args, **kwargs)

    def fit(
        self,
        X_train: Union[tf.data.Dataset, np.ndarray],
        y_train: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> tfk.callbacks.History:
        if isinstance(X_train, tf.data.Dataset):
            return super().fit(X_train, *args, **kwargs)
        else:
            return super().fit(
                X_train, [self.label_bin.fit_transform(y_train), X_train], *args, **kwargs
            )

    def call(self, inputs, training):
        encoded = self.encoder(inputs)
        z_mean, z_log_var, z = self.variational(encoded)
        decoded = self.decoder(z)

        reconstructed = self.reconstructor(decoded)
        main_output = self.classifier(decoded)
        # if training:
        #     # tf.maximum(z_log_var, 0.0)
        #     loss_log_var = tfnn.leaky_relu(z_log_var, self.var_leaky_relu_alpha)
        #     kl_loss = -0.5 * tf.reduce_mean(
        #         loss_log_var - self.kl_mu_weight * tf.square(z_mean) - tf.exp(loss_log_var) + 1
        #     )
        #     self.add_loss(self.kl_weight * kl_loss)
        return main_output, reconstructed

    def encode(self, inputs):
        return self.fwd.predict(inputs)

    def transform(self, inputs):
        return self.fwd.predict(inputs)

    def inverse_transform(self, inputs):
        return self.inv.predict(inputs)

    def log_var_embedding(self, inputs):
        return self.log_var_model.predict(inputs)

    def mu_embedding(self, inputs):
        return self.mu_model.predict(inputs)


def plot_clusters_and_var(
    model: ShaRP,
    X: np.ndarray,
    y: np.ndarray,
    fname: Optional[str] = None,
    figsize=(34, 20),
    vmax=4.0,
):
    if figsize is not None and fname is not None:
        plt.figure(figsize=figsize)
    X_proj = model.transform(X)
    plt.subplot(1, 2, 1)
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)

    xmin, xmax = X_proj[:, 0].min(), X_proj[:, 0].max()
    ymin, ymax = X_proj[:, 1].min(), X_proj[:, 1].max()
    x_coords = np.linspace(xmin, xmax, 250)
    y_coords = np.linspace(ymin, ymax, 250)
    xx, yy = np.meshgrid(x_coords, y_coords)

    nd_coords = model.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    log_vars = model.log_var_embedding(nd_coords)
    variances = np.exp(log_vars)
    if vmax is None:
        vmax = variances.max()
    del nd_coords
    plt.subplot(1, 2, 2)
    plt.scatter(
        xx.ravel(),
        yy.ravel(),
        c=variances.sum(axis=1),
        cmap="gray",
        vmin=0.0,
        vmax=vmax,
    )

    del xx, yy
    if fname is not None:
        plt.savefig(fname)
        plt.close("all")
    else:
        plt.show()
    return variances.max()


class ProjectDataCallback(callbacks.Callback):
    def __init__(self, X: np.ndarray, y: np.ndarray, base_fname: str) -> None:
        super().__init__()

        self.X = X.copy()
        self.y = y.copy()
        self.base_fname = base_fname

    def on_epoch_end(self, epoch, logs=None):
        X_proj = self.model.transform(self.X)
        fig, ax = plt.subplots()
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        cmap = pylab.cm.get_cmap("tab10")
        for lab in range(10):
            ax.scatter(
                X_proj[self.y == lab, 0],
                X_proj[self.y == lab, 1],
                c=[cmap(lab) for _ in range(len(X_proj[self.y == lab]))],
                marker=f"${lab}$",
                # marker=".",
                alpha=0.3,
                s=60,
            )
        fig.savefig(f"{self.base_fname}_{epoch}.png")
        plt.close(fig)

    def on_train_end(self, logs=None):
        import glob

        from PIL import Image

        imgs = (
            Image.open(f)
            for f in sorted(
                glob.glob(f"{self.base_fname}_*.png"),
                key=lambda fname: int(fname.split("_")[-1].removesuffix(".png")),
            )
        )
        img = next(imgs)
        img.save(
            fp=f"{self.base_fname}.gif",
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )


if __name__ == "__main__":
    if "code" not in os.getcwd():
        os.chdir("./code")

    X = np.load("../data/mnist/X.npy")
    y = np.load("../data/mnist/y.npy")

    original_dim = X.shape[1]
    sharp = ShaRP(
        original_dim,
        len(np.unique(y)),
        "diagonal_normal",
        variational_layer_kwargs=dict(kl_weight=0.1),
        var_leaky_relu_alpha=-0.0001,
        bottleneck_activation="linear",
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, stratify=y)

    label_bin = LabelBinarizer()
    label_bin.fit(y_train)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, (label_bin.transform(y_train), X_train))
    ).shuffle(buffer_size=1024)
    val_dataset = train_dataset.take(500).batch(32)
    train_dataset = train_dataset.skip(500).take(30000).batch(64)

    epochs = 20

    sharp.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=True,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="loss",
                min_delta=1e-3,
                patience=20,
                restore_best_weights=True,
                mode="min",
            ),
            callbacks.TerminateOnNaN(),
        ],
    )

    X_i = sharp.transform(X_train)
    x_coords, y_coords = (
        np.linspace(X_i[:, 0].min(), X_i[:, 0].max(), 250),
        np.linspace(X_i[:, 1].min(), X_i[:, 1].max(), 250),
    )
    xx, yy = np.meshgrid(x_coords, y_coords)

    import var_metrics

    sample_points = np.c_[xx.ravel(), yy.ravel()]
    gradients = var_metrics.gradient_norm(sharp, sample_points).numpy()
    diffs = var_metrics.finite_differences(sharp, sample_points, eps=1e-2).numpy().sum(axis=1)
    variances = np.exp(sharp.log_var_embedding(sharp.inverse_transform(sample_points)) * 0.5).sum(
        axis=1
    )

    plt.subplot(2, 2, 1)
    plt.title("Projected Data")
    cmap = pylab.cm.get_cmap("tab10")
    for lab in range(10):
        plt.gca().scatter(
            X_i[y_train == lab, 0],
            X_i[y_train == lab, 1],
            c=[cmap(lab) for _ in range(len(X_i[y_train == lab]))],
            marker=f"${lab}$",
            # marker=".",
            alpha=0.3,
            s=60,
        )
    plt.gca().set_aspect("equal")

    plt.subplot(2, 2, 2)
    plt.title(f"Gradient Norm (min={gradients.min():.3g}, max={gradients.max():.3g})")
    plt.scatter(xx.ravel(), yy.ravel(), c=(gradients), cmap="gray", marker=",")
    plt.gca().set_aspect("equal")

    plt.subplot(2, 2, 3)
    plt.title(f"Finite Difference Gradient (min={diffs.min():.3g}, max={diffs.max():.3g})")
    plt.scatter(xx.ravel(), yy.ravel(), c=(diffs), cmap="gray", marker=",")
    plt.gca().set_aspect("equal")

    plt.subplot(2, 2, 4)
    plt.title(f"Learned Variance Model (min={variances.min():.3g}, max={variances.max():.3g})")
    plt.scatter(xx.ravel(), yy.ravel(), c=variances, cmap="gray", marker=",")
    plt.gca().set_aspect("equal")

    del xx, yy, x_coords, y_coords, sample_points
    plt.show()
