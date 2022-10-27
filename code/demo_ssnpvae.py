from code import interact
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sharp
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.callbacks as callbacks

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

X = np.load("../data/mnist/X.npy")
y = np.load("../data/mnist/y.npy")
X_train, _, y_train, _ = train_test_split(X, y, train_size=5000, stratify=y)
# Define the Keras TensorBoard callback.
logdir = "../logs/fit/" + datetime.now().strftime(r"%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
sharp = sharp.ShaRP(
    X.shape[1],
    len(np.unique(y)),
    kl_weight=1.0,
    kl_mu_weight=0.0,
    bottleneck_activation="linear",
    bottleneck_l1=0.0,
    bottleneck_l2=0.5,
)

vmax = 0


def plot_aux(batch, logs):
    if batch == 0:
        vmax = sharp.plot_clusters_and_var(
            sharp, X_train, y_train, f"../training/clusters_{batch}.png"
        )
    else:
        sharp.plot_clusters_and_var(
            sharp, X_train, y_train, f"../training/clusters_{batch}.png", vmax=2 * vmax
        )


plot_clusters_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda batch, logs: sharp.plot_clusters_and_var(
        sharp, X_train, y_train, f"../training/clusters_{batch}.png"
    )
)

sharp.fit(
    X_train,
    y_train,
    callbacks=[tensorboard_callback, plot_clusters_callback],
    epochs=20,
    batch_size=64,
    verbose=True,
)
interact(local=locals())

sharp.plot_clusters_and_var(sharp, X_train, y_train)
