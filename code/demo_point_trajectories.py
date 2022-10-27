import os
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from umap import UMAP

import sharp
import ssnp


if __name__ == "__main__":
    epochs = 100
    verbose = False

    data_dir = "../data"
    data_dirs = ["mnist", "fashionmnist", "har", "reuters"]

    dataset_name = d = data_dirs[3]

    X: np.ndarray = np.load(os.path.join(data_dir, d, "X.npy"))
    y: np.ndarray = np.load(os.path.join(data_dir, d, "y.npy"))
    n_samples = X.shape[0]
    train_size = min(int(n_samples * 0.9), 5000)

    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, stratify=y)
    label_bin = LabelBinarizer()
    label_bin.fit(y_train)

    sharp = sharp.ShaRP(
        X.shape[1],
        len(np.unique(y_train)),
        "diagonal_normal",
        variational_layer_kwargs=dict(kl_weight=0.1),
        bottleneck_activation="linear",
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
    )
    sharp.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
    X_sharp = sharp.transform(X_train)

    X_umap = UMAP().fit_transform(X_train)

    for c in np.unique(y_train):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        n_elems_in_class = X_train[y_train == c].shape[0]

        ax1.scatter(
            X_sharp[y_train == c, 0],
            X_sharp[y_train == c, 1],
            c=np.ones(n_elems_in_class) * c,
            cmap="tab10",
            alpha=0.3,
        )
        ax1.set_title(f"SSNPVAE, class = {c}")
        ax2.scatter(
            X_umap[y_train == c, 0],
            X_umap[y_train == c, 1],
            c=np.ones(n_elems_in_class) * c,
            cmap="tab10",
            alpha=0.3,
        )
        ax2.set_title(f"UMAP, class = {c}")

        for i in range(X_train.shape[0]):
            if y_train[i] != c:
                continue
            con = ConnectionPatch(
                X_sharp[i],
                X_umap[i],
                "data",
                axesA=ax1,
                axesB=ax2,
                color="grey",
                alpha=0.1,
                lw=0.5,
            )
            ax2.add_artist(con)

    plt.show()
