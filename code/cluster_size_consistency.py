from typing import Literal
import numpy as np
from sklearn.preprocessing import LabelBinarizer

from sharp import ShaRP
from metrics import cluster_size_consistency, plot_data_and_hulls


def main():
    from MulticoreTSNE import MulticoreTSNE as TSNE
    from sklearn.model_selection import train_test_split

    X, y = np.load("../data/mnist/X.npy"), np.load("../data/mnist/y.npy")
    X_train, _, y_train, _ = train_test_split(X, y, train_size=5_000, stratify=y)

    X_tsne = TSNE(n_jobs=16).fit_transform(X_train)
    consistency_score, hulls = cluster_size_consistency(X_train, y_train, X_tsne, return_hulls=True)
    print(consistency_score)

    # plot_data_and_hulls(X_tsne, y_train, hulls, n_classes=10)

    label_bin = LabelBinarizer()
    label_bin.fit(y_train)
    sharp = ShaRP(
        X.shape[1],
        len(np.unique(y_train)),
        "diagonal_normal",
        variational_layer_kwargs=dict(kl_weight=0.1, kl_mu_weight=0.0),
        bottleneck_activation="linear",
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
    )

    sharp.fit(X_train, y_train, epochs=20, verbose=True, batch_size=64)
    X_sharp = sharp.transform(X_train)
    consistency_score, hulls = cluster_size_consistency(
        X_train, y_train, X_sharp, return_hulls=True
    )
    print(consistency_score)

    plot_data_and_hulls(X_sharp, y_train, hulls, n_classes=10)


# Idea: monitor correlation's evolution throughout training with a callback?
if __name__ == "__main__":
    main()
