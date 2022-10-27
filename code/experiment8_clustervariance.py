import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from umap import UMAP

import ssnp
from sharp import ShaRP


def standardize(obs: np.ndarray) -> np.ndarray:
    assert obs.ndim == 1, "this function is meant to be called on 1d arrays only"
    return StandardScaler().fit_transform(obs.reshape(-1, 1)).reshape(-1)


def convex_hull_sizes(proj_data, y, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))
    areas = np.zeros(n_classes, dtype=np.float32)
    for cl in range(n_classes):
        class_data = proj_data[y == cl]
        hull = ConvexHull(class_data, incremental=False)
        area = hull.area
        areas[cl] = area
    return areas


def cluster_convex_hull_sizes(proj_data, y, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))
    areas = np.zeros(n_classes, dtype=np.float32)
    for cl in range(n_classes):
        data = proj_data[y == cl]
        dbscan = DBSCAN(eps=1.0, min_samples=9)
        dbscan.fit(data)
        for_hull = data[dbscan.labels_ != -1]
        hull = ConvexHull(for_hull, incremental=False)
        areas[cl] = hull.area
    return areas


def class_variances(X, y, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))
    variances = np.zeros(n_classes, dtype=np.float32)
    for cl in range(n_classes):
        class_data = X[y == cl]
        # variances[cl] = np.var(class_data, axis=0).sum()
        variances[cl] = np.trace(np.cov(class_data, rowvar=False)) / class_data.shape[0]
    return variances


def cluster_size_consistency(X_high, y, X_proj, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))
    variances = np.zeros(n_classes, dtype=np.float32)
    areas = np.zeros_like(variances)
    for cl in range(n_classes):
        orig_class_data = X_high[y == cl]
        proj_class_data = X_proj[y == cl]
        main_cluster_points = proj_class_data[IsolationForest().fit_predict(proj_class_data) == 1]
        hull = ConvexHull(main_cluster_points, incremental=False)
        areas[cl] = hull.area
        variances[cl] = np.trace(np.cov(orig_class_data, rowvar=False)) / proj_class_data.shape[0]
    return spearmanr(variances, areas)


if __name__ == "__main__":
    output_dir = "results_clustervariance"
    epochs = 100
    verbose = False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = "../data"
    data_dirs = ["mnist", "fashionmnist", "har", "reuters"]

    for d in data_dirs:
        dataset_name = d

        X: np.ndarray = np.load(os.path.join(data_dir, d, "X.npy"))
        y: np.ndarray = np.load(os.path.join(data_dir, d, "y.npy"))

        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        n_samples = X.shape[0]
        train_size = min(int(n_samples * 0.9), 5_000)

        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        label_bin = LabelBinarizer()
        label_bin.fit(y_train)
        sharp = ShaRP(
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

        ssnpgt = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        ssnpgt.fit(X_train, y_train)
        X_ssnpgt = ssnpgt.transform(X_train)

        tsne = TSNE(n_jobs=4, random_state=420)
        X_tsne = tsne.fit_transform(X_train)

        ump = UMAP(random_state=420)
        X_umap = ump.fit_transform(X_train)

        # Since methods might project datasets in different scales, we should check
        # ONLY whether cluster size correlates with intra-class variability. We can't
        # compare cluster sizes between methods, only this correlation.

        variances = class_variances(X_train, y_train)

        sharp_areas = convex_hull_sizes(X_sharp, y_train)
        ssnpgt_areas = convex_hull_sizes(X_ssnpgt, y_train)
        tsne_areas = convex_hull_sizes(X_tsne, y_train)
        umap_areas = convex_hull_sizes(X_umap, y_train)

        sharp_denoised_areas = cluster_convex_hull_sizes(X_sharp, y_train)
        ssnpgt_denoised_areas = cluster_convex_hull_sizes(X_ssnpgt, y_train)
        tsne_denoised_areas = cluster_convex_hull_sizes(X_tsne, y_train)
        umap_denoised_areas = cluster_convex_hull_sizes(X_umap, y_train)

        get_pearson_r = partial(pearsonr, variances)
        get_spearman_r = partial(spearmanr, variances)

        print("PEARSON")
        print(
            f"SSNP-VAE: {get_pearson_r(sharp_areas)}, "
            f"SSNP-GT: {get_pearson_r(ssnpgt_areas)}, "
            f"TSNE: {get_pearson_r(tsne_areas)}, "
            f"UMAP: {get_pearson_r(umap_areas)}"
        )
        print("SPEARMAN")
        print(
            f"SSNP-VAE: {get_spearman_r(sharp_areas)}, "
            f"SSNP-GT: {get_spearman_r(ssnpgt_areas)}, "
            f"TSNE: {get_spearman_r(tsne_areas)}, "
            f"UMAP: {get_spearman_r(umap_areas)}"
        )

        print("DENOISED-PEARSON")
        print(
            f"SSNP-VAE: {get_pearson_r(sharp_denoised_areas)}, "
            f"SSNP-GT: {get_pearson_r(ssnpgt_denoised_areas)}, "
            f"TSNE: {get_pearson_r(tsne_denoised_areas)}, "
            f"UMAP: {get_pearson_r(umap_denoised_areas)}"
        )
        print("DENOISED-SPEARMAN")
        print(
            f"SSNP-VAE: {get_spearman_r(sharp_denoised_areas)}, "
            f"SSNP-GT: {get_spearman_r(ssnpgt_denoised_areas)}, "
            f"TSNE: {get_spearman_r(tsne_denoised_areas)}, "
            f"UMAP: {get_spearman_r(umap_denoised_areas)}"
        )
