import argparse
import os

import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelBinarizer
from umap import UMAP

import ae
import ssnp
from metrics import cluster_size_consistency, plot_data_and_hulls
from sharp import ShaRP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment_9")
    parser.add_argument("--output-dir", type=str, default="results_clusterconsistency")
    args = parser.parse_args()
    output_dir = args.output_dir
    print(f"Outputting data to {output_dir}")
    epochs = 20
    verbose = False

    results = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = os.path.join("..", "data")
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

        sharp_gt = ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            "diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.05),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )

        sharp_gt.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_gt = sharp_gt.transform(X_train)

        n_classes = (1 if dataset_name == "reuters" else 2) * len(np.unique(y_train))
        sharp_km = ShaRP(
            X.shape[1],
            n_classes,
            "diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.05),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        C = KMeans(n_clusters=n_classes)
        y_km = C.fit_predict(X_train)
        sharp_km.fit(X_train, y_km, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_km = sharp_km.transform(X_train)

        sharp_ag = ShaRP(
            X.shape[1],
            n_classes,
            "diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.05),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        C = AgglomerativeClustering(n_clusters=n_classes)
        y_ag = C.fit_predict(X_train)
        sharp_ag.fit(X_train, y_ag, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_ag = sharp_ag.transform(X_train)

        del sharp_gt, sharp_ag, sharp_km
        sharp_gt_sq = ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            "generalized_normal",
            variational_layer_kwargs=dict(power=10),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )

        sharp_gt_sq.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_gt_sq = sharp_gt_sq.transform(X_train)

        n_classes = (1 if dataset_name == "reuters" else 2) * len(np.unique(y_train))
        sharp_km_sq = ShaRP(
            X.shape[1],
            n_classes,
            "generalized_normal",
            variational_layer_kwargs=dict(power=10),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        C = KMeans(n_clusters=n_classes)
        y_km = C.fit_predict(X_train)
        sharp_km_sq.fit(X_train, y_km, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_km_sq = sharp_km_sq.transform(X_train)

        sharp_ag_sq = ShaRP(
            X.shape[1],
            n_classes,
            "generalized_normal",
            variational_layer_kwargs=dict(power=10),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        C = AgglomerativeClustering(n_clusters=n_classes)
        y_ag = C.fit_predict(X_train)
        sharp_ag_sq.fit(X_train, y_ag, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_ag_sq = sharp_ag_sq.transform(X_train)

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

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)

        for X_proj, algo in (
            (X_sharp_gt, "ShaRP-GT"),
            (X_sharp_km, "ShaRP-KMeans"),
            (X_sharp_ag, "ShaRP-AG"),
            (X_sharp_gt_sq, "ShaRP-GT-SQ"),
            (X_sharp_km_sq, "ShaRP-KMeans-SQ"),
            (X_sharp_ag_sq, "ShaRP-AG-SQ"),
            (X_ssnpgt, "SSNP-GT"),
            (X_tsne, "TSNE"),
            (X_umap, "UMAP"),
            (X_aep, "AE"),
        ):
            score_and_pvalue_iso, hulls_iso = cluster_size_consistency(
                X_train, y_train, X_proj, return_hulls=True
            )
            score_and_pvalue_lof, hulls_lof = cluster_size_consistency(
                X_train,
                y_train,
                X_proj,
                outlier_detector_maker=LocalOutlierFactor,
                return_hulls=True,
            )
            results.append((dataset_name, algo) + score_and_pvalue_iso + score_and_pvalue_lof)
            fname = os.path.join(output_dir, f"{dataset_name}_{algo}_ISO.png")
            print(fname)
            plot_data_and_hulls(X_proj, y_train, hulls_iso, fname=fname, save_without_hulls=True)
            fname = os.path.join(output_dir, f"{dataset_name}_{algo}_LOF.png")
            print(fname)
            plot_data_and_hulls(X_proj, y_train, hulls_lof, fname=fname, save_without_hulls=False)

    df = pd.DataFrame(
        results,
        columns=[
            "dataset_name",
            "test_name",
            "spearman_corr_iso",
            "p_value_iso",
            "spearman_corr_lof",
            "p_value_lof",
        ],
    )
    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)
