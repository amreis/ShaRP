import argparse
import os

import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from umap import UMAP

import ae
import metrics
import region_growing
import ssnp
from sharp import ShaRP


def labels_scatter_plot(X_proj, y):
    s_plot = region_growing.pixel_scatter_plot(X_proj, y)
    return region_growing.pixels_to_labels(s_plot)


def plot_and_grow_regions(X_proj, y):
    return region_growing.region_growth(labels_scatter_plot(X_proj, y))


def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)
    DSC = metrics.distance_consistency(X_2d, y)
    CC = 0.5 + 0.5 * metrics.cluster_size_consistency_r(X, y, X_2d)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0

    return T, C, R, S, N, DSC, CC, MSE


def main():
    parser = argparse.ArgumentParser(description="experiment_11")

    parser.add_argument("--output-dir", type=str, default="results_regiongrowing")
    parser.add_argument(
        "--datasets", nargs="*", default=["mnist", "fashionmnist", "har", "reuters", "usps"]
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose

    results = []
    output_dir = args.output_dir
    print(f"Outputting data to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = os.path.join("..", "data")
    data_dirs = args.datasets
    print(data_dirs)

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10 * 2
    epochs_dataset["mnist"] = 10 * 2
    epochs_dataset["har"] = 10 * 2
    epochs_dataset["reuters"] = 10 * 2
    epochs_dataset["usps"] = 10 * 2

    classes_mult = {}
    classes_mult["fashionmnist"] = 2
    classes_mult["mnist"] = 2
    classes_mult["har"] = 2
    classes_mult["reuters"] = 1
    classes_mult["usps"] = 1

    for d in data_dirs:
        dataset_name = d

        X = np.load(os.path.join(data_dir, d, "X.npy"))
        y = np.load(os.path.join(data_dir, d, "y.npy"))

        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        n_clusters = len(np.unique(y)) * classes_mult[dataset_name]
        n_samples = X.shape[0]

        train_size = min(int(n_samples * 0.9), 5000)

        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        label_bin = LabelBinarizer()
        label_bin.fit(y_train)
        D_high = metrics.compute_distance_list(X_train)

        epochs = epochs_dataset[dataset_name]

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
            (X_ssnpgt, "SSNP-GT"),
            (X_tsne, "TSNE"),
            (X_umap, "UMAP"),
            (X_aep, "AE"),
        ):
            results.append(
                (dataset_name, algo)
                + compute_all_metrics(
                    X_train, X_proj, D_high, metrics.compute_distance_list(X_proj), y_train
                )
            )
            out_img = os.path.join(output_dir, f"{dataset_name}_{algo}.png")
            region_growing.render_img(
                plot_and_grow_regions(X_proj, y_train),
                fname=out_img,
            )
            print(out_img)

    df = pd.DataFrame(
        results,
        columns=[
            "dataset_name",
            "test_name",
            "T_train",
            "C_train",
            "R_train",
            "S_train",
            "N_train",
            "DSC_train",
            "CC_train",
            "MSE_train",
        ],
    )

    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)


if __name__ == "__main__":
    main()
