import argparse
import os
from colorsys import rgb_to_hsv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image
from sklearn import inspection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP

import ae
import ssnp
import sharp

BoundingBox: type = tuple[float, float, float, float]

cmap = ListedColormap(
    [
        "#aee39a",
        "#9e37d0",
        "#7cee4d",
        "#713d83",
        "#1be19f",
        "#fb2076",
        "#458612",
        "#e89ff0",
        "#115d52",
        "#f79302",
    ]
)


def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, x_max = X_proj[:, 0].min(), X_proj[:, 0].max()
    y_min, y_max = X_proj[:, 1].min(), X_proj[:, 1].max()

    return x_min, x_max, y_min, y_max


def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )

    return np.c_[xx.ravel(), yy.ravel()]


def entropy(probs: np.ndarray) -> float:
    nonzero_probs = probs[probs.nonzero()]
    return -np.sum(nonzero_probs * np.log(nonzero_probs))


def normalized_entropy(probs: np.ndarray) -> float:
    return entropy(probs) / np.log(len(probs))


def batch_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(normalized_entropy, -1, probs)


class EstimatorAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, classes) -> None:
        super().__init__()
        self._base_model = base_model
        self.is_fitted_ = True
        self.classes_ = classes

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self._base_model.predict(X), axis=1)

    def predict_proba(self, X):
        return self._base_model.predict(X)

    def predict_entropy(self, X):
        return batch_normalized_entropy(self._base_model.predict(X))

    def predict_with_entropy(self, X):
        preds = self._base_model.predict(X)
        return np.argmax(preds, axis=1), batch_normalized_entropy(preds)


def dbm_for_estimator(model, bounding_box: BoundingBox, grid_res: int, ax: Axes, cmap=cmap):
    # inspection.DecisionBoundaryDisplay.from_estimator(
    #     model,
    #     proj_data,
    #     response_method="predict",
    #     plot_method="pcolormesh",
    #     grid_resolution=grid_res,
    #     ax=ax,
    #     cmap=cmap,
    # )
    grid = make_grid(*bounding_box, grid_res)
    classes = model.predict(grid).astype(np.uint8)

    cmapped = cmap(classes)
    ax.imshow(
        cmapped.reshape((grid_res, grid_res, 4)),
        origin="lower",
        interpolation="none",
        resample=False,
    )
    ax.axis("off")


def gen_confusion_dbm(
    estimator: EstimatorAdapter,
    bounding_box: BoundingBox,
    grid_res: int,
    ax: Axes,
    confusion_alpha=0.8,
    cmap=cmap,
):
    grid = make_grid(*bounding_box, grid_res)
    classes, confusion = estimator.predict_with_entropy(grid)
    certainty = 1.0 - confusion**confusion_alpha

    classes = classes.astype(np.uint8)
    hsv = [rgb_to_hsv(*cmap(cl)[:3]) for cl in classes]
    confused_hsv = [(h, c, v) for (h, s, v), c in zip(hsv, certainty)]
    img = Image.fromarray(
        (np.reshape(confused_hsv, (grid_res, grid_res, 3)) * 255).astype(np.uint8), mode="HSV"
    )
    ax.imshow(img, origin="lower", interpolation="none", resample=False)
    ax.axis("off")


def gen_and_save_dbm(
    X_2d: np.ndarray,
    classifier: ClassifierMixin,
    output_dir: str,
    grid_res: int,
    dataset_name: str,
    alg_name: str,
):
    fig, ax = plt.subplots(figsize=(20, 20))
    dbm_for_estimator(classifier, get_bounding_box(X_2d), grid_res=grid_res, ax=ax, cmap=cmap)
    fig.savefig(
        fname := os.path.join(output_dir, f"{dataset_name}_{alg_name}.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    print(fname)
    plt.close(fig)


def gen_and_save_confusion_dbm(
    X_2d: np.ndarray,
    classifier: ClassifierMixin,
    output_dir: str,
    grid_res: int,
    dataset_name: str,
    alg_name: str,
):
    fig, ax = plt.subplots(figsize=(20, 20))
    gen_confusion_dbm(classifier, get_bounding_box(X_2d), grid_res=grid_res, ax=ax, cmap=cmap)
    fig.tight_layout()
    fig.savefig(
        fname := os.path.join(output_dir, f"{dataset_name}_{alg_name}_Confusion.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    print(fname)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("experiment_13")
    parser.add_argument("--output-dir", type=str, default="results_dbm")
    parser.add_argument(
        "--datasets", nargs="*", default=["mnist", "fashionmnist", "har", "reuters", "usps"]
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--grid_res", "-g", type=int, default=300)
    parser.add_argument("--neighbors", "-n", type=int, default=11)

    args = parser.parse_args()

    output_dir = args.output_dir
    print(f"Outputtting data to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    verbose = args.verbose

    data_dir = "../data"
    data_dirs = args.datasets
    print(data_dirs)

    epochs = args.epochs
    grid_res = args.grid_res
    n_neighbors = args.neighbors

    for d in data_dirs:
        dataset_name = d

        X = np.load(os.path.join(data_dir, d, "X.npy"))
        y = np.load(os.path.join(data_dir, d, "y.npy"))
        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=min(5_000, int(0.9 * X.shape[0])), random_state=420, stratify=y
        )

        def make_and_fit_knn(data) -> KNeighborsClassifier:
            return KNeighborsClassifier(n_neighbors=n_neighbors).fit(data, y_train)

        sharp_gt = sharp.ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            variational_layer="diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.1),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        sharp_gt.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_gt = sharp_gt.transform(X_train)
        sharp_adapter = EstimatorAdapter(sharp_gt.class_model, np.unique(y_train))
        gen_and_save_dbm(X_sharp_gt, sharp_adapter, output_dir, grid_res, dataset_name, "ShaRP-GT")
        gen_and_save_confusion_dbm(
            X_sharp_gt, sharp_adapter, output_dir, grid_res, dataset_name, "ShaRP-GT"
        )
        sharp_knn = make_and_fit_knn(X_sharp_gt)
        gen_and_save_dbm(
            X_sharp_gt, sharp_knn, output_dir, grid_res, dataset_name, "ShaRP-GT-(KNN)"
        )

        ssnp_gt = ssnp.SSNP(
            epochs=epochs, verbose=verbose, patience=0, opt="adam", bottleneck_activation="linear"
        )
        ssnp_gt.fit(X_train, y_train)
        X_ssnp_gt = ssnp_gt.transform(X_train)
        ssnp_adapter = EstimatorAdapter(ssnp_gt.latent_clustering, np.unique(y_train))
        gen_and_save_dbm(X_ssnp_gt, ssnp_adapter, output_dir, grid_res, dataset_name, "SSNP-GT")
        ssnp_knn = make_and_fit_knn(X_ssnp_gt)
        gen_and_save_confusion_dbm(
            X_ssnp_gt, ssnp_adapter, output_dir, grid_res, dataset_name, "SSNP-GT"
        )
        gen_and_save_dbm(X_ssnp_gt, ssnp_knn, output_dir, grid_res, dataset_name, "SSNP-GT-(KNN)")

        tsne = TSNE(n_jobs=4)
        X_tsne = tsne.fit_transform(X_train)
        tsne_knn = make_and_fit_knn(X_tsne)
        gen_and_save_dbm(X_tsne, tsne_knn, output_dir, grid_res, dataset_name, "TSNE")

        ump = UMAP(random_state=420)
        X_umap = ump.fit_transform(X_train)
        umap_knn = make_and_fit_knn(X_umap)
        gen_and_save_dbm(X_umap, umap_knn, output_dir, grid_res, dataset_name, "UMAP")

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)
        aep_knn = make_and_fit_knn(X_aep)
        gen_and_save_dbm(X_aep, aep_knn, output_dir, grid_res, dataset_name, "AE")

        isomap = Isomap().fit(X_train)
        X_isomap = isomap.transform(X_train)
        isomap_knn = make_and_fit_knn(X_isomap)
        gen_and_save_dbm(X_isomap, isomap_knn, output_dir, grid_res, dataset_name, "ISOMAP")


if __name__ == "__main__":
    main()
