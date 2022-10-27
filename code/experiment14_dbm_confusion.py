import argparse
import os
from colorsys import rgb_to_hsv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

import ssnp
import sharp

BoundingBox = tuple[float, float, float, float]

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
    x_min: float, x_max: float, y_min: float, y_max: float, resolution: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
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


def get_blend_dbm(grid_res: int, bounding_box: BoundingBox, estimator: EstimatorAdapter):
    grid = make_grid(*bounding_box, grid_res)
    probabilities = estimator.predict_proba(grid)
    class_colors = np.array([cmap(cl) for cl in estimator.classes_])[:, :3]

    blended = (probabilities[..., np.newaxis] * class_colors).sum(axis=1)

    img = Image.fromarray(
        (np.reshape(blended, (grid_res, grid_res, 3)) * 255).astype(np.uint8), mode="RGB"
    )
    plt.imshow(img, origin="lower", interpolation="none", resample=False)
    plt.axis("off")
    plt.tight_layout()


def get_confusion_dbm(
    grid_res: int, bounding_box: BoundingBox, estimator: EstimatorAdapter, confusion_alpha=0.8
):
    grid = make_grid(*bounding_box, grid_res)
    classes, confusion = estimator.predict_with_entropy(grid)
    certainty = 1.0 - confusion**confusion_alpha

    classes = classes.astype(np.uint8)
    hsv = [rgb_to_hsv(*cmap(cl)[:3]) for cl in classes]
    confused_hsv = [(h, s * c, v) for (h, s, v), c in zip(hsv, certainty)]
    img = Image.fromarray(
        (np.reshape(confused_hsv, (grid_res, grid_res, 3)) * 255).astype(np.uint8), mode="HSV"
    )
    plt.imshow(img, origin="lower", interpolation="none", resample=False)
    plt.axis("off")
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser("experiment_13")
    parser.add_argument("--output-dir", type=str, default="results_dbm_confusion")
    parser.add_argument(
        "--datasets", nargs="+", default=["mnist", "fashionmnist", "har", "reuters", "usps"]
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--grid-res", "-g", type=int, default=500)
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--confusion-alpha", type=float, default=0.8)

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
    confusion_alpha = args.confusion_alpha

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

        sharp_gt = sharp.ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            variational_layer="diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.01),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        sharp_gt.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_gt = sharp_gt.transform(X_train)

        get_confusion_dbm(
            grid_res,
            bbox := get_bounding_box(X_sharp_gt),
            sharp_adapter := EstimatorAdapter(sharp_gt.class_model, np.unique(y_train)),
            confusion_alpha=confusion_alpha,
        )
        plt.savefig(
            fname := os.path.join(output_dir, f"{dataset_name}_Confusion_ShaRP-GT.png"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        print(fname)
        plt.close()

        get_blend_dbm(grid_res, bbox, sharp_adapter)
        plt.savefig(
            fname := os.path.join(output_dir, f"{dataset_name}_Blend_ShaRP-GT.png"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        print(fname)
        plt.close()

        ssnp_gt = ssnp.SSNP(
            epochs=epochs, verbose=verbose, patience=0, opt="adam", bottleneck_activation="linear"
        )
        ssnp_gt.fit(X_train, y_train)
        X_ssnp_gt = ssnp_gt.transform(X_train)
        ssnp_adapter = EstimatorAdapter(ssnp_gt.latent_clustering, np.unique(y_train))
        get_confusion_dbm(
            grid_res,
            bbox := get_bounding_box(X_ssnp_gt),
            ssnp_adapter,
        )
        plt.savefig(
            fname := os.path.join(output_dir, f"{dataset_name}_Confusion_SSNP-GT.png"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        print(fname)
        plt.close()

        get_blend_dbm(grid_res, bbox, ssnp_adapter)
        plt.savefig(
            fname := os.path.join(output_dir, f"{dataset_name}_Blend_SSNP-GT.png"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        print(fname)
        plt.close()


if __name__ == "__main__":
    main()
