from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image
from sklearn.preprocessing import LabelEncoder

EMPTY = -1


def pixel_scatter_plot(X_proj: np.ndarray, y: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(2, 2))

    norm = Normalize(0, y.max())
    cmap = plt.get_cmap("tab10")
    for cl in np.unique(y):
        ax.plot(
            X_proj[y == cl, 0],
            X_proj[y == cl, 1],
            color=cmap(norm(cl)),
            marker=",",
            lw=0,
            linestyle="",
        )
    ax.axis("off")
    fig.tight_layout()

    fig.canvas.draw()
    rgb = [int(px) for px in fig.canvas.tostring_rgb()]
    plt.close(fig)
    n_px = len(rgb)
    colors = 3
    height = width = int(np.sqrt(n_px / colors))
    rgb_grouped = np.reshape(rgb, (height, width, colors)).astype(np.uint8)

    return rgb_grouped


def pixels_to_labels(s_plot: np.ndarray) -> np.ndarray:
    assert (
        s_plot.ndim == 3
    ), f"input to `pixels_to_labels` must have 3 dimensions (h, w, c), got {s_plot.shape}"
    orig_h, orig_w, *unused = s_plot.shape
    s_plot = s_plot.copy().reshape((-1, 3))
    is_white_px = np.apply_along_axis(lambda pixel: tuple(pixel) == (255, 255, 255), -1, s_plot)

    labels = np.zeros(s_plot.shape[0], dtype=np.int)
    labels[is_white_px] = EMPTY

    labeler = LabelEncoder()
    tupled_pxs = np.apply_along_axis(lambda px: " ".join(map(str, px)), -1, s_plot[~is_white_px])
    labels[~is_white_px] = labeler.fit_transform(tupled_pxs)

    return labels.reshape((orig_h, orig_w))


def step_region_growth(img: np.ndarray, seeds: np.ndarray, empty_ixs: set[int]) -> np.ndarray:
    """Perform one step of region growth for every non-empty pixel in the image.

    Args:
        img (np.ndarray): image over which we're operating
        seeds (np.ndarray): seed points from which to expand the regions. The code assumes
        that seed points are _not_ empty.
        empty_ixs (set[int]): flat indices of empty pixels in the image. This
        parameter is modified by the algorithm and should contain all empty pixels
        when this function is called for the first time.

    Returns:
        np.ndarray: The new `seeds` to be used for the next iteration. If this is empty,
        the algorithm shouldn't be called again.
    """
    rows, cols = img.shape
    new_seeds = []
    for i, j in seeds:
        to_check = np.array([i, j]) + np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        flat_ixs = np.ravel_multi_index(tuple(to_check.transpose()), dims=(rows, cols), mode="clip")
        to_change = np.array([ix for ix in flat_ixs if ix in empty_ixs])
        if len(to_change) > 0:
            img.ravel()[to_change] = img[i, j]
            empty_ixs -= set(to_change)
            new_seeds.extend(to_change)

    return np.stack(np.unravel_index(np.array(new_seeds, dtype=np.intp), (rows, cols))).transpose()


def region_growth(img: np.ndarray, inplace=False) -> np.ndarray:
    if not inplace:
        img = img.copy()
    seeds = np.argwhere(img != EMPTY)
    flat_empty_ixs = set(
        np.ravel_multi_index(
            tuple(np.argwhere(img == EMPTY).transpose()), img.shape[:2], mode="raise"
        )
    )
    while len(seeds := step_region_growth(img, seeds, flat_empty_ixs)) > 0:
        pass
    return img


def render_img(img, cmap="tab10", fname: Optional[str] = None):
    from matplotlib.colors import ListedColormap

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
    plt.imshow(img, cmap=cmap, interpolation="none", resample=False)
    plt.axis("off")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, pad_inches=0.0, bbox_inches="tight")
    else:
        plt.show()


def prepare_img(img, cmap="tab10") -> Image:
    cmap = plt.get_cmap(cmap)
    from matplotlib.colors import Normalize

    norm = Normalize(0, np.max(img))
    rows, cols = img.shape
    new_img = np.zeros((rows, cols, 4), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            if img[i, j] == EMPTY:
                new_img[i, j, :] = 0.0
            else:
                new_img[i, j, :] = cmap(norm(img[i, j]))
    return Image.fromarray((new_img * 255.0).astype(np.uint8), mode="RGBA")


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    img = np.random.randint(10, size=(100, 100)).astype(np.float32)
    img[np.random.sample(img.shape) < 0.9] = EMPTY
    ax1.imshow(prepare_img(img))
    # plt.scatter(xx.ravel(), yy.ravel(), c=img.ravel(), cmap='tab10', marker=",")
    img = region_growth(img)  # or use inplace=True

    # img[img == EMPTY] = np.nan
    # print(np.unique(img.ravel()))
    # plt.scatter(xx.ravel(), yy.ravel(), c=img.ravel(), cmap='tab10', marker=",")
    ax2.imshow(prepare_img(img))
    plt.show()
