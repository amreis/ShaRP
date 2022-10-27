import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal, gennorm


def main():
    # We want to generate 3 plots in total, but I'll do that as individual images.
    # I'll use subfigures in LaTeX.
    output_dir = "sampling_normals"
    plt.rcParams.update({"font.size": 50})
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(20, 20))

    dist_1 = multivariate_normal(
        mean=np.zeros(2, dtype=np.float32), cov=np.ones(2, dtype=np.float32)
    )
    samples = dist_1.rvs(1_000)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.4, s=150, c="k", lw=0)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    # ax.set_title('Standard Normal')
    fig.savefig(os.path.join(output_dir, "std_normal.png"), bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 20))
    dist_2 = multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([2.0, 0.25]))
    samples = dist_2.rvs(1_000)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.4, s=150, c="k", lw=0)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    # ax.set_title(r'Normal with $\Sigma$=$\operatorname{diag}([5.0, 0.25])$')
    fig.savefig(
        os.path.join(output_dir, "normal_cov=5.0-0.25.png"), bbox_inches="tight", pad_inches=0.5
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 20))
    dist_3 = gennorm(20, loc=0, scale=2)
    samples = np.c_[dist_3.rvs(1_000), dist_3.rvs(1_000)]
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.4, s=150, c="k", lw=0)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    # ax.set_title(r'Generalized Normal with $\omega = 20$, $\mu = \vec{0}$, $\sigma=[5.0, 5.0]$')
    fig.savefig(
        os.path.join(output_dir, "gen_normal_omega=20.png"), bbox_inches="tight", pad_inches=0.5
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
