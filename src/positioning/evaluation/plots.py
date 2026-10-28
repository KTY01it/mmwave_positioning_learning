"""Standard figures: RMSE vs SNR with bound, Delta vs SNR, CDF, RMSE/PEB vs G."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EST_COLOR, BOUND_COLOR = "#1f5fbf", "#c8102e"


def rmse_vs_snr(ax, snr, rmse, bound, label, bound_label, ylabel, marker="o"):
    ax.semilogy(snr, rmse, marker=marker, color=EST_COLOR, label=label)
    ax.semilogy(snr, bound, "--", marker="x", color=BOUND_COLOR, label=bound_label)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)


def figure_rmse_panels(snr, panels, title, path):
    """panels: list of dicts(rmse, bound, label, bound_label, ylabel)."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(5.2, 2.6 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        rmse_vs_snr(ax, snr, p["rmse"], p["bound"], p["label"], p["bound_label"], p["ylabel"])
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_cdf(series: dict, xlabel, path, title=""):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for name, vals in series.items():
        v = np.sort(np.asarray(vals)[np.isfinite(vals)])
        ax.plot(v, np.arange(1, v.size + 1) / max(v.size, 1), label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_vs_beams(G, rmse_p90, peb_p90, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(G, peb_p90, "-s", color=BOUND_COLOR, label="PEB (CDF = 0.9)")
    ax.plot(G, rmse_p90, "--o", color=EST_COLOR, label="error (CDF = 0.9)")
    ax.set_xlabel("number of beams G")
    ax.set_ylabel("localization error [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_lines(x, series: dict, xlabel, ylabel, path, title=""):
    """Several curves on linear axes (e.g. normalised error in dB vs SNR)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for (name, ys), mk in zip(series.items(), "osd^v<>"):
        ax.plot(x, ys, marker=mk, label=name)
    ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
