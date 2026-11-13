from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_phase0_diagnostics(step_records: List[Dict[str, Any]], output_dir: str | Path) -> List[Path]:
    """
    Generate publication-style PDF & PNG diagnostic figures:
      1. `fig1a_prior_gap_nll.pdf`: Distribution of Base-Model Mean NLL & Tail CVaR NLL across
         Teacher Reasoning vs. Teacher Action vs. Rewritten Reasoning (if available).
      2. `fig1b_sufficiency_vs_prior.pdf`: Scatter plot in `(Prior Reachability, Action Info Gain)` space.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if not step_records:
        return saved_paths

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    tea_t_nll = np.array([-r["teacher_thought_mean_lp"] for r in step_records if r.get("teacher_thought_tokens", 0) > 0])
    tea_a_nll = np.array([-r["teacher_action_mean_lp"] for r in step_records if r.get("teacher_action_tokens", 0) > 0])
    rew_t_nll = np.array(
        [-r["rewritten_thought_mean_lp"] for r in step_records if "rewritten_thought_mean_lp" in r]
    )

    # --- Figure 1a: Prior Distribution Gap (Box/Violin + Median markers) ---
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = [tea_t_nll, tea_a_nll]
    labels = ["Teacher Thought\n($r_t^*$)", "Verified Action\n($a_t^*$)"]
    colors = ["#d95f02", "#1b9e77"]

    if len(rew_t_nll) > 0:
        groups.append(rew_t_nll)
        labels.append("Base-Aligned Thought\n($\\tilde{r}_t$, Ours)")
        colors.append("#2b5c8f")

    bp = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.45,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Base-Model Token NLL ($-\\log \\pi_0$, lower = more in-distribution)")
    ax.set_title("Off-Policy Surprise Concentrates in Teacher Reasoning")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    p1_pdf = out / "fig1a_prior_gap_nll.pdf"
    p1_png = out / "fig1a_prior_gap_nll.png"
    fig.savefig(p1_pdf)
    fig.savefig(p1_png, dpi=220)
    plt.close(fig)
    saved_paths.extend([p1_pdf, p1_png])

    # --- Figure 1b: Action Info Gain vs. Forward Prior Reachability ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    tea_x = [r["teacher_thought_mean_lp"] for r in step_records]
    tea_y = [r["teacher_info_gain"] for r in step_records]
    ax.scatter(
        tea_x,
        tea_y,
        c="#d95f02",
        alpha=0.55,
        s=28,
        label="Teacher Thought ($r_t^*$)",
        edgecolors="none",
    )

    if any("rewritten_thought_mean_lp" in r for r in step_records):
        rew_x = [r["rewritten_thought_mean_lp"] for r in step_records if "rewritten_thought_mean_lp" in r]
        rew_y = [r["rewritten_info_gain"] for r in step_records if "rewritten_info_gain" in r]
        ax.scatter(
            rew_x,
            rew_y,
            c="#2b5c8f",
            alpha=0.70,
            s=32,
            marker="^",
            label="ELBO-Aligned Thought ($\\tilde{r}_t$, Ours)",
            edgecolors="none",
        )

    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Forward Prior Reachability $\\frac{1}{|r_t|}\\log \\pi_0(r_t \\mid h_t)$ (higher = closer to Base)")
    ax.set_ylabel("Action Info Gain $\\Delta_{\\mathrm{act}}(r_t)$ (nats, >0 helps predict $a_t^*$)")
    ax.set_title("Posterior-to-Prior Tradeoff in Agent Step Supervision")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    p2_pdf = out / "fig1b_sufficiency_vs_prior.pdf"
    p2_png = out / "fig1b_sufficiency_vs_prior.png"
    fig.savefig(p2_pdf)
    fig.savefig(p2_png, dpi=220)
    plt.close(fig)
    saved_paths.extend([p2_pdf, p2_png])

    return saved_paths
