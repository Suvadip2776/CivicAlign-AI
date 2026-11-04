"""Figures for the CivicAlign AI paper.

    python figures/make_figures.py

Inputs
  results/eigenbench/direct_rating: bootstrap/summary.json, bootstrap/samples.json,
    per_criterion_elo.json (eigenbench/tools/per_criterion_elo.py)
  results/capability/capability_table.csv (capability_evals/compile_results.py)
Outputs (PDF + PNG) in figures/output/:
  fig1_elo_ranking, fig2_prompt_x_training, fig3_per_principle,
  fig4_capability_effects, fig5_alignment_vs_capability
"""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/eigenbench/direct_rating"
CAP = ROOT / "results/capability/capability_table.csv"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, INK2, GRID = "#1a1a1a", "#555555", "#e4e4e4"
BLUE, ORANGE, GRAY, LGRAY = "#2a78d6", "#eb6834", "#7d7d7d", "#b5b5b5"

# Display names: identical across every figure in the paper.
NAME = {
    "Qwen + tools, no constitution": "Baseline agent (Qwen 3.8 27B + tools)",
    "NYCC Agent": "NYCC Agent (+ constitution prompt)",
    "Character-trained model + tools": "Character-trained model + tools",
    "NYCC character trained agent": "NYCC character trained agent",
    "Character-trained model": "Character-trained model (no tools)",
    "Qwen 3.8 27B (bare)": "Qwen 3.8 27B (no tools)",
    "Claude Opus 4.7 (bare)": "Claude Opus 4.7 (no tools)",
    "GPT-6 Sol (bare)": "GPT-6 Sol (no tools)",
}
QWEN4 = ["Qwen + tools, no constitution", "NYCC Agent", "Character-trained model + tools",
         "NYCC character trained agent"]
PRINCIPLES = ["Truth before helpfulness", "Law vs. proposal", "Cite public authority",
              "Acknowledge uncertainty", "Political neutrality", "Competing views fairly",
              "Resident privacy", "Civic access", "Defer high-stakes decisions", "Be auditable"]


def name(raw):
    return NAME.get(raw, raw)


def color(raw):
    if raw in ("Qwen + tools, no constitution", "NYCC Agent"):
        return BLUE
    if raw in ("Character-trained model + tools", "NYCC character trained agent", "Character-trained model"):
        return ORANGE
    if raw == "Qwen 3.8 27B (bare)":
        return BLUE
    return GRAY


def tools(raw):
    return not ("(bare)" in raw or raw == "Character-trained model")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf")


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


summary = json.load(open(RUN / "bootstrap/summary.json"))
order = json.load(open(RUN / "analysis_config.json"))["model_order"]
boot = np.array([s["elo_vector"] for s in json.load(open(RUN / "bootstrap/samples.json"))])  # (B, models)
idx = {m: i for i, m in enumerate(order)}

# ------------------------------------------------------------------ Fig 1: ranking
rows = sorted(summary, key=lambda r: r["elo_mean"])
fig, ax = plt.subplots(figsize=(3.4, 3.3))
for y, r in enumerate(rows):
    m, c = r["model_name"], color(r["model_name"])
    ax.plot([r["elo_ci_lower"], r["elo_ci_upper"]], [y, y], color=c, lw=1.6, solid_capstyle="round")
    ax.plot(r["elo_mean"], y, "o", ms=4.8, mfc=c if tools(m) else "white", mec=c, mew=1.2, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([name(r["model_name"]) for r in rows])
for lab, r in zip(ax.get_yticklabels(), rows):
    if r["model_name"] in QWEN4:
        lab.set_fontweight("bold")
ax.axvline(1500, color=LGRAY, lw=0.6, ls=":", zorder=0)
ax.set_xlabel("EigenBench Elo (bootstrap mean, 95% CI)")
ax.tick_params(axis="y", length=0)
style(ax)
ax.legend(handles=[
    Line2D([], [], marker="o", color=BLUE, lw=0, ms=4.5, label="Qwen 3.8 27B, base weights"),
    Line2D([], [], marker="o", color=ORANGE, lw=0, ms=4.5, label="Qwen 3.8 27B, character-trained"),
    Line2D([], [], marker="o", color=GRAY, lw=0, ms=4.5, label="Frontier model"),
    Line2D([], [], marker="o", mfc="white", mec=INK2, lw=0, ms=4.5, label="Hollow: no tools"),
], loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.1)
save(fig, "fig1_elo_ranking")

# ------------------------------------------------------------------ Fig 2: 2x2
cells = {("base", 0): "Qwen + tools, no constitution", ("base", 1): "NYCC Agent",
         ("oct", 0): "Character-trained model + tools", ("oct", 1): "NYCC character trained agent"}


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


fig, ax = plt.subplots(figsize=(3.4, 2.4))
for w, c, lab, dx in (("base", BLUE, "Base weights", -0.03), ("oct", ORANGE, "Character-trained (OCT)", 0.03)):
    ys = [boot[:, idx[cells[(w, p)]]] for p in (0, 1)]
    means = [y.mean() for y in ys]
    ax.plot([0 + dx, 1 + dx], means, color=c, lw=1.4, label=lab)
    for p, y in enumerate(ys):
        lo, hi = ci(y)
        ax.plot([p + dx, p + dx], [lo, hi], color=c, lw=1.4)
        ax.plot(p + dx, y.mean(), "o", color=c, ms=5, mec="white", mew=0.8, zorder=3)


def contrast(a, b):
    d = boot[:, idx[b]] - boot[:, idx[a]]
    lo, hi = ci(d)
    return d.mean(), lo, hi


eff = {
    r"$\Delta_{\mathrm{prompt}}$": contrast(cells[("base", 0)], cells[("base", 1)]),
    r"$\Delta_{\mathrm{OCT}}$": contrast(cells[("base", 0)], cells[("oct", 0)]),
    r"$\Delta_{\mathrm{OCT}\mid\mathrm{prompt}}$": contrast(cells[("base", 1)], cells[("oct", 1)]),
}
txt = "\n".join(f"{k} = {m:+.0f}  [{lo:+.0f}, {hi:+.0f}]" for k, (m, lo, hi) in eff.items())
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
ax.set_xticks([0, 1])
ax.set_xticklabels(["No constitution", "Constitution prompt"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("EigenBench Elo (95% CI)")
ax.legend(loc="lower right", frameon=False)
style(ax, "y")
save(fig, "fig2_prompt_x_training")
json.dump({k: dict(zip(("mean", "lo", "hi"), v)) for k, v in eff.items()},
          open(OUT / "fig2_contrasts.json", "w"), indent=1)

# ------------------------------------------------------------------ Fig 3: per-principle heatmap
pc = json.load(open(RUN / "per_criterion_elo.json"))
cols = [r["model_name"] for r in sorted(summary, key=lambda r: -r["elo_mean"])]
M = np.array([[pc[str(k + 1)]["arms"][m]["elo"] for k in range(10)] for m in cols])
R = np.array([[pc[str(k + 1)]["arms"][m]["rank"] for k in range(10)] for m in cols])
fig, ax = plt.subplots(figsize=(7.0, 3.9))
lim = np.abs(M - 1500).max()
im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(1500, 1500 - lim, 1500 + lim), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        dark = abs(M[i, j] - 1500) > 0.55 * lim
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if dark else INK, fontweight="bold" if R[i, j] == 1 else "normal")
ax.set_yticks(range(len(cols)))
ax.set_yticklabels([name(m) for m in cols])
for lab, m in zip(ax.get_yticklabels(), cols):
    if m in QWEN4:
        lab.set_fontweight("bold")
ax.set_xticks(range(10))
ax.set_xticklabels([f"{k + 1}. {p}" for k, p in enumerate(PRINCIPLES)], rotation=35, ha="right")
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("Per-principle Elo (bold = rank 1)", fontsize=7)
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=6.5)
save(fig, "fig3_per_principle")

# ------------------------------------------------------------------ capability data
cap = {}
if CAP.exists():
    for r in csv.DictReader(open(CAP)):
        cap[r["benchmark"]] = r


def val(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "None", "nan") else None


# ------------------------------------------------------------------ Fig 4: capability effects
FOREST = [  # (row in table, short label)
    ("MMLU-Pro accuracy", "MMLU-Pro"), ("GPQA Diamond accuracy", "GPQA Diamond"), ("MATH accuracy", "MATH"),
    ("LegalBench (8 tasks) accuracy", "LegalBench (8 tasks)"), ("TruthfulQA MC1 accuracy", "TruthfulQA (MC1)"),
    ("SimpleQA correct", "SimpleQA correct"), ("MASK honesty", "MASK honesty"),
    ("Sycophancy: caves to pushback", "Sycophancy: caves to pushback"), ("BBQ accuracy", "BBQ accuracy"),
    ("XSTest safe: refusal rate", "XSTest: refuses safe prompts"),
    ("XSTest unsafe: refusal rate", "XSTest: refuses unsafe prompts"),
    ("ConfAIde 3: secret leaked (judged)", "ConfAIde 3: secret leaked"),
]
GROUPS = [(0, 4, "Knowledge & reasoning"), (4, 8, "Honesty & factuality"), (8, 12, "Fairness, refusal & privacy")]
# Sycophancy "truthfulness" is left out: it is built on the initial-answer grade, which marks the
# longer, qualified answers the constitution produces as wrong even when they are correct.
EFFECTS = [("Training (OCT − Base)", "Character training", ORANGE, 0.22),
           ("Prompt on base", "Constitution prompt (base)", BLUE, 0.0),
           ("Prompt on OCT", "Constitution prompt (trained)", "#8fb8ea", -0.22)]
rows4 = [(cap[k], lab) for k, lab in FOREST if k in cap]
if rows4:
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    n = len(rows4)
    for i, (r, lab) in enumerate(rows4):
        y = n - 1 - i
        hib = r["higher_is_better"]
        sign = -1.0 if hib == "False" else 1.0  # plot so that right = better
        for col, _, c, dy in EFFECTS:
            d, se = val(r, col), val(r, col + " se")
            if d is None:
                continue
            d *= sign
            h = 1.96 * (se or 0)
            sig = se and abs(d) > h
            ax.plot([d - h, d + h], [y + dy, y + dy], color=c, lw=1.1)
            ax.plot(d, y + dy, "o", ms=3.6, color=c if sig else "white", mec=c, mew=1.0, zorder=3)
    ax.axvline(0, color=INK2, lw=0.7)
    for a, b, g in GROUPS:
        if b < n:
            ax.axhline(n - b - 0.5, color=GRID, lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in rows4][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Change in score, percentage points (right = better)")
    ax.tick_params(axis="y", length=0)
    style(ax)
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=3.6, lw=1.1, label=l) for _, l, c, _ in EFFECTS] +
              [Line2D([], [], color=INK2, marker="o", mfc="white", ms=3.6, lw=0, label="Hollow: within ±1.96 SE")],
              loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False, columnspacing=0.8)
    save(fig, "fig4_capability_effects")
else:
    print("skipped fig4: no capability table")

# ------------------------------------------------------------------ Fig 5: alignment vs capability
CFG = {"Qwen + tools, no constitution": "Base", "NYCC Agent": "Base + const.",
       "Character-trained model + tools": "OCT", "NYCC character trained agent": "OCT + const."}
SHORT = {"Qwen + tools, no constitution": "Baseline", "NYCC Agent": "NYCC Agent",
         "Character-trained model + tools": "Char.-trained", "NYCC character trained agent": "NYCC char.-trained agent"}
PANELS = [("MASK honesty", "MASK honesty (%)"), ("XSTest safe: refusal rate", "Refuses safe prompts, XSTest (%)")]
if all(p in cap for p, _ in PANELS):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    elo = {r["model_name"]: r for r in summary}
    for ax, (row, ylab) in zip(axes, PANELS):
        r = cap[row]
        for m in QWEN4:
            e, y, se = elo[m], val(r, CFG[m]), val(r, CFG[m] + " se")
            if y is None:
                continue
            c = BLUE if m in QWEN4[:2] else ORANGE
            const = m in (QWEN4[1], QWEN4[3])
            ax.errorbar(e["elo_mean"], y, xerr=[[e["elo_mean"] - e["elo_ci_lower"]], [e["elo_ci_upper"] - e["elo_mean"]]],
                        yerr=1.96 * (se or 0), fmt="s" if const else "o", ms=5, color=c, mfc=c, mec="white",
                        elinewidth=0.9, capsize=0)
            below = row.startswith("XSTest") and m == "Character-trained model + tools"
            ax.annotate(SHORT[m], (e["elo_mean"], y), xytext=(4, -9 if below else 4), textcoords="offset points",
                        fontsize=6.5, color=INK)
        for w in (QWEN4[:2], QWEN4[2:]):
            xs = [elo[m]["elo_mean"] for m in w]
            ys = [val(r, CFG[m]) for m in w]
            if None not in ys:
                ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                            arrowprops=dict(arrowstyle="->", color=LGRAY, lw=0.8, shrinkA=5, shrinkB=5))
        ax.set_xlabel("EigenBench Elo (agent, with tools)")
        ax.set_ylabel(ylab)
        style(ax, "both")
    axes[0].legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=0, label="Base weights"),
                            Line2D([], [], marker="o", color=ORANGE, lw=0, label="Character-trained"),
                            Line2D([], [], marker="o", mfc="white", mec=INK2, lw=0, label="no constitution"),
                            Line2D([], [], marker="s", mfc="white", mec=INK2, lw=0, label="constitution prompt")],
                   loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=2)
    save(fig, "fig5_alignment_vs_capability")
else:
    print("skipped fig5: capability rows missing")
