"""Figures for the final report (Figs. 2-4; Fig. 1 is figures/hard_suite_examples.png from src.evaluate).
Usage: make_figures.py PUBLIC_LB_CSV PRIVATE_LB_CSV OUR_SUBMISSIONS_CSV OUT_DIR. The three CSVs come from the Kaggle
CLI after the competition closed (competitions leaderboard --download / --show -v, competitions submissions -v)."""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b4b3ae", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220,
                     "legend.frameon": False})
pub_csv, prv_csv, subs_csv, out = sys.argv[1:5]
out = Path(out)


def progression():
    runs = [("A", "baseline\n45k", .875, .865), ("B", "baseline\n50k", .892, .873), ("D", "KD", .891, .865),
            ("F", "function\nmatching", .901, .870), ("H", "800\nepochs", .897, .873),
            ("I", "diverse\nteachers", .900, .876), ("J", "+ balanced\ndecoding", .902, .884),
            ("M", "+ hard\naugment", .910, .893), ("N", "PreAct +\nhard aug", .909, .886)]
    x = range(len(runs))
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for col, key, label in ((2, BLUE, "Public (1,000 images)"), (3, ORANGE, "Private (1,000 images)")):
        y = [100 * r[col] for r in runs]
        ax.plot(x, y, color=key, linewidth=2, marker="o", markersize=6, label=label,
                markeredgecolor="white", markeredgewidth=1)
        for i in (0, 7):
            ax.annotate(f"{y[i]:.1f}", (i, y[i]), textcoords="offset points", xytext=(0, 7 if col == 2 else -13),
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), [f"Run {r[0]}\n{r[1]}" for r in runs], fontsize=7.5)
    ax.set_ylabel("Accuracy on hidden set (%)")
    ax.set_ylim(85.5, 92)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_progression.png")


def hard_suite():
    rows = [("clean", 97.92, 98.00), ("lowres 16 px", 80.42, 94.74), ("background 0.7x", 82.68, 95.58),
            ("zoom out 0.7x", 86.58, 96.24), ("confusable blend", 90.70, 91.14), ("zoom in 1.3x", 93.20, 96.44),
            ("rotation 20 deg", 93.82, 94.22), ("lowres 24 px", 94.90, 96.96), ("perspective 0.4", 95.28, 96.02),
            ("500 most ambiguous", 81.40, 81.80)]
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    h = 0.36
    y = range(len(rows))
    for k, (key, label) in enumerate(((GRAY, "Run I twin (no hard augmentation)"),
                                      (BLUE, "Run M twin (on-the-fly hard augmentation)"))):
        vals = [r[1 + k] for r in rows]
        ys = [i + (k - 0.5) * (h + 0.04) for i in y]
        ax.barh(ys, vals, height=h, color=key, label=label)
        for yy, v in zip(ys, vals):
            ax.text(v + 0.3, yy, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(75, 100.5)
    ax.set_xlabel("Accuracy on the 5,000 held-out images (%), flip+shift TTA")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig3_hard_suite.png")


def leaderboard():
    pub = pd.read_csv(pub_csv, encoding="utf-8-sig").rename(columns={"TeamId": "teamId", "Score": "pub"})
    prv = pd.read_csv(prv_csv).rename(columns={"score": "prv"})
    df = pub[["teamId", "TeamName", "pub"]].merge(prv[["teamId", "prv"]], on="teamId")
    df[["pub", "prv"]] *= 100
    gap = (df.prv - df.pub).median()
    subs = pd.read_csv(subs_csv)
    bal = {58014367, 58014369, 58031983, 58031987, 58063597, 58076534, 58078999}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    lo, hi = 82, 94
    for ax in (a, b):
        ax.plot([lo, hi], [lo + gap, hi + gap], color=MUTED, linewidth=1, linestyle="--",
                label=f"Median private - public ({gap:+.1f})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo - 2, hi - 1)
        ax.set_xlabel("Public accuracy (%)")
        ax.grid(color="#eeede9", linewidth=0.5)
        ax.set_axisbelow(True)
    a.set_ylabel("Private accuracy (%)")
    inside = df[(df.pub < hi) & (df.prv < hi - 1)]
    others = inside[inside.TeamName != "Survey Corps"]
    us = df[df.TeamName == "Survey Corps"].iloc[0]
    a.scatter(others.pub, others.prv, s=16, color=GRAY, edgecolor="white", linewidth=0.5, label="Other teams")
    a.scatter([us.pub], [us.prv], s=42, color=BLUE, edgecolor="white", linewidth=1, zorder=3, label="Survey Corps")
    a.annotate("Survey Corps:\npublic 18th, private 6th", (us.pub, us.prv), xytext=(91.0, 83.4),
               textcoords="data", fontsize=7.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.set_title("(a) All teams (two outliers off-chart)", fontsize=8.5)
    a.legend(loc="upper left", fontsize=7.5, handletextpad=0.2)
    s = subs.assign(pub=subs.publicScore * 100, prv=subs.privateScore * 100, bal=subs.ref.isin(bal))
    b.scatter(s[~s.bal].pub, s[~s.bal].prv, s=22, color=ORANGE, edgecolor="white", linewidth=0.5,
              label="No class balancing (14)")
    b.scatter(s[s.bal].pub, s[s.bal].prv, s=26, color=BLUE, edgecolor="white", linewidth=0.5, marker="s",
              label="Balanced decoding (7)")
    at = lambda ref: s[s.ref == ref].iloc[0]
    for src, dst in ((57966136, 58014369), (57991519, 58014367)):
        b.annotate("", (at(dst).pub, at(dst).prv), (at(src).pub, at(src).prv),
                   arrowprops=dict(arrowstyle="->", color=INK, lw=0.8, shrinkA=4, shrinkB=4))
    for x, y, name in ((89.6, 86.95, "H"), (90.06, 87.5, "I"), (89.17, 88.85, "K"), (90.03, 88.45, "J")):
        b.text(x, y, name, fontsize=8)
    b.annotate("Run M", (at(58031983).pub, at(58031983).prv), xytext=(5, 2), textcoords="offset points",
               fontsize=7.5)
    b.set_xlim(87, 92)
    b.set_ylim(84.5, 90.5)
    b.set_title("(b) Our 21 submissions (zoomed)", fontsize=8.5)
    b.legend(loc="upper left", fontsize=7.5, handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig4_public_private.png")


progression()
hard_suite()
leaderboard()
