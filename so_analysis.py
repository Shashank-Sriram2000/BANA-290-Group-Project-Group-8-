"""
so_analysis.py — BANA 290 Group 8
===================================
Loads the panel collected by so_data_collector.py and:
    1. Fixes known bad data points (python 2020-11-04 = 100, should be ~1000)
    2. Plots publication-ready time series and tag-level charts
    3. Runs ITS regression (OLS with HAC SEs)
    4. Runs DiD regression (tag FE, tag-clustered SEs)
    5. Exports result tables

Dependencies:
    pip install pandas numpy matplotlib seaborn statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
from pathlib import Path

DATA   = Path("data/so_panel.csv")
PLOTS  = Path("plots")
PLOTS.mkdir(exist_ok=True)

INTERVENTION_DATE = pd.Timestamp("2022-11-30")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

BLUE   = "#2563EB"
GRAY   = "#64748B"
RED    = "#DC2626"
ORANGE = "#F97316"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, parse_dates=["date"])
print(f"Loaded {len(df):,} rows  |  Tags: {df.tag.nunique()}  |  "
      f"Dates: {df.date.min().date()} -> {df.date.max().date()}")

# ── Fix known bad data point ──────────────────────────────────────────────────
# python 2020-11-04: cached as 100 during bad-key run, should be ~1000
bad_mask = (df["tag"] == "python") & (df["date"] == "2020-11-04")
if bad_mask.sum() > 0:
    neighbors = df[
        (df["tag"] == "python") &
        (df["date"].isin([pd.Timestamp("2020-10-28"), pd.Timestamp("2020-11-11")]))
    ]
    fill_val = int(neighbors["question_count"].mean()) if len(neighbors) > 0 else 1000
    df.loc[bad_mask, "question_count"] = fill_val
    print(f"Fixed bad data point: python 2020-11-04 -> {fill_val}")

print(df.head())

# ── 1. Total volume time series ───────────────────────────────────────────────
agg = (
    df.groupby("date")["question_count"]
    .sum()
    .reset_index()
    .rename(columns={"question_count": "total_questions"})
)

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(agg.date, agg.total_questions, lw=1.5, color=BLUE, alpha=0.9,
        label="Weekly questions (all tags)")
ax.axvline(INTERVENTION_DATE, color=RED, ls="--", lw=1.8, label="ChatGPT launch (Nov 2022)")
ax.fill_between(agg.date, agg.total_questions,
                where=agg.date >= INTERVENTION_DATE,
                alpha=0.07, color=RED)
ax.set_title("Stack Overflow — Weekly Question Volume", fontsize=14,
             fontweight="bold", pad=12)
ax.set_ylabel("Questions per week")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=35, ha="right")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(PLOTS / "01_total_volume.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 01_total_volume.png")

# ── 2. Treatment vs control ───────────────────────────────────────────────────
grp = (
    df.groupby(["date", "group"])["question_count"]
    .mean()
    .reset_index()
    .rename(columns={"question_count": "avg_questions"})
)

fig, ax = plt.subplots(figsize=(13, 4.5))
for g, color, label in [
    ("treatment", BLUE, "Treatment (AI-adjacent: Python, ML, NLP, PyTorch, TensorFlow)"),
    ("control",   GRAY, "Control (AI-insulated: SQL, Excel, R, VBA, Bash)"),
]:
    sub = grp[grp.group == g]
    ax.plot(sub.date, sub.avg_questions, lw=1.8, color=color, label=label, alpha=0.9)

ax.axvline(INTERVENTION_DATE, color=RED, ls="--", lw=1.8, label="ChatGPT launch (Nov 2022)")
ax.set_title("Avg Weekly Questions: Treatment vs Control Tags",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Avg questions / week")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=35, ha="right")
ax.legend(framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS / "02_treatment_vs_control.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_treatment_vs_control.png")

# ── 3. % change by tag ────────────────────────────────────────────────────────
pre  = df[df.post_intervention == 0].groupby("tag")["question_count"].mean()
post = df[df.post_intervention == 1].groupby("tag")["question_count"].mean()
pct  = ((post - pre) / pre * 100).sort_values()
grp_map = df.drop_duplicates("tag").set_index("tag")["group"]

bar_colors = [RED if grp_map.get(t) == "treatment" else ORANGE for t in pct.index]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(pct.index, pct.values, color=bar_colors, edgecolor="white", height=0.65)
ax.axvline(0, color="black", lw=0.9)

for bar, val in zip(bars, pct.values):
    ax.text(val - 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", ha="right",
            fontsize=9, color="white", fontweight="bold")

legend_els = [
    Patch(facecolor=RED,    label="Treatment (AI-adjacent)"),
    Patch(facecolor=ORANGE, label="Control (AI-insulated)"),
]
ax.legend(handles=legend_els, framealpha=0.9, fontsize=9)
ax.set_title("% Change in Avg Weekly Questions\nPre vs Post ChatGPT Launch",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("% change")
ax.set_xlim(pct.min() * 1.12, 5)
plt.tight_layout()
plt.savefig(PLOTS / "03_pct_change_by_tag.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 03_pct_change_by_tag.png")

# ── 4. ITS Regression ─────────────────────────────────────────────────────────
its_df = (
    df.groupby(["date", "time_index", "post_intervention", "time_since_intervention"])
    ["question_count"].sum().reset_index()
    .rename(columns={"question_count": "y"})
)
its_df["log_y"] = np.log1p(its_df["y"])

its_model = smf.ols(
    "log_y ~ time_index + post_intervention + time_since_intervention",
    data=its_df
).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

print("\n-- ITS Regression (HAC-robust SEs) --")
print(its_model.summary().tables[1])

# ── 5. DiD Regression ─────────────────────────────────────────────────────────
did_df = df[df["question_count"] > 0].copy()
did_df["log_q"] = np.log(did_df["question_count"])

did_model = smf.ols(
    "log_q ~ post_intervention + did_interaction + C(tag)",
    data=did_df
).fit(cov_type="cluster", cov_kwds={"groups": did_df["tag"]})

print("\n-- DiD Regression (tag FE, tag-clustered SEs) --")
print("Note: is_treatment omitted -- absorbed by tag fixed effects")
key_coefs = did_model.params.filter(["Intercept", "post_intervention", "did_interaction"])
key_ses   = did_model.bse.filter(key_coefs.index)
key_pval  = did_model.pvalues.filter(key_coefs.index)

coef_table = pd.DataFrame({
    "coef":    key_coefs.round(4),
    "std_err": key_ses.round(4),
    "p_value": key_pval.round(4),
})
print(coef_table.to_string())

did_est = key_coefs.get("did_interaction", float("nan"))
print(f"\nDiD estimator (b3) = {did_est:.4f}")
print(f"-> Treatment tags changed {((np.exp(did_est)-1)*100):.1f}% MORE than control post-ChatGPT")

# ── 6. Export ──────────────────────────────────────────────────────────────────
coef_table.to_csv(Path("data") / "did_results.csv")
print("\nSaved did_results.csv")
print("All done. Check plots/ and data/ directories.")
