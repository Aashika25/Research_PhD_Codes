import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

DATA_DIR   = "."          
SPLIT_FRAC = 0.5          
ROLL_WIN   = 200       
SEED       = 42
np.random.seed(SEED)

print("Loading data …")
ratings = pd.read_csv(
    r"data/ratings.dat",
    sep="::", engine="python",
    names=["UserID","MovieID","Rating","Timestamp"],
)
movies = pd.read_csv(
    r"data/movies.dat",
    sep="::", engine="python", encoding="latin-1",
    names=["MovieID","Title","Genres"],
)

ratings["Timestamp"] = pd.to_datetime(ratings["Timestamp"], unit="s")
ratings = ratings.sort_values("Timestamp").reset_index(drop=True)

print(f"  Ratings : {len(ratings):,}")
print(f"  Date range: {ratings.Timestamp.min().date()} → {ratings.Timestamp.max().date()}")

user_stats = ratings.groupby("UserID")["Rating"].agg(
    user_mean_rating="mean", user_num_ratings="count"
).reset_index()

# movie-level aggregates
movie_stats = ratings.groupby("MovieID")["Rating"].agg(
    movie_mean_rating="mean", movie_num_ratings="count"
).reset_index()

# genre one-hot (top genres)
genre_dummies = movies["Genres"].str.get_dummies("|")
top_genres    = genre_dummies.sum().nlargest(6).index.tolist()
movies_enc    = pd.concat([movies[["MovieID"]], genre_dummies[top_genres]], axis=1)

df = (ratings
      .merge(user_stats,  on="UserID",  how="left")
      .merge(movie_stats, on="MovieID", how="left")
      .merge(movies_enc,  on="MovieID", how="left"))

df[top_genres] = df[top_genres].fillna(0)

FEATURES = ["user_mean_rating", "user_num_ratings",
            "movie_mean_rating", "movie_num_ratings"] + top_genres

split_idx = int(len(df) * SPLIT_FRAC)
split_ts  = df.iloc[split_idx]["Timestamp"]
print(f"\nTemporal split at: {split_ts.date()}  (index {split_idx:,})")

phase1 = df.iloc[:split_idx].copy()
phase2 = df.iloc[split_idx:].copy()
print(f"  Phase-1 (early): {len(phase1):,} ratings  "
      f"({phase1.Timestamp.min().date()} → {phase1.Timestamp.max().date()})")
print(f"  Phase-2 (late) : {len(phase2):,} ratings  "
      f"({phase2.Timestamp.min().date()} → {phase2.Timestamp.max().date()})")

print("\n── Data Drift (KS-test per feature) ──")
ks_results = {}
for feat in FEATURES:
    stat, pval = stats.ks_2samp(phase1[feat].dropna(), phase2[feat].dropna())
    ks_results[feat] = {"KS": stat, "p": pval, "drift": pval < 0.05}
    flag = "DRIFT" if pval < 0.05 else "ok"
    print(f"  {feat:<28} KS={stat:.4f}  p={pval:.4e}  [{flag}]")

continuous_feats = ["user_mean_rating", "user_num_ratings",
                    "movie_mean_rating", "movie_num_ratings"]
n_feats = len(continuous_feats)

fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
fig1.suptitle(
    "Figure 1 – Data Drift: Feature Distribution Shift Across Temporal Phases",
    fontsize=14, fontweight="bold", y=1.01,
)
axes = axes.flatten()

colours = {"Phase-1 (early)": "#3B82F6", "Phase-2 (late)": "#EF4444"}

for i, feat in enumerate(continuous_feats):
    ax = axes[i]
    p1_vals = phase1[feat].dropna()
    p2_vals = phase2[feat].dropna()

    # clip user_num_ratings for readability
    clip = None
    if "num_ratings" in feat:
        clip = p1_vals.quantile(0.97)
        p1_vals = p1_vals.clip(upper=clip)
        p2_vals = p2_vals.clip(upper=clip)

    bins = np.linspace(min(p1_vals.min(), p2_vals.min()),
                       max(p1_vals.max(), p2_vals.max()), 40)

    ax.hist(p1_vals, bins=bins, density=True, alpha=0.55,
            color=colours["Phase-1 (early)"], label="Phase-1 (early)")
    ax.hist(p2_vals, bins=bins, density=True, alpha=0.55,
            color=colours["Phase-2 (late)"],  label="Phase-2 (late)")

    res = ks_results[feat]
    drift_label = "⚠ DRIFT" if res["drift"] else "✓ stable"
    ax.set_title(
        f"{feat}\nKS={res['KS']:.4f}, p={res['p']:.2e}  {drift_label}",
        fontsize=10,
    )
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    if clip:
        ax.set_xlabel(f"{feat}  (clipped at 97th pct)")

fig1.tight_layout()
fig1.savefig(r"Image/fig1_data_drift.png", dpi=150, bbox_inches="tight")
print("\nSaved → fig1_data_drift.png")

X_p1 = phase1[FEATURES].values
y_p1 = phase1["Rating"].values
X_p2 = phase2[FEATURES].values
y_p2 = phase2["Rating"].values

scaler = StandardScaler().fit(X_p1)
X_p1s  = scaler.transform(X_p1)
X_p2s  = scaler.transform(X_p2)

# train on Phase-1
model = Ridge(alpha=1.0)
model.fit(X_p1s, y_p1)

# predictions on Phase-2 (no retraining yet)
pred_p2_old = model.predict(X_p2s)

# rolling MAE – before retraining
roll_mae_before = []
for start in range(0, len(y_p2) - ROLL_WIN, ROLL_WIN // 2):
    end  = start + ROLL_WIN
    roll_mae_before.append(mean_absolute_error(y_p2[start:end], pred_p2_old[start:end]))

# retrain on Phase-2 first 30 % (simulates periodic retraining)
retrain_cut = int(len(phase2) * 0.30)
X_rt  = np.vstack([X_p1s, X_p2s[:retrain_cut]])
y_rt  = np.concatenate([y_p1, y_p2[:retrain_cut]])
model_retrained = Ridge(alpha=1.0).fit(X_rt, y_rt)

# rolling MAE – after retraining (applied to the remaining 70 % of Phase-2)
X_p2s_post  = X_p2s[retrain_cut:]
y_p2_post   = y_p2[retrain_cut:]
pred_p2_new = model_retrained.predict(X_p2s_post)

roll_mae_after = []
for start in range(0, len(y_p2_post) - ROLL_WIN, ROLL_WIN // 2):
    end = start + ROLL_WIN
    roll_mae_after.append(mean_absolute_error(y_p2_post[start:end], pred_p2_new[start:end]))

# Phase-1 rolling MAE (reference)
pred_p1 = model.predict(X_p1s)
roll_mae_p1 = []
for start in range(0, len(y_p1) - ROLL_WIN, ROLL_WIN // 2):
    end = start + ROLL_WIN
    roll_mae_p1.append(mean_absolute_error(y_p1[start:end], pred_p1[start:end]))

print("\n── Concept Drift (MAE summary) ──")
print(f"  Phase-1 mean MAE  (training period)      : {np.mean(roll_mae_p1):.4f}")
print(f"  Phase-2 mean MAE  (before retraining)    : {np.mean(roll_mae_before):.4f}")
print(f"  Phase-2 mean MAE  (after  retraining)    : {np.mean(roll_mae_after):.4f}")

fig2, ax = plt.subplots(figsize=(14, 6))

# x-axis: rolling window index → treated as sequential time proxy
n_p1     = len(roll_mae_p1)
n_before = len(roll_mae_before)
n_after  = len(roll_mae_after)

x_p1     = np.arange(n_p1)
x_before = np.arange(n_p1, n_p1 + n_before)
x_after  = np.arange(n_p1 + n_before, n_p1 + n_before + n_after)

ax.plot(x_p1,     roll_mae_p1,     color="#3B82F6", lw=2,   label="Phase-1 (training, Ridge)")
ax.plot(x_before, roll_mae_before, color="#EF4444", lw=2,   label="Phase-2 before retraining")
ax.plot(x_after,  roll_mae_after,  color="#22C55E", lw=2,   label="Phase-2 after retraining")

# shade regions
ax.axvspan(x_p1[0],     x_p1[-1],     alpha=0.07, color="#3B82F6")
ax.axvspan(x_before[0], x_before[-1], alpha=0.07, color="#EF4444")
ax.axvspan(x_after[0],  x_after[-1],  alpha=0.07, color="#22C55E")

# vertical lines
ax.axvline(n_p1,              color="grey",    lw=1.5, ls="--")
ax.axvline(n_p1 + n_before,   color="grey",    lw=1.5, ls="--")

ax.text(n_p1 + 0.5,           ax.get_ylim()[0] + 0.002, "Phase split", fontsize=8, color="grey")
ax.text(n_p1 + n_before + 0.5, ax.get_ylim()[0] + 0.002, "Retrain",    fontsize=8, color="grey")

# moving-average smoothing overlay
def smooth(arr, w=5):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

sw = 5
if len(roll_mae_before) > sw:
    ax.plot(x_before[sw-1:], smooth(roll_mae_before, sw),
            color="#991B1B", lw=2.5, ls="-", alpha=0.8, label="Trend (before, MA-5)")
if len(roll_mae_after) > sw:
    ax.plot(x_after[sw-1:], smooth(roll_mae_after, sw),
            color="#15803D", lw=2.5, ls="-", alpha=0.8, label="Trend (after, MA-5)")

ax.set_title(
    "Figure 2 – Concept Drift: Rolling MAE (Window=%d ratings)\n"
    "Ridge Regression Degradation & Post-Retraining Recovery" % ROLL_WIN,
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Rolling Window Index (time proxy →)")
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

fig2.tight_layout()
fig2.savefig(r"Image/fig2_concept_drift.png", dpi=150, bbox_inches="tight")
print("Saved fig2_concept_drift.png")

plt.show()