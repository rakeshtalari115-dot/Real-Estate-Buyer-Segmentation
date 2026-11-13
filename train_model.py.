"""Reproducible synthetic real-estate buyer segmentation pipeline.
Run from the repository root: python train_model.py
All records are synthetic demonstration data, not real customer records.
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "real_estate_buyers_synthetic.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
VISUALS = ROOT / "visuals"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "buyer_segmentation_pipeline.joblib"
REFERENCE_DATE = pd.Timestamp("2026-11-13")
RANDOM_SEED = 42

for folder in (PROCESSED, REPORTS, VISUALS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

# 1. Load and validate the source data
if not RAW.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW}. Keep the CSV in data/raw/."
    )

raw = pd.read_csv(RAW)
required = {
    "client_id", "date_of_birth", "gender", "client_type", "region",
    "acquisition_purpose", "loan_applied", "referral_channel",
    "income_bracket", "property_budget_usd", "units_interested",
    "satisfaction_score",
}
missing = sorted(required.difference(raw.columns))
if missing:
    raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
if raw.empty:
    raise ValueError("The source dataset contains no buyer records.")

raw_rows = len(raw)
duplicate_rows = int(raw["client_id"].duplicated().sum())
df = raw.drop_duplicates(subset=["client_id"]).copy()

# 2. Clean fields and derive country and age
country_map = {
    "North America": "United States",
    "Europe": "Germany",
    "Asia Pacific": "India",
    "Middle East": "United Arab Emirates",
    "Latin America": "Brazil",
}
df["country"] = df["region"].map(country_map).fillna("Other")

for col in [
    "gender", "client_type", "region", "acquisition_purpose",
    "loan_applied", "referral_channel", "income_bracket",
]:
    df[col] = df[col].fillna("Unknown").astype(str)

for col in ["satisfaction_score", "property_budget_usd", "units_interested"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    median_value = df[col].median()
    df[col] = df[col].fillna(0 if pd.isna(median_value) else median_value)

dates = pd.to_datetime(df["date_of_birth"], errors="coerce")
df["age"] = ((REFERENCE_DATE - dates).dt.days / 365.25).round()
invalid_age_rows = int(df["age"].isna().sum())
df = df.dropna(subset=["age"]).copy()
if len(df) < 3:
    raise ValueError("At least 3 valid buyer records are required for clustering.")
df["age"] = df["age"].astype(int)
df = df.drop(columns=["date_of_birth"])
df.to_csv(PROCESSED / "real_estate_buyers_clean.csv", index=False)
clean_rows = len(df)
clean_columns = len(df.columns)

# 3. Encode categorical fields and scale numeric fields
cat = [
    "client_type", "region", "country", "acquisition_purpose",
    "loan_applied", "referral_channel", "income_bracket",
]
num = [
    "age", "satisfaction_score", "property_budget_usd", "units_interested",
]
features = cat + num
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
        ("numeric", StandardScaler(), num),
    ]
)
X = preprocessor.fit_transform(df[features])
X = X.toarray() if hasattr(X, "toarray") else X

# 4. Compare valid K-Means cluster counts
max_k = min(8, len(df) - 1)
if max_k < 2:
    raise ValueError("Not enough records to compare clustering models.")

silhouette_by_k = {}
inertia_by_k = {}
for k in range(2, max_k + 1):
    candidate = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
    candidate_labels = candidate.fit_predict(X)
    if len(np.unique(candidate_labels)) < 2:
        continue
    silhouette_by_k[k] = float(silhouette_score(X, candidate_labels))
    inertia_by_k[k] = float(candidate.inertia_)

if not silhouette_by_k:
    raise ValueError("Could not calculate a valid silhouette score for this dataset.")

best_k = max(silhouette_by_k, key=silhouette_by_k.get)
model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=20)
df["cluster"] = model.fit_predict(X)

# 5. Profile and label the resulting clusters
profile = df.groupby("cluster").agg(
    buyers=("client_id", "count"),
    avg_age=("age", "mean"),
    investment_share=("acquisition_purpose", lambda values: (values == "Investment").mean()),
    corporate_share=("client_type", lambda values: (values == "Corporate").mean()),
    loan_share=("loan_applied", lambda values: (values == "Yes").mean()),
    avg_budget=("property_budget_usd", "mean"),
    avg_units=("units_interested", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
)

names = {}
for cluster, row in profile.iterrows():
    if row.corporate_share > 0.55 and row.avg_units > 2:
        label = "Corporate Portfolio Buyers"
    elif row.avg_budget > profile["avg_budget"].quantile(0.75) and row.loan_share < 0.25:
        label = "Premium / Luxury Buyers"
    elif (
        row.avg_age < profile["avg_age"].median()
        and row.loan_share > 0.5
        and row.investment_share < 0.5
    ):
        label = "First-Time / Home Buyers"
    else:
        label = "Investment-Oriented Buyers"

    if label in names.values():
        label = f"{label} (Segment {int(cluster) + 1})"
    names[int(cluster)] = label

df["buyer_segment"] = df["cluster"].map(names)
df.to_csv(PROCESSED / "real_estate_buyers_segmented.csv", index=False)
profile["buyer_segment"] = [names[int(cluster)] for cluster in profile.index]
profile.to_csv(REPORTS / "cluster_profile.csv")
joblib.dump(
    {
        "preprocessor": preprocessor,
        "kmeans": model,
        "features": features,
        "cluster_names": names,
    },
    MODEL_PATH,
)

# 6. Compare with hierarchical clustering on a reproducible sample
sample_size = min(500, len(X))
rng = np.random.default_rng(RANDOM_SEED)
if sample_size < len(X):
    indices = np.sort(rng.choice(len(X), size=sample_size, replace=False))
else:
    indices = np.arange(len(X))
hier_labels = AgglomerativeClustering(
    n_clusters=best_k, linkage="ward"
).fit_predict(X[indices])
hierarchical_silhouette = float(silhouette_score(X[indices], hier_labels))

# 7. Save evaluation metrics
metrics = {
    "synthetic": True,
    "seed": RANDOM_SEED,
    "reference_date_for_age": REFERENCE_DATE.strftime("%Y-%m-%d"),
    "raw_rows": int(raw_rows),
    "raw_columns": int(raw.shape[1]),
    "duplicate_rows_removed": duplicate_rows,
    "invalid_age_rows_removed": invalid_age_rows,
    "clean_rows": int(clean_rows),
    "clean_columns": int(clean_columns),
    "best_k": int(best_k),
    "kmeans_silhouette": float(silhouette_by_k[best_k]),
    "hierarchical_sample_silhouette": hierarchical_silhouette,
    "hierarchical_sample_size": int(sample_size),
    "silhouette_by_k": {str(k): value for k, value in silhouette_by_k.items()},
    "inertia_by_k": {str(k): value for k, value in inertia_by_k.items()},
    "cluster_names": {str(k): value for k, value in names.items()},
}
with open(REPORTS / "analysis_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

# 8. Regenerate evaluation and summary charts
plt.figure(figsize=(7, 4))
plt.plot(list(inertia_by_k), list(inertia_by_k.values()), marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.tight_layout()
plt.savefig(VISUALS / "elbow_method.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(list(silhouette_by_k), list(silhouette_by_k.values()), marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score by Cluster Count")
plt.tight_layout()
plt.savefig(VISUALS / "silhouette_scores.png", dpi=160)
plt.close()

counts = df["buyer_segment"].value_counts()
plt.figure(figsize=(8, 4))
counts.plot(kind="bar")
plt.xlabel("Buyer segment")
plt.ylabel("Buyers")
plt.title("Buyer Segment Distribution")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(VISUALS / "segment_distribution.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 4))
for label, group in df.groupby("buyer_segment"):
    plt.hist(group["property_budget_usd"], bins=25, alpha=0.55, label=label)
plt.xlabel("Property budget (USD)")
plt.ylabel("Frequency")
plt.title("Budget Distribution by Buyer Segment")
plt.legend()
plt.tight_layout()
plt.savefig(VISUALS / "budget_by_cluster.png", dpi=160)
plt.close()

print(f"Raw rows: {raw_rows} | unique rows: {len(raw.drop_duplicates(subset=['client_id']))}")
print(f"Clean rows: {clean_rows} | selected k: {best_k}")
print(
    f"K-Means silhouette: {silhouette_by_k[best_k]:.3f} | "
    f"hierarchical sample silhouette: {hierarchical_silhouette:.3f}"
)
print("Updated processed data, model, metrics, cluster profile and visuals.")
