"""Reproducible synthetic real-estate buyer segmentation pipeline.
Run from project root: python src/train_model.py
All records are synthetic demonstration data, not real customer records.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "real_estate_buyers_synthetic.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
VISUALS = ROOT / "visuals"
MODEL_PATH = ROOT / "models" / "buyer_segmentation_pipeline.joblib"
for folder in (PROCESSED, REPORTS, VISUALS, MODEL_PATH.parent):
    folder.mkdir(parents=True, exist_ok=True)

# 1. Load and clean
raw = pd.read_csv(RAW)
df = raw.drop_duplicates(subset=["client_id"]).copy()
df["country"] = df["region"].map({
    "North America": "United States",
    "Europe": "Germany",
    "Asia Pacific": "India",
    "Middle East": "United Arab Emirates",
    "Latin America": "Brazil"
}).fillna("Other")
for col in ["gender", "referral_channel"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
if df["satisfaction_score"].isna().any():
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
reference_date = pd.Timestamp("2026-11-04")
df["age"] = ((reference_date - df["date_of_birth"]).dt.days / 365.25).round()
df = df.dropna(subset=["age"]).copy()
df["age"] = df["age"].astype(int)
df = df.drop(columns=["date_of_birth"])
df.to_csv(PROCESSED / "real_estate_buyers_clean.csv", index=False)

# 2. Transform features
cat = ["client_type", "region", "country", "acquisition_purpose", "loan_applied", "referral_channel", "income_bracket"]
num = ["age", "satisfaction_score", "property_budget_usd", "units_interested"]
features = cat + num
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
    ("numeric", StandardScaler(), num),
])
X = preprocessor.fit_transform(df[features])
X = X.toarray() if hasattr(X, "toarray") else X

# 3. Compare K-Means cluster counts
silhouette_by_k, inertia_by_k = {}, {}
for k in range(2, 9):
    candidate = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = candidate.fit_predict(X)
    silhouette_by_k[k] = float(silhouette_score(X, labels))
    inertia_by_k[k] = float(candidate.inertia_)
best_k = max(silhouette_by_k, key=silhouette_by_k.get)
model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = model.fit_predict(X)

# 4. Profile and interpret clusters
profile = df.groupby("cluster").agg(
    buyers=("client_id", "count"),
    avg_age=("age", "mean"),
    investment_share=("acquisition_purpose", lambda x: (x == "Investment").mean()),
    corporate_share=("client_type", lambda x: (x == "Corporate").mean()),
    loan_share=("loan_applied", lambda x: (x == "Yes").mean()),
    avg_budget=("property_budget_usd", "mean"),
    avg_units=("units_interested", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
)
names = {}
for cluster, row in profile.iterrows():
    if row.corporate_share > .55 and row.avg_units > 2:
        label = "Corporate Portfolio Buyers"
    elif row.avg_budget > profile.avg_budget.quantile(.75) and row.loan_share < .25:
        label = "Premium / Luxury Buyers"
    elif row.avg_age < profile.avg_age.median() and row.loan_share > .5 and row.investment_share < .5:
        label = "First-Time / Home Buyers"
    else:
        label = "Investment-Oriented Buyers"
    names[int(cluster)] = label

df["buyer_segment"] = df["cluster"].map(names)
df.to_csv(PROCESSED / "real_estate_buyers_segmented.csv", index=False)
profile["buyer_segment"] = [names[int(i)] for i in profile.index]
profile.to_csv(REPORTS / "cluster_profile.csv")
joblib.dump({"preprocessor": preprocessor, "kmeans": model, "features": features, "cluster_names": names}, MODEL_PATH)

# 5. Hierarchical comparison on deterministic sample
sample_size = min(500, len(X))
rng = np.random.default_rng(42)
indices = np.sort(rng.choice(len(X), size=sample_size, replace=False)) if sample_size < len(X) else np.arange(len(X))
hier_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X[indices])
hierarchical_silhouette = float(silhouette_score(X[indices], hier_labels))

# 6. Save evaluation metrics
metrics = {
    "synthetic": True,
    "seed": 42,
    "reference_date_for_age": "2026-11-04",
    "raw_rows": int(len(raw)),
    "raw_columns": int(raw.shape[1]),
    "clean_rows": int(len(df)),
    "clean_columns": int(len(df.columns) - 2),
    "duplicates_removed": int(len(raw) - len(df) - (raw["client_id"].duplicated().sum() - (len(raw) - len(raw.drop_duplicates(subset=["client_id"]))))),
    "best_k": int(best_k),
    "kmeans_silhouette": float(silhouette_by_k[best_k]),
    "hierarchical_sample_silhouette": hierarchical_silhouette,
    "hierarchical_sample_size": int(sample_size),
    "silhouette_by_k": {str(k): v for k, v in silhouette_by_k.items()},
    "inertia_by_k": {str(k): v for k, v in inertia_by_k.items()},
    "cluster_names": {str(k): v for k, v in names.items()},
}
with open(REPORTS / "analysis_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 7. Regenerate model evaluation and summary charts
plt.figure(figsize=(7, 4))
plt.plot(list(inertia_by_k), list(inertia_by_k.values()), marker="o")
plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia"); plt.title("K-Means Elbow Method")
plt.tight_layout(); plt.savefig(VISUALS / "elbow_method.png", dpi=160); plt.close()
plt.figure(figsize=(7, 4))
plt.plot(list(silhouette_by_k), list(silhouette_by_k.values()), marker="o")
plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette score"); plt.title("Silhouette Score by Cluster Count")
plt.tight_layout(); plt.savefig(VISUALS / "silhouette_scores.png", dpi=160); plt.close()
counts = df["buyer_segment"].value_counts()
plt.figure(figsize=(8, 4)); counts.plot(kind="bar")
plt.xlabel("Buyer segment"); plt.ylabel("Buyers"); plt.title("Buyer Segment Distribution")
plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(VISUALS / "segment_distribution.png", dpi=160); plt.close()
plt.figure(figsize=(8, 4))
for label, group in df.groupby("buyer_segment"):
    plt.hist(group["property_budget_usd"], bins=25, alpha=.55, label=label)
plt.xlabel("Property budget (USD)"); plt.ylabel("Frequency"); plt.title("Budget Distribution by Buyer Segment"); plt.legend()
plt.tight_layout(); plt.savefig(VISUALS / "budget_by_cluster.png", dpi=160); plt.close()

print(f"Raw rows: {len(raw)} | cleaned rows: {len(df)} | selected k: {best_k}")
print(f"K-Means silhouette: {silhouette_by_k[best_k]:.3f} | hierarchical sample silhouette: {hierarchical_silhouette:.3f}")
print("Updated processed data, model, metrics, cluster profile and visuals.")
