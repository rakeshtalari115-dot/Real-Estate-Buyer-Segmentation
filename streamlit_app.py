from pathlib import Path
import subprocess
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "real_estate_buyers_segmented.csv"

st.set_page_config(page_title="Real Estate Buyer Intelligence", page_icon="🏙️", layout="wide")
st.title("Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Synthetic demonstration dataset | Decision-support prototype | USD")

# Rebuild the processed file automatically if the user launches the app
# before running the training script or accidentally removes the output file.
if not DATA.exists():
    with st.spinner("Preparing the dataset and model for the dashboard..."):
        result = subprocess.run(
            [sys.executable, str(ROOT / "train_model.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        st.error("The data preparation step failed. Open a terminal in the project folder and run: python train_model.py.
        st.code(result.stderr or result.stdout)
        st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

df = load_data()
required_columns = {
    "client_id", "country", "region", "acquisition_purpose", "client_type",
    "loan_applied", "buyer_segment", "age", "property_budget_usd", "units_interested"
}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    st.error("The processed dataset is missing required columns: " + ", ".join(missing_columns))
    st.info("Run `python train_model.py. from the project root, then refresh this page.")
    st.stop()
st.sidebar.header("Buyer filters")
countries = st.sidebar.multiselect("Country", sorted(df.country.dropna().unique()), default=[])
regions = st.sidebar.multiselect("Region", sorted(df.region.dropna().unique()), default=[])
purposes = st.sidebar.multiselect("Acquisition purpose", sorted(df.acquisition_purpose.dropna().unique()), default=[])
types = st.sidebar.multiselect("Client type", sorted(df.client_type.dropna().unique()), default=[])

filtered = df.copy()
if countries: filtered = filtered[filtered.country.isin(countries)]
if regions: filtered = filtered[filtered.region.isin(regions)]
if purposes: filtered = filtered[filtered.acquisition_purpose.isin(purposes)]
if types: filtered = filtered[filtered.client_type.isin(types)]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Buyers", f"{len(filtered):,}")
c2.metric("Investment purpose", f"{(filtered.acquisition_purpose == 'Investment').mean():.1%}" if len(filtered) else "—")
c3.metric("Loan applicants", f"{(filtered.loan_applied == 'Yes').mean():.1%}" if len(filtered) else "—")
c4.metric("Average budget", f"${filtered.property_budget_usd.mean():,.0f}" if len(filtered) else "—")

if filtered.empty:
    st.warning("No buyers match the selected filters.")
    st.stop()

left,right = st.columns(2)
with left:
    st.subheader("Buyer segment distribution")
    counts = filtered.buyer_segment.value_counts().rename_axis("segment").reset_index(name="buyers")
    st.plotly_chart(px.bar(counts,x="segment",y="buyers",color="segment"),use_container_width=True)
with right:
    st.subheader("Purpose by region")
    pivot = filtered.groupby(["region","acquisition_purpose"]).size().reset_index(name="buyers")
    st.plotly_chart(px.bar(pivot,x="region",y="buyers",color="acquisition_purpose",barmode="group"),use_container_width=True)

left,right = st.columns(2)
with left:
    st.subheader("Financing patterns")
    loan = filtered.loan_applied.value_counts().rename_axis("loan_applied").reset_index(name="buyers")
    st.plotly_chart(px.pie(loan,names="loan_applied",values="buyers",hole=.45),use_container_width=True)
with right:
    st.subheader("Budget by buyer segment")
    st.plotly_chart(px.box(filtered,x="buyer_segment",y="property_budget_usd",color="buyer_segment"),use_container_width=True)

st.subheader("Segment profile")
summary = filtered.groupby("buyer_segment").agg(
    buyers=("client_id","count"),
    mean_age=("age","mean"),
    investment_share=("acquisition_purpose",lambda x:(x=="Investment").mean()),
    loan_share=("loan_applied",lambda x:(x=="Yes").mean()),
    average_budget_usd=("property_budget_usd","mean"),
    average_units=("units_interested","mean")
).round(2)
st.dataframe(summary,use_container_width=True)
st.subheader("Filtered buyer records")
st.dataframe(filtered,use_container_width=True)
st.download_button("Download filtered data",filtered.to_csv(index=False).encode("utf-8"),"filtered_buyer_segments.csv","text/csv")
st.info("All records are synthetic. Cluster labels are descriptive interpretations, not verified customer identities or investment advice.")
