# Real Estate Buyer Segmentation & Investment Profiling

## Project status
A reproducible end-to-end demonstration project using **synthetic data**. It is not real Parcl customer data and must not be represented as such.

## Quick start
Recommended: Python 3.10–3.12.

1. Open a terminal in this project folder.
2. Create and activate a virtual environment (`python -m venv .venv`).
3. Install dependencies: `pip install -r requirements.txt`.
4. Rebuild all processed data, the trained model, evaluation metrics, cluster profile and charts: `python src/train_model.py`.
5. Launch the dashboard: `streamlit run app/streamlit_app.py`.
6. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

If the dashboard reports that the segmented CSV is missing, run step 4 first.

## Dataset
- Raw file: `data/raw/real_estate_buyers_synthetic.csv`
- Processed file: `data/processed/real_estate_buyers_clean.csv`
- Segmented file: `data/processed/real_estate_buyers_segmented.csv`
- 1,512 raw rows including 12 deliberately duplicated records; 1,500 unique buyer records after deduplication.
- Additional synthetic fields (income bracket, property budget, units interested) are included because the brief's original fields do not contain income or investment size.

## Modeling
Categorical variables are one-hot encoded; numeric variables are standardized. K-Means is evaluated for k=2..8 using inertia and silhouette score. The best silhouette in this generated run is at k=2 (score=0.242). Hierarchical clustering is used as a sample-based validation comparison (silhouette=0.248).

## Important interpretation note
The selected k is data-driven and is not forced to match the four illustrative segment names in the project brief. This synthetic sample currently separates into two broad behavioral groups. If a business requires four operational personas, evaluate k=4 as a business-design option and compare stability, interpretability and segment usefulness before deployment.

## Files
- `src/train_model.py`: reproducible preprocessing, model evaluation, cluster profiling and chart-generation pipeline
- `app/streamlit_app.py`: interactive dashboard
- `models/`: fitted preprocessing and clustering model
- `reports/`: data dictionary, cluster profile, metrics, final report and slides
- `visuals/`: evaluation and segment charts

## Limitations
Synthetic distributions were designed for demonstration. No real-world inference, financial suitability assessment, or investment recommendation should be made from these records. In production, use consented, representative data, fairness review, privacy controls, temporal validation and stakeholder approval.


## Final submission checks
- Keep the synthetic-data notice in the report, presentation and dashboard.
- Submit the ZIP together with the PDF report and presentation if the portal requests separate uploads.
- Before submission, run the Quick start commands on your own computer and test each dashboard filter and the CSV export.
- Take genuine screenshots of the running dashboard and add them to your report or presentation if the portal expects application evidence.
- Do not describe the synthetic results as actual real-estate market findings or actual company customer data.

## Easiest Windows launch (recommended)
1. Install **Python 3.11 (64-bit)** from the official Python website and select **Add Python to PATH** during installation.
2. Extract the entire ZIP file into a normal folder (do not run it from inside the ZIP preview).
3. Make sure the computer has an internet connection for the first-time dependency installation.
4. Double-click `START_PROJECT.bat`. It creates a local virtual environment, installs the packages, rebuilds the analysis outputs, and launches the dashboard.
5. Keep the black terminal window open while using the dashboard. Open the local URL it prints (normally `http://localhost:8501`).
6. On later runs, double-click `RUN_DASHBOARD.bat`.

## Troubleshooting
- **`python` or `py` is not recognized:** install Python 3.11 (64-bit), enable “Add Python to PATH”, close and reopen Command Prompt, then double-click `START_PROJECT.bat` again.
- **Package installation fails:** check internet access, then retry `START_PROJECT.bat`. The first launch needs to download packages.
- **Port 8501 is busy:** in the terminal run `python -m streamlit run app/streamlit_app.py --server.port 8502` after activating `.venv`.
- **Dashboard data file missing:** run `python src/train_model.py` from the extracted project folder. The dashboard also attempts to rebuild it automatically.
- **Browser does not open automatically:** copy the `Local URL` shown in the terminal into Chrome or Edge.
- **Do not close the terminal:** the dashboard stops when its Streamlit process is closed.
