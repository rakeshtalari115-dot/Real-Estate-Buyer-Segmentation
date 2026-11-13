
# Real Estate Buyer Segmentation & Investment Profiling

## Project Overview
This project uses machine learning to segment real-estate buyers and analyze their investment profiles.

**Note:** All records are synthetic demonstration data, not real customer records.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Plotly
- Joblib

## Project Structure
- `data/raw/` – Original synthetic dataset
- `data/processed/` – Cleaned and segmented datasets
- `train_model.py` – Data cleaning, clustering and model training
- `streamlit_app.py` – Interactive dashboard
- `requirements.txt` – Required Python packages
- `models/` – Saved machine learning model
- `reports/` – Metrics and cluster profiles
- `visuals/` – Generated charts

## How to Run the Project

### 1. Install Python
Use Python 3.10 or 3.11 (64-bit).

### 2. Install dependencies
Open Command Prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

### 5. Open the dashboard
Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Machine Learning Methods
- Data cleaning and preprocessing
- One-hot encoding
- Standard scaling
- K-Means clustering
- Silhouette score evaluation
- Hierarchical clustering comparison

## Dashboard Features
- Buyer filters by country and region
- Buyer segment distribution
- Investment purpose analysis
- Financing patterns
- Budget analysis
- Buyer records table
- Filtered data download

## Important Limitations
This project uses synthetic data for educational demonstration. Results are not actual customer insights, financial advice, or verified market findings.
