# Submission QA checklist

## Data and reproducibility
- [x] Synthetic dataset is explicitly identified as synthetic.
- [x] 1,512 raw rows include 12 intentional duplicate client IDs.
- [x] Cleaned dataset contains 1,500 unique buyer IDs.
- [x] Missing values are imputed and age is derived from date of birth using 2026-11-04 as the reference date.
- [x] Training script reruns and regenerates cleaned/segmented data, model, metrics, cluster profile and charts.
- [x] K-Means candidate counts k=2..8 are evaluated; best silhouette in this sample is k=2 (approximately 0.242).
- [x] Hierarchical comparison is reported on a deterministic 500-row sample (silhouette approximately 0.248).

## Before uploading (user must complete)
- [ ] Install dependencies using `pip install -r requirements.txt` in a clean virtual environment.
- [ ] Run `python src/train_model.py` without errors.
- [ ] Run `streamlit run app/streamlit_app.py` and open the local dashboard URL.
- [ ] Test each filter separately and in combination; verify the empty-result message.
- [ ] Test the filtered CSV download.
- [ ] Capture screenshots of the running dashboard if application screenshots are required.
- [ ] Confirm the portal's file-size, file-type and naming rules.
- [ ] Upload the ZIP, report and presentation in the portal's requested fields.

## Interpretation safeguards
- [x] Synthetic extensions (income bracket, property budget and units interested) are disclosed.
- [x] Cluster labels are descriptive interpretations, not verified customer identities.
- [x] Modest silhouette scores are not overstated as strong separation.
- [x] The report states that production use requires representative, consented data and privacy/fairness validation.
