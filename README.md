# EstateIQ — Indian Real Estate Investment Analyzer

An interactive Streamlit application using the supplied `india_housing_prices.csv` dataset (250,000 rows, 23 columns).

## Features
- Regression model: predicts asking price in ₹ lakh from property characteristics.
- Classification model: screens for listings priced at or below the city median price per sq ft.
- Property valuation and investment screening forms.
- Market and locality explorer with interactive Plotly charts.
- Add new property records to the active session; dashboard and exports update immediately.
- Model diagnostics and CSV/JSON exports.

## Project files
- `app.py` — Streamlit application and model pipelines.
- `india_housing_prices.csv` — supplied dataset.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes local environment/cache files.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository (e.g. `estateiq-real-estate-ai`).
2. Upload `app.py`, `requirements.txt`, and `india_housing_prices.csv` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository, branch, and `app.py`.
5. Deploy. The first launch trains both ML models; allow a little time for training.

## Model design and important limitations
### Regression
`HistGradientBoostingRegressor` predicts `Price_in_Lakhs`. The training pipeline uses ordinal encoding for categorical fields and median/mode imputation. Up to 90,000 randomly selected rows are used to keep Streamlit Cloud startup practical. The app reports held-out R², MAE and RMSE.

### Classification
`HistGradientBoostingClassifier` predicts a *relative-value screening label*: whether a record's price per sq ft is at or below its city's median price per sq ft. The classifier does not predict appreciation, rental yield, default risk, resale time or guaranteed returns. The target is a proxy derived from the supplied data.

### Live-added records
New properties update the active Streamlit session's dataset and exports immediately. Streamlit Community Cloud sessions are ephemeral: records added in the UI are not permanently written back to the repository CSV or shared between users. To make records persistent, connect a database or add a controlled write-back workflow. The models remain trained on the original data until the user selects **Retrain models on current session dataset**.

### Due diligence
The supplied data does not include verified transaction prices, rental income, vacancy, maintenance, taxes, financing, RERA/legal status, title verification, construction quality or historical appreciation. Model estimates are indicative only. Independently verify every property and consult qualified financial/legal professionals before investing.

## Dataset note
The source includes `Price_per_SqFt` values stored in lakh-per-sq-ft scale (for example, approximately 0.10 corresponds to ₹10,000/sq ft). The app also computes rupees per sq ft from `Price_in_Lakhs * 100000 / Size_in_SqFt` for display.
