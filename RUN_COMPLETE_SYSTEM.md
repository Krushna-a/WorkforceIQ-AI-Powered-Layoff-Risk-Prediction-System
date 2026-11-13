# 🚀 WorkforceIQ - Complete Usage Guide

## Quick Start

### Step 1: Setup Environment (5 minutes)

```powershell
# Navigate to project
cd "c:\Code\My_Projects\AIML\Layoff Prediction"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Enrichment (10 minutes)

```powershell
# Run data enrichment pipeline
python data\data_enrichment.py
```

**Expected Output:**
- `data/processed/layoffs_enriched.csv`
- `data/processed/layoffs_training_data.csv`

### Step 3: Train ML Model (5 minutes)

```powershell
# Start Jupyter
jupyter notebook

# Open: notebooks/model_training.ipynb
# Run all cells (Kernel -> Restart & Run All)
```

**Expected Output:**
- `backend/models/layoff_predictor.joblib`
- `backend/models/shap_explainer.joblib`
- `backend/models/feature_list.joblib`
- `backend/models/model_info.json`
- Several PNG visualization files

### Step 4: Start Backend API

Open **Terminal 1**:

```powershell
cd "c:\Code\My_Projects\AIML\Layoff Prediction"
.\venv\Scripts\activate
cd backend
uvicorn api.main:app --reload
```

**Verify:** Open http://localhost:8000 in browser
- Should see: `{"message": "WorkforceIQ API", ...}`

**Test API:**
```powershell
# In another terminal
curl http://localhost:8000/health
curl http://localhost:8000/api/predict/AAPL
```

### Step 5: Start Frontend Dashboard

Open **Terminal 2** (keep API running):

```powershell
cd "c:\Code\My_Projects\AIML\Layoff Prediction"
.\venv\Scripts\activate
streamlit run frontend\app.py
```

**Verify:** Browser opens automatically at http://localhost:8501

---

## 🎯 Using the System

### Web Dashboard

1. **Enter Ticker**: Type a stock ticker (e.g., AAPL, GOOGL, TSLA)
2. **Click Analyze**: Wait 5-10 seconds for analysis
3. **View Results**:
   - Risk gauge (0-100%)
   - Risk level (LOW/MEDIUM/HIGH)
   - Top contributing factors (SHAP)
   - Financial metrics
   - Historical trends
   - Key insights

### API Endpoints

**Health Check:**
```bash
GET http://localhost:8000/health
```

**Predict Risk:**
```bash
GET http://localhost:8000/api/predict/AAPL
```

Response:
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "risk_score": 0.23,
  "risk_level": "LOW",
  "top_factors": [...],
  "confidence": 0.87,
  "financial_metrics": {...}
}
```

**List Companies:**
```bash
GET http://localhost:8000/api/companies
```

---

## 🧪 Testing Examples

### Good Examples (Low Risk):
- **AAPL** (Apple) - Strong financials, low risk
- **MSFT** (Microsoft) - Stable, profitable
- **GOOGL** (Google) - Strong revenue growth

### Medium Risk Examples:
- **UBER** - Growth but not profitable
- **LYFT** - Competitive pressure

### Higher Risk Examples:
- **SNAP** - Revenue challenges
- Companies with declining revenue or high debt

---

## 📊 Understanding Results

### Risk Score
- **0-40%**: LOW risk - stable financials
- **40-70%**: MEDIUM risk - monitor closely
- **70-100%**: HIGH risk - concerning indicators

### SHAP Values
- **Positive**: Increases layoff risk
- **Negative**: Decreases layoff risk
- **Magnitude**: Strength of impact

### Key Factors
1. **Revenue Growth**: Declining = risk
2. **Debt-to-Cash Ratio**: High = risk
3. **Net Income Change**: Declining = risk
4. **Profitability**: Unprofitable = risk
5. **Cash Flow**: Negative = risk

---

## ⚙️ Configuration

### Change Risk Thresholds

Edit `config.py`:

```python
PREDICTION_CONFIG = {
    "high_risk_threshold": 0.7,    # 70%
    "medium_risk_threshold": 0.4,  # 40%
}
```

### Change API Port

Edit `backend/api/main.py` (bottom):

```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port here
```

Also update `frontend/app.py`:

```python
API_URL = "http://localhost:8000"  # Update if port changed
```

### Change Lookback Period

Edit `data/data_enrichment.py`:

```python
enricher = FinancialDataEnricher(lookback_months=6)  # Change to 3, 9, 12, etc.
```

---

## 🐛 Troubleshooting

### Issue: "Model not loaded"
**Solution:**
- Run the Jupyter notebook first
- Ensure `.joblib` files exist in `backend/models/`

### Issue: API connection error
**Solution:**
- Check API is running: http://localhost:8000/health
- Verify port 8000 is not in use
- Check firewall settings

### Issue: yfinance timeout
**Solution:**
- Check internet connection
- Some tickers may not have data
- Try a different ticker

### Issue: Empty predictions
**Solution:**
- Ensure model is trained
- Check feature names match
- Verify financial data fetched successfully

### Issue: Streamlit won't start
**Solution:**
```powershell
# Kill any existing streamlit processes
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process

# Restart
streamlit run frontend\app.py
```

---

## 📁 File Checklist

Before running, ensure these exist:

### Data Files:
- [ ] `data/processed/layoffs_training_data.csv`

### Model Files:
- [ ] `backend/models/layoff_predictor.joblib`
- [ ] `backend/models/shap_explainer.joblib`
- [ ] `backend/models/feature_list.joblib`
- [ ] `backend/models/model_info.json`

### Code Files:
- [ ] `backend/api/main.py`
- [ ] `frontend/app.py`
- [ ] `data/data_enrichment.py`
- [ ] `notebooks/model_training.ipynb`

---

## 🔄 Retraining Model

If you want to retrain with new data:

1. Update `data/processed/layoffs_training_data.csv`
2. Open `notebooks/model_training.ipynb`
3. Run all cells
4. Restart API (models reload automatically)

---

## 🎓 System Architecture

```
User Browser
     ↓
Streamlit Dashboard (Port 8501)
     ↓
FastAPI Backend (Port 8000)
     ↓
┌─────────────┬──────────────┐
│   XGBoost   │   yfinance   │
│    Model    │   API Call   │
└─────────────┴──────────────┘
     ↓              ↓
SHAP Explainer  Financial Data
     ↓              ↓
  Prediction + Top Factors
```

---

## ✅ Complete System Checklist

- [ ] Environment setup complete
- [ ] Data enrichment successful
- [ ] Model trained (ROC-AUC > 0.6)
- [ ] API running (http://localhost:8000)
- [ ] Dashboard running (http://localhost:8501)
- [ ] Test prediction works for AAPL
- [ ] SHAP explanations display correctly
- [ ] Financial metrics show properly

---

## 🎉 Success Indicators

You'll know everything works when:

1. **API Health Check** returns `"model_loaded": true`
2. **Dashboard** shows green "✅ API Ready"
3. **Prediction** for AAPL completes in <10 seconds
4. **Risk Gauge** displays correctly
5. **SHAP Chart** shows top factors
6. **Financial Metrics** populate with real data

---

## 📞 Quick Commands Reference

```powershell
# Activate environment
.\venv\Scripts\activate

# Run data enrichment
python data\data_enrichment.py

# Start Jupyter
jupyter notebook

# Start API
cd backend ; uvicorn api.main:app --reload

# Start Dashboard
streamlit run frontend\app.py

# Check API health
curl http://localhost:8000/health

# Test prediction
curl http://localhost:8000/api/predict/AAPL
```

---

**Ready to predict layoffs!** 🚀
