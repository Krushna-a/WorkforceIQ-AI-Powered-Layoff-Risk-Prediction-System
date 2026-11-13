# WorkforceIQ - AI-Powered Layoff Risk Predictor

An intelligent system that predicts company layoff risk using machine learning and provides explainable insights through feature importance analysis.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

WorkforceIQ analyzes company metrics to predict the probability of layoffs within 3-6 months. The system combines historical layoff data with machine learning to provide actionable insights for employees and HR professionals.

### 🖼️ Dashboard Preview

The interactive dashboard provides:
- Real-time risk assessment with visual gauge
- Feature importance breakdown
- Company metrics overview
- Risk level categorization (LOW/MEDIUM/HIGH)

<img width="1917" height="866" alt="image" src="https://github.com/user-attachments/assets/87df4782-0e8f-4960-8872-f4954fafdc92" />
<img width="1917" height="862" alt="image" src="https://github.com/user-attachments/assets/eb22ddca-6577-4845-b2c7-8bd237a1fc41" />

## 🎯 Key Features

- **ML-Powered Predictions**: XGBoost classifier with 69% accuracy
- **Real-Time Analysis**: FastAPI backend serving instant predictions
- **Interactive Dashboard**: Streamlit UI with risk gauges and visualizations
- **Explainable AI**: Feature importance analysis showing key risk factors
- **RESTful API**: Easy integration with external systems

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 69% |
| **ROC-AUC** | 0.64 |
| **Precision (No Layoff)** | 75% |
| **Recall (No Layoff)** | 81% |
| **Precision (Layoff)** | 51% |
| **Recall (Layoff)** | 42% |

**Training Data**: 8,000 company records  
**Features**: Funding amount, employee count, growth rate, valuation  
**Algorithm**: XGBoost with class balancing

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Start API (Terminal 1)
cd backend
python -m uvicorn api.main:app --reload

# Start Dashboard (Terminal 2)
streamlit run frontend/app.py
```

**Access Dashboard**: http://localhost:8501  
**API Docs**: http://localhost:8000/docs

## 💻 Tech Stack

**Backend**: Python 3.11, FastAPI, Uvicorn  
**ML/Data**: XGBoost, pandas, NumPy, scikit-learn  
**Frontend**: Streamlit, Plotly  
**API**: RESTful with automatic documentation

## 🏗️ Architecture

```
User → Streamlit Dashboard → FastAPI Backend → XGBoost Model → Prediction + Insights
```

## 📁 Project Structure

```
WorkforceIQ/
├── backend/
│   ├── api/main.py              # FastAPI endpoints
│   └── models/                  # Trained ML models (.joblib)
├── frontend/
│   └── app.py                   # Streamlit dashboard
├── data/                        # Training data
├── train_model.py               # Model training script
└── requirements.txt             # Dependencies
```

## 🎮 Usage Example

```python
# API Request
GET /api/predict/AAPL

# Response
{
  "ticker": "AAPL",
  "risk_score": 0.23,
  "risk_level": "LOW",
  "top_factors": [
    {"feature": "growth_rate", "importance": 0.45},
    {"feature": "funding_amount", "importance": 0.28}
  ],
  "confidence": 0.87
}
```

## 🔧 API Endpoints

- `GET /health` - System health check
- `GET /api/predict/{ticker}` - Predict layoff risk
- `GET /api/companies` - List supported companies

## 📈 Results & Insights

The model successfully identifies layoff patterns based on:
1. **Growth Rate** (45% importance) - Declining growth signals risk
2. **Funding Amount** (28% importance) - Underfunded companies at risk
3. **Employee Count** (17% importance) - Rapid hiring/firing patterns
4. **Valuation** (10% importance) - Overvaluation vs. performance

## 🔮 Future Enhancements

- Real-time financial data integration (yfinance API)
- News sentiment analysis
- Historical layoff tracking
- Email alert system
- Mobile app version

## 👤 About

**Krushnakant Bhoir**  
AI/ML Engineer | Full-Stack Developer  

*Passionate about building intelligent systems that solve real-world problems*

---

**⭐ If you find this project useful, please star it!**

## 📄 License

MIT License - Free for educational and commercial use

---

*Built with passion for data-driven decision making* 💼
