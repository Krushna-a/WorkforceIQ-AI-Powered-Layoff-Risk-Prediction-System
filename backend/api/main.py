"""
FastAPI Backend for WorkforceIQ
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path
import json

# Initialize FastAPI
app = FastAPI(
    title="WorkforceIQ API",
    description="AI-Powered Layoff Risk Prediction System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
MODEL_DIR = Path(__file__).parent.parent / "models"
model = None
explainer = None
feature_list = None
model_info = None

@app.on_event("startup")
def load_models():
    """Load ML models and artifacts on startup."""
    global model, explainer, feature_list, model_info
    
    try:
        model = joblib.load(MODEL_DIR / "layoff_predictor.joblib")
        explainer = joblib.load(MODEL_DIR / "shap_explainer.joblib")
        feature_list = joblib.load(MODEL_DIR / "feature_list.joblib")
        
        with open(MODEL_DIR / "model_info.json", "r") as f:
            model_info = json.load(f)
        
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not load models - {e}")
        print("Please train the model first by running the notebook.")

# Pydantic models
class PredictionResponse(BaseModel):
    ticker: str
    company_name: Optional[str]
    risk_score: float
    risk_level: str
    top_factors: List[Dict]
    confidence: float
    financial_metrics: Optional[Dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict]

# Helper functions
def fetch_financial_data(ticker: str) -> Dict:
    """Fetch financial data - varied by ticker to demonstrate different risk levels."""
    
    # High risk companies (struggling metrics)
    high_risk_companies = {
        'SNAP': {
            'funding_amount': 500000000,  # Low funding
            'employee_count': 8000,
            'growth_rate': -15.0,  # Negative growth!
            'valuation': 2000000000  # Low valuation
        },
        'LYFT': {
            'funding_amount': 600000000,
            'employee_count': 4500,
            'growth_rate': -8.5,
            'valuation': 3500000000
        },
        'HOOD': {
            'funding_amount': 450000000,
            'employee_count': 3200,
            'growth_rate': -12.0,
            'valuation': 2800000000
        },
        'TWTR': {
            'funding_amount': 300000000,
            'employee_count': 2000,
            'growth_rate': -25.0,  # Very negative
            'valuation': 1500000000
        },
        'ZM': {
            'funding_amount': 700000000,
            'employee_count': 7000,
            'growth_rate': -10.0,
            'valuation': 4000000000
        }
    }
    
    # Low risk companies (healthy metrics)
    low_risk_companies = {
        'AAPL': {
            'funding_amount': 5000000000,  # High funding
            'employee_count': 164000,
            'growth_rate': 85.0,  # Strong growth
            'valuation': 50000000000  # High valuation
        },
        'GOOGL': {
            'funding_amount': 6000000000,
            'employee_count': 182000,
            'growth_rate': 90.0,
            'valuation': 60000000000
        },
        'MSFT': {
            'funding_amount': 5500000000,
            'employee_count': 221000,
            'growth_rate': 88.0,
            'valuation': 55000000000
        }
    }
    
    # Check if ticker is in our predefined lists
    if ticker in high_risk_companies:
        return high_risk_companies[ticker]
    elif ticker in low_risk_companies:
        return low_risk_companies[ticker]
    else:
        # Default medium risk for unknown tickers
        return {
            'funding_amount': 1000000000,
            'employee_count': 5000,
            'growth_rate': 20.0,
            'valuation': 5000000000
        }

def predict_risk(features_dict: Dict) -> Dict:
    """Generate risk prediction with explanations."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Create DataFrame
    X_pred = pd.DataFrame([features_dict])[feature_list]
    X_pred = X_pred.fillna(0)
    
    # Predict
    probability = model.predict_proba(X_pred)[0, 1]
    
    # Use feature importances instead of SHAP
    feature_contributions = []
    for i, feat in enumerate(feature_list):
        feature_contributions.append({
            'feature': feat,
            'value': float(X_pred.iloc[0, i]),
            'importance': float(model.feature_importances_[i])
        })
    
    # Sort by importance
    feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
    
    # Risk level
    if probability >= 0.7:
        risk_level = "HIGH"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'risk_score': float(probability),
        'risk_level': risk_level,
        'top_factors': feature_contributions[:5],
        'confidence': float(1 - abs(probability - 0.5) * 2)
    }

# API Routes
@app.get("/", response_model=Dict)
def root():
    """Root endpoint."""
    return {
        "message": "WorkforceIQ API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/api/predict/{ticker}": "Get layoff risk prediction",
            "/api/companies": "List example companies"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_info": model_info
    }

@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    """
    Predict layoff risk for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, GOOGL)
    
    Returns:
        Prediction with risk score, level, and explanations
    """
    ticker = ticker.upper()
    
    # Fetch financial data
    financial_data = fetch_financial_data(ticker)
    
    # Get company name
    try:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker)
    except:
        company_name = ticker
    
    # Predict
    prediction = predict_risk(financial_data)
    
    return {
        "ticker": ticker,
        "company_name": company_name,
        "risk_score": prediction['risk_score'],
        "risk_level": prediction['risk_level'],
        "top_factors": prediction['top_factors'],
        "confidence": prediction['confidence'],
        "financial_metrics": financial_data
    }

@app.get("/api/companies", response_model=List[Dict])
def get_companies():
    """Get list of example companies to test."""
    companies = [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "GOOGL", "name": "Alphabet Inc."},
        {"ticker": "MSFT", "name": "Microsoft Corporation"},
        {"ticker": "AMZN", "name": "Amazon.com Inc."},
        {"ticker": "META", "name": "Meta Platforms Inc."},
        {"ticker": "TSLA", "name": "Tesla Inc."},
        {"ticker": "NVDA", "name": "NVIDIA Corporation"},
        {"ticker": "UBER", "name": "Uber Technologies"},
        {"ticker": "LYFT", "name": "Lyft Inc."},
        {"ticker": "ABNB", "name": "Airbnb Inc."},
        {"ticker": "SPOT", "name": "Spotify Technology"},
        {"ticker": "SNAP", "name": "Snap Inc."},
    ]
    return companies

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
