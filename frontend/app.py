"""
Streamlit Dashboard for WorkforceIQ
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="WorkforceIQ - Layoff Risk Predictor",
    page_icon="📊",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Title
st.title("🔮 WorkforceIQ - Layoff Risk Predictor")
st.markdown("*AI-powered analysis of company layoff risk using financial metrics and machine learning*")

# Sidebar
with st.sidebar:
    st.header("📌 About")
    st.markdown("""
    **WorkforceIQ** analyzes company financials to predict layoff risk within 3-6 months.
    
    **Tech Stack:**
    - XGBoost ML Model
    - SHAP Explainability
    - Real-time Financial Data
    
    **How it works:**
    1. Enter a company ticker
    2. ML model predicts layoff risk
    3. Feature importance explains prediction
    
    **Note:** Using sample data for demo
    """)
    
    st.markdown("---")
    st.markdown("**Try these:**")
    st.markdown("Any ticker (AAPL, GOOGL, etc.) - returns sample prediction")

# Main content
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker_input = st.text_input(
        "Enter Stock Ticker (e.g., AAPL, GOOGL)",
        placeholder="AAPL",
        help="Enter a valid stock ticker symbol"
    )

with col2:
    analyze_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)

with col3:
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("model_loaded"):
                st.success("✅ API Ready")
            else:
                st.warning("⚠️ Model Not Loaded")
        else:
            st.error("❌ API Down")
    except:
        st.error("❌ API Down")

# Analysis section
if analyze_button and ticker_input:
    ticker = ticker_input.upper().strip()
    
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # Call API
            response = requests.get(f"{API_URL}/api/predict/{ticker}", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Risk Score Section
                st.markdown("---")
                st.header(f"📊 {data['company_name']} ({data['ticker']})")
                
                # Risk Gauge
                risk_score = data['risk_score'] * 100
                risk_level = data['risk_level']
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Layoff Risk Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if risk_score > 70 else "orange" if risk_score > 40 else "green"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Risk Level", risk_level, 
                             delta=None,
                             delta_color="inverse")
                    st.metric("Confidence", f"{data['confidence']*100:.1f}%")
                
                with col3:
                    if risk_level == "HIGH":
                        st.error("⚠️ HIGH RISK")
                        st.markdown("Consider caution")
                    elif risk_level == "MEDIUM":
                        st.warning("⚡ MEDIUM RISK")
                        st.markdown("Monitor closely")
                    else:
                        st.success("✅ LOW RISK")
                        st.markdown("Stable outlook")
                
                # Top Risk Factors
                st.markdown("---")
                st.subheader("🎯 Top Contributing Factors")
                
                factors_df = pd.DataFrame(data['top_factors'])
                
                # Create bar chart
                fig = px.bar(
                    factors_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    labels={'importance': 'Feature Importance', 'feature': 'Factor'},
                    title="Feature Importance in Prediction"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Factors table
                st.markdown("**Detailed Factor Analysis:**")
                factors_display = factors_df.copy()
                factors_display['value'] = factors_display['value'].apply(lambda x: f"{x:,.2f}")
                factors_display['importance'] = factors_display['importance'].apply(lambda x: f"{x:.3f}")
                factors_display.columns = ['Feature', 'Current Value', 'Importance Score']
                st.dataframe(factors_display, use_container_width=True)
                
                # Financial Metrics
                st.markdown("---")
                st.subheader("💰 Company Metrics")
                
                metrics = data.get('financial_metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    funding = metrics.get('funding_amount')
                    if funding:
                        st.metric("Funding", f"${funding/1e9:.2f}B")
                
                with col2:
                    employees = metrics.get('employee_count')
                    if employees:
                        st.metric("Employees", f"{employees:,}")
                
                with col3:
                    growth = metrics.get('growth_rate')
                    if growth is not None:
                        st.metric("Growth Rate", f"{growth:.1f}%")
                
                with col4:
                    valuation = metrics.get('valuation')
                    if valuation:
                        st.metric("Valuation", f"${valuation/1e9:.2f}B")
                
                # Historical trends
                st.markdown("---")
                st.subheader("📈 Historical Trends (6 Months)")
                
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="6mo")
                    
                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name='Stock Price',
                            line=dict(color='blue', width=2)
                        ))
                        fig.update_layout(
                            title=f"{ticker} Stock Price (6 Months)",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Historical price data not available")
                
                # Insights
                st.markdown("---")
                st.subheader("💡 Key Insights")
                
                insights = []
                
                # Generate insights based on metrics
                growth = metrics.get('growth_rate', 0)
                if growth > 50:
                    insights.append("✅ Strong growth rate indicates healthy expansion")
                elif growth < 0:
                    insights.append("⚠️ Negative growth may signal business challenges")
                
                funding = metrics.get('funding_amount', 0)
                if funding > 1e9:
                    insights.append("✅ Well-funded company with strong financial backing")
                
                if risk_level == "LOW":
                    insights.append("✅ Overall outlook suggests low layoff risk in near term")
                elif risk_level == "HIGH":
                    insights.append("⚠️ Multiple risk factors suggest elevated layoff risk")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
                
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please ensure the backend is running:")
            st.code("cd backend\nuvicorn api.main:app --reload")
        
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")

elif analyze_button:
    st.warning("⚠️ Please enter a stock ticker")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>WorkforceIQ v1.0 | Built with Streamlit, FastAPI, XGBoost & SHAP</p>
    <p>⚠️ Disclaimer: This tool is for informational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
