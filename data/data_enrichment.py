"""
Data Enrichment Pipeline for WorkforceIQ
Fetches quarterly financial metrics from yfinance and enriches company data
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import warnings
warnings.filterwarnings('ignore')


class FinancialDataEnricher:
    """
    Enriches company data with financial metrics from Yahoo Finance.
    Fetches data from 6 months ago to align with layoff prediction window.
    """
    
    def __init__(self, lookback_months: int = 6):
        """
        Initialize the enricher.
        
        Args:
            lookback_months: How many months back to fetch financial data
        """
        self.lookback_months = lookback_months
        self.target_date = datetime.now() - timedelta(days=30 * lookback_months)
        
    def get_financial_metrics(self, ticker: str) -> Optional[Dict]:
        """
        Fetch quarterly financial metrics for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            
        Returns:
            Dictionary with financial metrics or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly financials
            quarterly_financials = stock.quarterly_financials
            quarterly_balance = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            # Get stock info for additional metrics
            info = stock.info
            
            # Calculate key metrics
            metrics = {}
            
            # Revenue metrics
            if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
                revenues = quarterly_financials.loc['Total Revenue']
                if len(revenues) >= 2:
                    metrics['revenue'] = revenues.iloc[0]
                    metrics['revenue_growth'] = ((revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1] * 100)
                else:
                    metrics['revenue'] = revenues.iloc[0] if len(revenues) > 0 else None
                    metrics['revenue_growth'] = None
            
            # Operating income/profit
            if not quarterly_financials.empty and 'Operating Income' in quarterly_financials.index:
                metrics['operating_income'] = quarterly_financials.loc['Operating Income'].iloc[0]
            
            # Net income
            if not quarterly_financials.empty and 'Net Income' in quarterly_financials.index:
                net_incomes = quarterly_financials.loc['Net Income']
                metrics['net_income'] = net_incomes.iloc[0]
                if len(net_incomes) >= 2:
                    metrics['net_income_change'] = ((net_incomes.iloc[0] - net_incomes.iloc[1]) / abs(net_incomes.iloc[1]) * 100) if net_incomes.iloc[1] != 0 else None
            
            # Debt metrics
            if not quarterly_balance.empty:
                if 'Total Debt' in quarterly_balance.index:
                    metrics['total_debt'] = quarterly_balance.loc['Total Debt'].iloc[0]
                if 'Cash And Cash Equivalents' in quarterly_balance.index:
                    cash = quarterly_balance.loc['Cash And Cash Equivalents'].iloc[0]
                    metrics['cash'] = cash
                    if 'total_debt' in metrics:
                        metrics['debt_to_cash_ratio'] = metrics['total_debt'] / cash if cash > 0 else None
            
            # Cash flow metrics
            if not quarterly_cashflow.empty:
                if 'Operating Cash Flow' in quarterly_cashflow.index:
                    metrics['operating_cashflow'] = quarterly_cashflow.loc['Operating Cash Flow'].iloc[0]
                if 'Free Cash Flow' in quarterly_cashflow.index:
                    metrics['free_cashflow'] = quarterly_cashflow.loc['Free Cash Flow'].iloc[0]
            
            # Market metrics from info
            metrics['market_cap'] = info.get('marketCap')
            metrics['employee_count'] = info.get('fullTimeEmployees')
            metrics['profit_margin'] = info.get('profitMargins')
            metrics['beta'] = info.get('beta')  # Stock volatility
            metrics['pe_ratio'] = info.get('trailingPE')
            
            # Add ticker and fetch date
            metrics['ticker'] = ticker
            metrics['data_fetch_date'] = datetime.now().strftime('%Y-%m-%d')
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def enrich_dataset(self, 
                      company_data: pd.DataFrame, 
                      ticker_column: str = 'ticker',
                      rate_limit_delay: float = 0.5) -> pd.DataFrame:
        """
        Enrich a dataset of companies with financial metrics.
        
        Args:
            company_data: DataFrame with company information
            ticker_column: Name of column containing stock tickers
            rate_limit_delay: Delay between API calls (seconds)
            
        Returns:
            Enriched DataFrame with financial metrics
        """
        enriched_data = []
        
        print(f"Starting enrichment for {len(company_data)} companies...")
        print(f"Fetching data from approximately {self.target_date.strftime('%Y-%m-%d')}")
        
        for idx, row in company_data.iterrows():
            ticker = row.get(ticker_column)
            
            if pd.isna(ticker) or ticker == '':
                print(f"Row {idx}: No ticker found, skipping...")
                enriched_data.append(row.to_dict())
                continue
            
            print(f"Processing {idx + 1}/{len(company_data)}: {ticker}")
            
            # Fetch financial metrics
            metrics = self.get_financial_metrics(ticker)
            
            # Merge with original data
            row_dict = row.to_dict()
            if metrics:
                row_dict.update(metrics)
            
            enriched_data.append(row_dict)
            
            # Rate limiting
            time.sleep(rate_limit_delay)
        
        enriched_df = pd.DataFrame(enriched_data)
        print(f"\nEnrichment complete! Added {len(metrics.keys()) if metrics else 0} new features.")
        
        return enriched_df
    
    def create_ticker_mapping(self, 
                             company_names: List[str], 
                             manual_mappings: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Create a mapping from company names to stock tickers.
        For the synthetic dataset, we'll need manual mapping for real companies.
        
        Args:
            company_names: List of company names
            manual_mappings: Optional dictionary of known company -> ticker mappings
            
        Returns:
            Dictionary mapping company names to tickers
        """
        # Common tech company mappings
        default_mappings = {
            'Google': 'GOOGL',
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Amazon': 'AMZN',
            'Meta': 'META',
            'Tesla': 'TSLA',
            'Netflix': 'NFLX',
            'Nvidia': 'NVDA',
            'Intel': 'INTC',
            'AMD': 'AMD',
            'Oracle': 'ORCL',
            'Salesforce': 'CRM',
            'Adobe': 'ADBE',
            'IBM': 'IBM',
            'Cisco': 'CSCO',
            'Uber': 'UBER',
            'Lyft': 'LYFT',
            'Airbnb': 'ABNB',
            'Spotify': 'SPOT',
            'Snap': 'SNAP',
            'Twitter': 'TWTR',
            'Zoom': 'ZM',
            'Slack': 'WORK',
            'Stripe': 'STRIP',
            'SpaceX': None,  # Not public
        }
        
        if manual_mappings:
            default_mappings.update(manual_mappings)
        
        return default_mappings


def enrich_real_world_dataset(input_path: str, output_path: str):
    """
    Enrich the real-world layoff dataset with current financial metrics.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save enriched CSV
    """
    print("=" * 60)
    print("ENRICHING REAL-WORLD LAYOFF DATASET")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df)} records from {input_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Map company names to tickers (you may need to manually map more)
    ticker_mappings = {
        'Google': 'GOOGL',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Amazon': 'AMZN',
        'Meta': 'META',
        'Tesla': 'TSLA',
        'Intel': 'INTC',
        'AMD': 'AMD',
        'Oracle': 'ORCL',
        'Salesforce': 'CRM',
        'Uber': 'UBER',
        'Lyft': 'LYFT',
        'Airbnb': 'ABNB',
        'Spotify': 'SPOT',
        'Snap Inc.': 'SNAP',
        'Netflix': 'NFLX',
        'Zoom': 'ZM',
        'Coinbase': 'COIN',
        'Robinhood': 'HOOD',
        'DocuSign': 'DOCU',
        'Zillow': 'ZG',
        'Pinterest': 'PINS',
        'Twilio': 'TWLO',
        'SAP': 'SAP',
        'Alibaba': 'BABA',
        'Philips': 'PHG',
        'Ericsson': 'ERIC',
        'BioNTech': 'BNTX',
        'Sea Limited': 'SE',
        'Rakuten': 'RKUNY',
        'SoftBank': 'SFTBY',
        'HubSpot': 'HUBS',
    }
    
    # Add ticker column
    df['ticker'] = df['company'].map(ticker_mappings)
    
    # Filter to only companies with tickers (public companies)
    df_with_tickers = df[df['ticker'].notna()].copy()
    print(f"\nFound {len(df_with_tickers)} records with valid stock tickers")
    print(f"Unique companies: {df_with_tickers['ticker'].nunique()}")
    
    # Initialize enricher
    enricher = FinancialDataEnricher(lookback_months=6)
    
    # Enrich the dataset
    enriched_df = enricher.enrich_dataset(df_with_tickers, ticker_column='ticker')
    
    # Save enriched data
    enriched_df.to_csv(output_path, index=False)
    print(f"\n✅ Enriched dataset saved to: {output_path}")
    print(f"Total records: {len(enriched_df)}")
    print(f"Total columns: {len(enriched_df.columns)}")
    
    return enriched_df


def prepare_training_data(enriched_path: str, output_path: str):
    """
    Prepare the final training dataset with proper feature engineering.
    
    Args:
        enriched_path: Path to enriched CSV
        output_path: Path to save training-ready CSV
    """
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATASET")
    print("=" * 60)
    
    df = pd.read_csv(enriched_path)
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    
    # Create binary target: did layoff happen within next 6 months?
    # For now, we'll use the number_laid_off > 0 as proxy
    df['layoff_risk'] = (df['number_laid_off'] > 0).astype(int)
    
    # Create additional derived features
    if 'revenue' in df.columns and 'employee_count' in df.columns:
        df['revenue_per_employee'] = df['revenue'] / df['employee_count']
    
    if 'operating_income' in df.columns and 'revenue' in df.columns:
        df['operating_margin'] = (df['operating_income'] / df['revenue']) * 100
    
    if 'net_income' in df.columns:
        df['is_profitable'] = (df['net_income'] > 0).astype(int)
    
    # Select relevant features for ML
    feature_columns = [
        'revenue', 'revenue_growth', 'operating_income', 'net_income',
        'net_income_change', 'total_debt', 'cash', 'debt_to_cash_ratio',
        'operating_cashflow', 'free_cashflow', 'market_cap', 'employee_count',
        'profit_margin', 'beta', 'pe_ratio', 'revenue_per_employee',
        'operating_margin', 'is_profitable'
    ]
    
    # Keep only rows with sufficient data
    available_features = [col for col in feature_columns if col in df.columns]
    df_clean = df[available_features + ['layoff_risk', 'company', 'ticker', 'date_announced']].copy()
    
    # Drop rows with too many missing values
    threshold = len(available_features) * 0.5  # At least 50% of features must be present
    df_clean = df_clean.dropna(thresh=threshold)
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"Features: {available_features}")
    print(f"Target distribution:\n{df_clean['layoff_risk'].value_counts()}")
    
    # Save training data
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ Training dataset saved to: {output_path}")
    
    return df_clean


if __name__ == "__main__":
    print("🚀 WorkforceIQ Data Enrichment Pipeline\n")
    
    # Paths
    raw_data_path = "../archive (1)/global_ai_tech_layoffs_2020_2026.csv"
    enriched_path = "processed/layoffs_enriched.csv"
    training_path = "processed/layoffs_training_data.csv"
    
    # Step 1: Enrich real-world dataset with financial metrics
    enriched_df = enrich_real_world_dataset(raw_data_path, enriched_path)
    
    # Step 2: Prepare training dataset
    training_df = prepare_training_data(enriched_path, training_path)
    
    print("\n" + "=" * 60)
    print("✅ DATA ENRICHMENT COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review the enriched data: {enriched_path}")
    print(f"2. Explore the training data: {training_path}")
    print(f"3. Run EDA in notebooks/exploratory_analysis.ipynb")
    print(f"4. Train the model in notebooks/model_training.ipynb")
