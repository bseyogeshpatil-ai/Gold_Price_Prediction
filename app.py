import streamlit as st
import pandas as pd
import pickle # Used instead of joblib
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import yfinance as yf
import os

# --- Configuration ---
MODEL_FILE = 'arima_gold_model.pkl' 
TICKER_SYMBOL = "GLD"  # Gold ETF
FORECAST_PERIOD = 14
ARIMA_ORDER = (5, 1, 0) 

# --- Data Fetching and Model Management ---

@st.cache_data(ttl=timedelta(hours=12))
def fetch_historical_data():
    """Fetches 2 years of historical data for model training."""
    end_date = date.today()
    start_date = end_date - timedelta(days=2 * 365)
    
    try:
        data = yf.download(TICKER_SYMBOL, start=start_date, end=end_date, progress=False)
        data = data[['Close']].dropna()
        data.rename(columns={'Close': 'Price'}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching historical data from yfinance: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_and_train_model(historical_data):
    """Loads a saved model or trains a new one if necessary, using pickle."""
    if historical_data.empty:
        return None

    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model_fit = pickle.load(f)
            st.success("Trained ARIMA model loaded successfully!")
            return model_fit
        except Exception as e:
            st.warning(f"Could not load saved model ({e}). Retraining...")

    st.info("Training new ARIMA model...")
    try:
        series = historical_data['Price'].copy()
        model = ARIMA(series, order=ARIMA_ORDER)
        model_fit = model.fit()

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_fit, f)
            
        st.success("New model trained and saved successfully!")
        return model_fit
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None

# --- Live Data Fetching ---

@st.cache_data(ttl=timedelta(minutes=1))
def fetch_live_price():
    """Fetches the latest available price snapshot."""
    try:
        ticker = yf.Ticker(TICKER_SYMBOL)
        info = ticker.info
        current_price = info.get('currentPrice')
        
        if current_price is None:
             current_price = info.get('regularMarketPrice') 

        return current_price
    except Exception:
        return None

# --- Forecasting ---

def forecast_gold_price(model_fit, historical_data, periods):
    """Generates an N-day forecast."""
    forecast_results = model_fit.get_forecast(steps=periods)
    forecast = forecast_results.predicted_mean
    
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='B') 

    forecast_df = pd.DataFrame(
        {'Date': future_dates, 'Predicted_Price': forecast.values}
    )
    return forecast_df

# --- Streamlit UI ---

st.set_page_config(page_title="Gold Price Forecast", layout="wide")

st.title(f"💰 {TICKER_SYMBOL} Gold Price 14-Day Forecast")
st.markdown("---")

# 1. Fetch Data and Model
historical_data = fetch_historical_data()
model = load_and_train_model(historical_data)
latest_price = fetch_live_price()

# 2. Display Live Price
col1, col2 = st.columns(2)

if latest_price and not historical_data.empty:
    last_close_price = historical_data['Price'].iloc[-1].item()
    
    col1.metric(
        label=f"Latest Available Price ({TICKER_SYMBOL})", 
        value=f"${latest_price:,.2f}",
        delta=f"vs. Last Close: ${latest_price - last_close_price:,.2f}",
        delta_color="normal"
    )
elif latest_price:
    col1.metric(
        label=f"Latest Available Price ({TICKER_SYMBOL})", 
        value=f"${latest_price:,.2f}"
    )
else:
    col1.warning("Live price data is currently unavailable.")

if model is not None and not historical_data.empty:
    col2.metric(
        label="Last Data Point Used for Forecast",
        value=historical_data.index[-1].strftime('%Y-%m-%d')
    )

st.markdown("---")

# 3. Forecasting Logic
if model is not None and not historical_data.empty:
    
    if st.button(f'Generate {FORECAST_PERIOD}-Day Forecast', use_container_width=True):
        
        with st.spinner('Generating time series forecast...'):
            forecast_df = forecast_gold_price(model, historical_data, FORECAST_PERIOD) 
            
            st.subheader("🗓️ Predicted Gold Prices")
            
            st.dataframe(
                forecast_df.set_index('Date').style.format({'Predicted_Price': "${:,.2f}"}),
                use_container_width=True
            )
            
            # --- Visualization ---
            plot_hist = historical_data['Price'].tail(180) 
            plot_forecast = forecast_df.set_index('Date')['Predicted_Price']
            
            last_hist_date = plot_hist.index[-1]
            last_hist_price = plot_hist.iloc[-1] 
            
            forecast_series = pd.Series(
                [last_hist_price] + plot_forecast.tolist(),
                index=[last_hist_date] + plot_forecast.index.tolist()
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(plot_hist.index, plot_hist, label='Historical Price', color='blue')
            ax.plot(forecast_series.index, forecast_series.values, 
                    label=f'{FORECAST_PERIOD}-Day Forecast', color='red', linestyle='--')
            
            ax.set_title(f'{TICKER_SYMBOL} Price Trend and Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True, alpha=0.6)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.success("Forecast generation complete!")
            
    st.subheader("Historical Gold Price Data (Last 30 Days)")
    st.line_chart(historical_data['Price'].tail(30))
    
else:
    st.error("Cannot run the forecast. Check the errors above regarding data fetching or model training.")
