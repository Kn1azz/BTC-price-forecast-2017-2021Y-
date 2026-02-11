import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Helper functions (redefined for self-containment in the Streamlit script context)
def returns_to_price(close_today, pred_return):
    return close_today * np.exp(pred_return)

def eval_returns(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = (np.sign(y_true) == np.sign(y_pred)).mean()
    corr = np.corrcoef(y_true, y_pred)[0, 1] if (np.std(y_true) > 0 and np.std(y_pred) > 0) else np.nan
    return f"{name:18s} | MAE={mae:.6f} RMSE={rmse:.6f} DirAcc={dir_acc:.3f} Corr={corr:.3f}"

def eval_prices(name, actual_price, pred_price):
    mae = mean_absolute_error(actual_price, pred_price)
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mape = mean_absolute_percentage_error(actual_price, pred_price) * 100
    return f"{name:22s} | MAE={mae:,.2f} RMSE={rmse:,.2f} MAPE={mape:.2f}%"

st.set_page_config(layout="wide")

st.title("Bitcoin Price Prediction App")

st.sidebar.title("Navigation")

# --- 1. Data Overview ---
st.header("1. Data Overview")
st.write("Here's a glimpse of the processed data used for prediction:")
st.write(f"Shape of the processed DataFrame: {processed_df.shape}")
st.dataframe(processed_df.head())

# --- 2. Feature Descriptions ---
st.header("2. Feature Descriptions")
st.write("The following features are engineered from the raw data to predict Bitcoin's log returns:")
st.markdown(
    """
    - **`log_return`**: The natural logarithm of the ratio of the current day's closing price to the previous day's closing price. It represents the continuous compounded return for the day.
    - **`close_lag_X`**: The closing price from `X` days ago. These are historical price points used to capture past trends.
    - **`ret_lag_X`**: The log return from `X` days ago. These are historical return values, important for capturing volatility and momentum.
    - **`roll_mean_X`**: The rolling mean of the closing price over the last `X` days (shifted by 1 to avoid data leakage). This smooths out short-term fluctuations and highlights longer-term trends.
    - **`roll_std_X`**: The rolling standard deviation of the closing price over the last `X` days (shifted by 1). This measures price volatility over a given period.
    - **`roll_ret_std_30`**: The rolling standard deviation of the log returns over the last 30 days (shifted by 1). This specifically measures the volatility of returns.
    - **`ema_7`, `ema_14`**: Exponential Moving Averages of the closing price over 7 and 14 days, respectively (shifted by 1). EMAs give more weight to recent prices, making them more responsive to new information than simple moving averages.
    - **`weekday`**: The day of the week (0 for Monday, 6 for Sunday). This can capture weekly patterns or anomalies.
    - **`is_weekend`**: A binary indicator (1 if it's a weekend, 0 otherwise). This helps in identifying differences in market behavior during weekends.
    """
)

# --- 3. Modeling Results ---
st.header("3. Modeling Results")

st.subheader("Performance Metrics")
st.write("Baseline: 0 return is a simple prediction where tomorrow's return is assumed to be zero.")
st.write("LightGBM is the trained machine learning model.")
st.write("--- ")

st.markdown("#### Returns Metrics")
pred_ret_zero = np.zeros_like(y_test.values)
st.code(eval_returns("Baseline: 0 return", y_test.values, pred_ret_zero))
st.code(eval_returns("LightGBM", y_test.values, pred_ret_lgb_res))

st.markdown("#### Price Metrics")
st.code(eval_prices("Baseline (price naive)", actual_price_res, pred_price_zero_res))
st.code(eval_prices("LightGBM (price)", actual_price_res, pred_price_lgb_res))

st.write("--- ")

st.subheader("Price Prediction Visualizations")

# Plot 1: Actual vs Predicted Price
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(X_test.index, actual_price_res, label="Actual Price", linewidth=2)
ax1.plot(X_test.index, pred_price_lgb_res, label="Pred Price (LightGBM)", linewidth=2)
ax1.set_title("BTC — Actual vs Predicted (reconstructed from predicted returns)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)
plt.close(fig1) # Close the figure to free up memory

# Plot 2: Price Error
err = actual_price_res - pred_price_lgb_res
fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(X_test.index, err)
ax2.set_title("Price Error (Actual - Pred)")
ax2.grid(True)
st.pyplot(fig2)
plt.close(fig2) # Close the figure to free up memory

# Plot 3: Zoomed-in Last N Days
N = 100
fig3, ax3 = plt.subplots(figsize=(14, 5))
ax3.plot(X_test.index[-N:], actual_price_res[-N:], label="Actual")
ax3.plot(X_test.index[-N:], pred_price_lgb_res[-N:], label="Pred")
ax3.set_title(f"Zoom: last {N} days")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)
plt.close(fig3) # Close the figure to free up memory
