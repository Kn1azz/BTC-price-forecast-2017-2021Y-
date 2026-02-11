def prepare_data(df_raw):
    # Date -> datetime and sort
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Date"]).sort_values("Date")

    # Set Date as index and select relevant columns
    df = df_raw.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Filter by date
    df = df.loc[df.index >= "2017-01-01"].copy()

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")].copy()

    # log_return(t) = ln(Close(t)/Close(t-1))
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # target = log_return(t+1)
    df["target"] = df["log_return"].shift(-1)

    # --- Feature engineering ---
    # Lag price
    for lag in [1, 2, 3, 7]:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

    # Lag returns
    for lag in [1, 2, 5]:
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    # Rolling price stats (past only: shift(1))
    for w in [7, 14, 30]:
        df[f"roll_mean_{w}"] = df["Close"].rolling(w).mean().shift(1)
    df["roll_std_7"]  = df["Close"].rolling(7).std().shift(1)
    df["roll_std_30"] = df["Close"].rolling(30).std().shift(1)

    # Rolling return volatility
    df["roll_ret_std_30"] = df["log_return"].rolling(30).std().shift(1)

    # EMA (past only)
    df["ema_7"]  = df["Close"].ewm(span=7, adjust=False).mean().shift(1)
    df["ema_14"] = df["Close"].ewm(span=14, adjust=False).mean().shift(1)

    # Time features
    df["weekday"] = df.index.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Final cleanup
    df = df.dropna().copy()
    
    return df

# Apply the function to the raw dataframe
processed_df = prepare_data(df_raw)

print("Processed DataFrame shape:", processed_df.shape)
print("Processed DataFrame columns:", processed_df.columns.tolist())
print("Processed DataFrame head:\n", processed_df.head())
def train_predict_evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test):

    def returns_to_price(close_today, pred_return):
        # Price(t+1) = Close(t) * exp(pred_return(t+1))
        return close_today * np.exp(pred_return)

    def eval_returns(name, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        dir_acc = (np.sign(y_true) == np.sign(y_pred)).mean()
        corr = np.corrcoef(y_true, y_pred)[0, 1] if (np.std(y_true) > 0 and np.std(y_pred) > 0) else np.nan
        print(f"{name:18s} | MAE={mae:.6f} RMSE={rmse:.6f} DirAcc={dir_acc:.3f} Corr={corr:.3f}")
        return mae, rmse, dir_acc, corr

    def eval_prices(name, actual_price, pred_price):
        mae = mean_absolute_error(actual_price, pred_price)
        rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
        mape = mean_absolute_percentage_error(actual_price, pred_price) * 100
        print(f"{name:22s} | MAE={mae:,.2f} RMSE={rmse:,.2f} MAPE={mape:.2f}%")
        return mae, rmse, mape

    # Initialize and train LightGBM model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1, # Ensure max_depth is included as per original definition in kyH064dwJq2B
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    print("\n--- Training LightGBM model ---")
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)] # Set verbose to False to reduce output during function call
    )
    print("Training complete.")

    # Make predictions on X_test
    pred_ret_lgb = lgb_model.predict(X_test)

    # Calculate actual and predicted prices
    close_today_test = X_test["Close"].values
    actual_price_test = returns_to_price(close_today_test, y_test.values)
    pred_price_lgb = returns_to_price(close_today_test, pred_ret_lgb)

    # Baseline: 0 return
    pred_ret_zero = np.zeros_like(y_test.values)
    pred_price_zero = returns_to_price(close_today_test, pred_ret_zero)

    # Evaluate returns performance
    print("\n=== RETURNS METRICS ===")
    eval_returns("Baseline: 0 return", y_test.values, pred_ret_zero)
    eval_returns("LightGBM", y_test.values, pred_ret_lgb)

    # Evaluate price performance
    print("\n=== PRICE METRICS ===")
    eval_prices("Baseline (price naive)", actual_price_test, pred_price_zero)
    eval_prices("LightGBM (price)", actual_price_test, pred_price_lgb)

    return lgb_model, pred_ret_lgb, pred_price_lgb, actual_price_test, pred_price_zero

# Example of how to call the function (this part will not be executed yet, but shows usage)
# model, pred_ret_lgb_res, pred_price_lgb_res, actual_price_res, pred_price_zero_res = train_predict_evaluate_model(
#     X_train, y_train, X_valid, y_valid, X_test, y_test
# )
model, pred_ret_lgb_res, pred_price_lgb_res, actual_price_res, pred_price_zero_res = train_predict_evaluate_model(
    X_train, y_train, X_valid, y_valid, X_test, y_test
)
import streamlit as st

st.set_page_config(layout="wide")

st.title("Bitcoin Price Prediction App")

st.sidebar.title("Navigation")

st.header("1. Data Overview")
st.write("Content for data overview will go here.")

st.header("2. Feature Descriptions")
st.write("Content for feature descriptions will go here.")

st.header("3. Modeling Results")
st.write("Content for modeling results will go here.")

import streamlit as st

st.set_page_config(layout="wide")

st.title("Bitcoin Price Prediction App")

st.sidebar.title("Navigation")

st.header("1. Data Overview")
st.write("Here's a glimpse of the processed data used for prediction:")
st.write(f"Shape of the processed DataFrame: {processed_df.shape}")
st.dataframe(processed_df.head())

st.header("2. Feature Descriptions")
st.write("Content for feature descriptions will go here.")

st.header("3. Modeling Results")
st.write("Content for modeling results will go here.")




import streamlit as st

st.set_page_config(layout="wide")

st.title("Bitcoin Price Prediction App")

st.sidebar.title("Navigation")

st.header("1. Data Overview")
st.write("Here's a glimpse of the processed data used for prediction:")
st.write(f"Shape of the processed DataFrame: {processed_df.shape}")
st.dataframe(processed_df.head())

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

st.header("3. Modeling Results")
st.write("Content for modeling results will go here.")
