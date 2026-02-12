import numpy as np
import pandas as pd
import streamlit as st

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="BTC Time Series Forecast", layout="wide")
st.title("BTC: прогноз доходности (t+1) и восстановление цены")


st.markdown("""
### Приложение для анализа временного ряда Bitcoin и прогноза доходностей

Это исследовательское приложение на базе машинного обучения, которое показывает полный пайплайн работы с временными рядами:  
от подготовки данных до обучения модели и интерпретации результатов.

**Проект:** Прогнозирование логарифмической доходности Bitcoin (t+1)
""")

st.markdown("**Разработчик:** Muhamadasror Abbosov")

st.markdown("""
Данные: **coin_Bitcoin.csv** (ежедневные OHLCV)  
Диапазон: **2013-04-29 → 2021-07-06**  
Для моделирования используется период: **с 2017-01-01** (1648 записей до построения признаков)

Модель обучается предсказывать **log-return(t+1)**:  
`log_return(t+1) = ln(Close(t+1) / Close(t))`  
Это делает ряд ближе к стационарному и удобному для прогнозирования.

В приложении вы увидите:
- обзор датасета и график цены BTC  
- шаги feature engineering (23 признака: лаги, rolling, EMA, календарные)  
- time split (Train / Valid / Test) без утечки будущего  
- сравнение с бейзлайнами и результаты LightGBM  
- визуализацию прогнозов (доходности и восстановленная цена)  
- важность признаков и итоговый вывод исследования

**Перейдите в раздел «Отчёт проекта»**, чтобы посмотреть все шаги и результаты!
""")


# -------------------------
# Helpers
# -------------------------
def eval_returns(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = (np.sign(y_true) == np.sign(y_pred)).mean()
    corr = np.corrcoef(y_true, y_pred)[0, 1] if (np.std(y_true) > 0 and np.std(y_pred) > 0) else np.nan
    return mae, rmse, dir_acc, corr

def returns_to_price(close_today, ret):
    return close_today * np.exp(ret)

@st.cache_data(show_spinner=False)
def load_and_prepare(file, cutoff_date, horizon, use_price_feats=True):
    df_raw = pd.read_csv(file)

    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Date"]).sort_values("Date")

    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df_raw.set_index("Date")[base_cols].copy()
    df = df.loc[df.index >= pd.to_datetime(cutoff_date)].copy()
    df = df[~df.index.duplicated(keep="last")]

    # log-return(t)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # target = log_return(t+horizon)
    df["target"] = df["log_return"].shift(-horizon)

    # --- Features
    if use_price_feats:
        for lag in [1, 2, 3, 7]:
            df[f"close_lag_{lag}"] = df["Close"].shift(lag)

        for w in [7, 14, 30]:
            df[f"roll_mean_{w}"] = df["Close"].rolling(w).mean().shift(1)
        df["roll_std_7"]  = df["Close"].rolling(7).std().shift(1)
        df["roll_std_30"] = df["Close"].rolling(30).std().shift(1)

        df["ema_7"]  = df["Close"].ewm(span=7, adjust=False).mean().shift(1)
        df["ema_14"] = df["Close"].ewm(span=14, adjust=False).mean().shift(1)

    # return lags + vol
    for lag in [1, 2, 5]:
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)
    df["roll_ret_std_30"] = df["log_return"].rolling(30).std().shift(1)

    # time feats
    df["weekday"] = df.index.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df = df.dropna().copy()
    return df

def time_split(df, train_ratio=0.70, valid_ratio=0.15):
    N = len(df)
    i1 = int(N * train_ratio)
    i2 = int(N * (train_ratio + valid_ratio))
    train = df.iloc[:i1].copy()
    valid = df.iloc[i1:i2].copy()
    test  = df.iloc[i2:].copy()
    return train, valid, test

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Настройки")
uploaded = st.sidebar.file_uploader("Загрузи coin_Bitcoin.csv", type=["csv"])

cutoff_date = st.sidebar.text_input("Cutoff date", value="2017-01-01")
horizon = st.sidebar.number_input("Горизонт (дни вперёд)", min_value=1, max_value=30, value=1, step=1)

use_price_feats = st.sidebar.checkbox("Использовать price-features (Close lags/rolling/EMA)", value=True)

train_ratio = st.sidebar.slider("Train ratio", 0.50, 0.85, 0.70, 0.01)
valid_ratio = st.sidebar.slider("Valid ratio", 0.05, 0.30, 0.15, 0.01)

st.sidebar.subheader("LightGBM")
learning_rate = st.sidebar.number_input("learning_rate", value=0.02, min_value=0.001, max_value=0.3, step=0.001, format="%.3f")
num_leaves = st.sidebar.number_input("num_leaves", value=31, min_value=8, max_value=512, step=1)
subsample = st.sidebar.slider("subsample", 0.4, 1.0, 0.8, 0.05)
colsample = st.sidebar.slider("colsample_bytree", 0.4, 1.0, 0.8, 0.05)
reg_lambda = st.sidebar.number_input("reg_lambda", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
stopping_rounds = st.sidebar.number_input("early_stopping rounds", value=300, min_value=50, max_value=2000, step=50)

run_btn = st.sidebar.button("Запустить обучение")

st.sidebar.caption("Abbosov Muhammadasror Golibovich")
st.sidebar.caption("        Dushanbe TAjikistan")
st.sidebar.caption('            Feb 2026')
# -------------------------
# Main
# -------------------------
if not uploaded:
    st.info("Загрузи CSV, чтобы начать.")
    st.stop()

df = load_and_prepare(uploaded, cutoff_date, horizon, use_price_feats)

st.subheader("Данные после фичей")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df)}")
c2.metric("Features (incl log_return)", f"{df.drop(columns=['target']).shape[1]}")
c3.metric("Target std", f"{df['target'].std():.6f}")

st.dataframe(df.head(10), use_container_width=True)

# quick sanity: target scale should be small
if df["target"].std() >= 1.0:
    st.error("Target выглядит НЕ как log-return (слишком большой std). Проверь формулу target.")
    st.stop()

train, valid, test = time_split(df, train_ratio, valid_ratio)
X_train, y_train = train.drop(columns=["target"]), train["target"]
X_valid, y_valid = valid.drop(columns=["target"]), valid["target"]
X_test,  y_test  = test.drop(columns=["target"]),  test["target"]

st.subheader("Split по времени")
st.write(
    f"Train: {X_train.shape} ({train.index.min().date()} → {train.index.max().date()})  \n"
    f"Valid: {X_valid.shape} ({valid.index.min().date()} → {valid.index.max().date()})  \n"
    f"Test : {X_test.shape} ({test.index.min().date()} → {test.index.max().date()})"
)

# baselines always computed
pred0 = np.zeros_like(y_test.values)
mae0, rmse0, dir0, corr0 = eval_returns(y_test.values, pred0)

# baseline ret=ret_lag_1 (если есть)
if "ret_lag_1" in X_test.columns:
    pred_ret1 = X_test["ret_lag_1"].values
    mae1, rmse1, dir1, corr1 = eval_returns(y_test.values, pred_ret1)
else:
    mae1 = rmse1 = dir1 = corr1 = np.nan

st.subheader("Baseline метрики (returns)")
b1, b2 = st.columns(2)
b1.write("**baseline: ret=0**")
b1.write({"MAE": mae0, "RMSE": rmse0, "DirAcc": dir0, "Corr": corr0})
b2.write("**baseline: ret=ret_lag_1**")
b2.write({"MAE": mae1, "RMSE": rmse1, "DirAcc": dir1, "Corr": corr1})

if not run_btn:
    st.warning("Нажми «Запустить обучение», чтобы обучить LightGBM и визуализировать прогноз.")
    st.stop()

# Train model
model = lgb.LGBMRegressor(
    n_estimators=20000,
    learning_rate=learning_rate,
    num_leaves=int(num_leaves),
    subsample=subsample,
    colsample_bytree=colsample,
    reg_lambda=reg_lambda,
    random_state=42,
    n_jobs=-1
)

with st.spinner("Обучаю LightGBM..."):
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=int(stopping_rounds), verbose=False)]
    )

pred_lgb = model.predict(X_test)
mae, rmse, dir_acc, corr = eval_returns(y_test.values, pred_lgb)

st.subheader("LightGBM результаты (returns)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("MAE", f"{mae:.6f}")
m2.metric("RMSE", f"{rmse:.6f}")
m3.metric("DirAcc", f"{dir_acc:.3f}")
m4.metric("Corr", f"{corr:.3f}" if not np.isnan(corr) else "nan")
m5.metric("Best iter", f"{model.best_iteration_}")

# Price reconstruction on test
close_today = X_test["Close"].values
actual_price = returns_to_price(close_today, y_test.values)
pred_price = returns_to_price(close_today, pred_lgb)

# Plot: Actual vs Predicted price
st.subheader("Визуализация прогноза цены на тесте")
fig = plt.figure(figsize=(12, 5))
plt.plot(y_test.index, actual_price, label="Actual price", linewidth=2)
plt.plot(y_test.index, pred_price, label="Predicted price (LGBM)", linewidth=2)
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig)

# Plot returns comparison
st.subheader("Визуализация доходностей (t+horizon)")
fig2 = plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test.values, label="Actual return", linewidth=1)
plt.plot(y_test.index, pred_lgb, label="Pred return (LGBM)", linewidth=1)
plt.axhline(0, linewidth=1)
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# Feature importance
st.subheader("Feature importance (top 20)")
imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
st.bar_chart(imp)

# Table: show few rows
st.subheader("Первые строки test: факты vs прогноз")
out = pd.DataFrame({
    "Close(t)": close_today,
    "Actual_return": y_test.values,
    "Pred_return": pred_lgb,
    "Actual_price(t+h)": actual_price,
    "Pred_price(t+h)": pred_price
}, index=y_test.index)
st.dataframe(out.head(15), use_container_width=True)
