import os
import numpy as np
import pandas as pd
import streamlit as st

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# =========================
# Page config
# =========================
st.set_page_config(page_title="BTC Time Series Project", layout="wide")
st.title("📈 BTC Time Series Forecasting — ML исследование (как презентация)")
st.caption("Датасет: data/coin_Bitcoin.csv | Цель: прогноз дневной log-return(t+1) и анализ предсказуемости")


# =========================
# Helpers
# =========================
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

def make_split_plot(train_idx, valid_idx, test_idx):
    fig = plt.figure(figsize=(12, 1.6))
    y = np.ones(len(train_idx))
    plt.plot(train_idx, y, linewidth=6, label="Train")
    plt.plot(valid_idx, y, linewidth=6, label="Valid")
    plt.plot(test_idx,  y, linewidth=6, label="Test")
    plt.yticks([])
    plt.grid(True, axis="x")
    plt.legend(loc="upper center", ncol=3)
    plt.tight_layout()
    return fig


# =========================
# Load + prepare (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_and_prepare(csv_path: str):
    # ---- Load
    df_raw = pd.read_csv(csv_path)

    # ---- Date index
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Date"]).sort_values("Date")

    # ---- Keep OHLCV
    df = df_raw.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()

    # ---- Cutoff (как у тебя)
    df = df.loc[df.index >= "2017-01-01"].copy()

    # ---- remove duplicated timestamps
    df = df[~df.index.duplicated(keep="last")].copy()

    raw_info = {
        "raw_shape": df.shape,
        "min_date": df.index.min(),
        "max_date": df.index.max()
    }

    # ---- Target
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["target"] = df["log_return"].shift(-1)

    target_desc = df["target"].describe()

    # ---- Features
    for lag in [1, 2, 3, 7]:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

    for lag in [1, 2, 5]:
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    for w in [7, 14, 30]:
        df[f"roll_mean_{w}"] = df["Close"].rolling(w).mean().shift(1)

    df["roll_std_7"] = df["Close"].rolling(7).std().shift(1)
    df["roll_std_30"] = df["Close"].rolling(30).std().shift(1)

    df["roll_ret_std_30"] = df["log_return"].rolling(30).std().shift(1)

    df["ema_7"] = df["Close"].ewm(span=7, adjust=False).mean().shift(1)
    df["ema_14"] = df["Close"].ewm(span=14, adjust=False).mean().shift(1)

    df["weekday"] = df.index.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # ---- Final clean
    df = df.dropna().copy()

    feat_info = {
        "after_features_shape": df.shape,
        "n_columns": len(df.columns),
        "n_features": len(df.columns) - 1  # exclude target
    }
    return df, raw_info, target_desc, feat_info


@st.cache_data(show_spinner=False)
def train_and_evaluate(df):
    # ---- Time split 70/15/15
    N = len(df)
    i1 = int(N * 0.70)
    i2 = int(N * 0.85)

    train = df.iloc[:i1].copy()
    valid = df.iloc[i1:i2].copy()
    test  = df.iloc[i2:].copy()

    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_valid, y_valid = valid.drop(columns=["target"]), valid["target"]
    X_test,  y_test  = test.drop(columns=["target"]),  test["target"]

    # ---- Baseline ret=0
    pred0 = np.zeros_like(y_test.values)
    m0 = eval_returns(y_test.values, pred0)

    # ---- LightGBM (фиксированные параметры, без UI)
    model = lgb.LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
    )

    pred = model.predict(X_test)
    m = eval_returns(y_test.values, pred)

    # ---- Reconstruct price(t+1) from returns
    close_today = X_test["Close"].values
    actual_price = returns_to_price(close_today, y_test.values)
    pred_price   = returns_to_price(close_today, pred)

    # ---- Feature importance
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # ---- Pack
    out = {
        "train": train, "valid": valid, "test": test,
        "X_train": X_train, "y_train": y_train,
        "X_valid": X_valid, "y_valid": y_valid,
        "X_test": X_test,   "y_test": y_test,
        "baseline_metrics": m0,
        "lgb_metrics": m,
        "best_iteration": model.best_iteration_,
        "pred_ret": pred,
        "actual_price": actual_price,
        "pred_price": pred_price,
        "importance": imp
    }
    return out


# =========================
# Load dataset from repo
# =========================
CSV_PATH = os.path.join("data", "coin_Bitcoin.csv")

if not os.path.exists(CSV_PATH):
    st.error(
        "Файл data/coin_Bitcoin.csv не найден.\n\n"
        "1) Создай папку data/\n"
        "2) Положи туда coin_Bitcoin.csv\n"
        "3) Убедись, что .gitignore не блокирует этот файл\n"
    )
    st.stop()

df, raw_info, target_desc, feat_info = load_and_prepare(CSV_PATH)
res = train_and_evaluate(df)


# =========================
# PRESENTATION CONTENT
# =========================

# ---- Step 1: Dataset
st.header("1) Данные и период")
c1, c2, c3 = st.columns(3)
c1.metric("Строк (после cutoff)", f"{raw_info['raw_shape'][0]}")
c2.metric("Начало", f"{raw_info['min_date'].date()}")
c3.metric("Конец", f"{raw_info['max_date'].date()}")

st.write("""
Мы используем ежедневные свечи Bitcoin: **Open, High, Low, Close, Volume**.  
Период анализа: **2017-01-01 → 2021-07-06**.
""")

with st.expander("Показать пример данных"):
    st.dataframe(df[["Open","High","Low","Close","Volume"]].head(10), use_container_width=True)

# Price plot
st.subheader("График цены BTC (Close)")
fig = plt.figure(figsize=(12, 4))
plt.plot(df.index, df["Close"])
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)


# ---- Step 2: Stationarity + target
st.header("2) Стационаризация и таргет")
st.write(r"""
Цена нестационарна, поэтому строим **лог-доходности**:

\[
log\_return(t) = \ln(\frac{Close(t)}{Close(t-1)})
\]
и задаём таргет:
\[
target = log\_return(t+1)
\]
""")

st.subheader("Проверка масштаба target (должен быть маленьким)")
st.dataframe(target_desc.to_frame("value"), use_container_width=True)

st.subheader("Распределение лог-доходностей")
fig = plt.figure(figsize=(12, 4))
plt.hist(df["log_return"].values, bins=60)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)


# ---- Step 3: Features
st.header("3) Признаки")
st.write(f"""
Мы создали лаги, rolling-статистики, EMA и календарные признаки.  
После фичей: **{feat_info['after_features_shape'][0]} строк** и **{feat_info['n_features']} признака** (без target).
""")

with st.expander("Список колонок"):
    st.code(", ".join(df.columns))

# (опционально) корреляции
st.subheader("Корреляции (фрагмент)")
corr_cols = ["Close","log_return","ret_lag_1","ret_lag_2","ret_lag_5","roll_ret_std_30","Volume","roll_std_30"]
corr = df[corr_cols].corr()
fig = plt.figure(figsize=(8, 5))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
plt.tight_layout()
st.pyplot(fig)


# ---- Step 4: Split
st.header("4) Train / Valid / Test split")
st.write("""
Делаем честный split по времени:
- **Train 70%**
- **Valid 15%** (для early stopping)
- **Test 15%**
""")

fig = make_split_plot(res["train"].index, res["valid"].index, res["test"].index)
st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("Train rows", str(len(res["train"])))
c2.metric("Valid rows", str(len(res["valid"])))
c3.metric("Test rows", str(len(res["test"])))


# ---- Step 5: Baseline
st.header("5) Бейзлайн")
st.write("""
Сильный бейзлайн для финансового ряда: **return = 0**, т.е. “цена завтра = цена сегодня”.  
Он часто очень конкурентен, потому что дневные изменения относительно цены невелики.
""")

b_mae, b_rmse, b_dir, b_corr = res["baseline_metrics"]
st.metric("Baseline (ret=0) MAE", f"{b_mae:.6f}")
st.write(f"RMSE: **{b_rmse:.6f}**, DirAcc: **{b_dir:.3f}**")


# ---- Step 6: Model
st.header("6) Модель (LightGBM + early stopping)")
st.write("""
Мы обучили **LightGBMRegressor** на созданных признаках.  
Использовали **early stopping**, чтобы модель не переобучалась на валидации.
""")

st.metric("Best iteration", str(res["best_iteration"]))

st.subheader("Feature importance (Top 15)")
top15 = res["importance"].head(15)
fig = plt.figure(figsize=(10, 5))
plt.bar(range(len(top15)), top15.values)
plt.xticks(range(len(top15)), top15.index, rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)


# ---- Step 7: Results + visualization
st.header("7) Результаты и визуализация прогноза")
l_mae, l_rmse, l_dir, l_corr = res["lgb_metrics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("LightGBM MAE", f"{l_mae:.6f}")
c2.metric("LightGBM RMSE", f"{l_rmse:.6f}")
c3.metric("DirAcc", f"{l_dir:.3f}")
c4.metric("Corr", f"{l_corr:.3f}" if not np.isnan(l_corr) else "nan")

st.write("""
Наблюдение (как у тебя в результатах):
- LightGBM почти **не превосходит** baseline по MAE/RMSE
- DirAcc около **0.5** и корреляция около **0**
- Early stopping останавливается очень рано (у тебя было **9 итераций**)

Это означает: **на горизонте t+1 дневные доходности BTC близки к шуму**.
""")

# Plot returns (test)
st.subheader("Actual vs Predicted (returns) на тесте")
fig = plt.figure(figsize=(12, 4))
plt.plot(res["y_test"].index, res["y_test"].values, label="Actual return", linewidth=1)
plt.plot(res["y_test"].index, res["pred_ret"], label="Pred return (LGBM)", linewidth=1)
plt.axhline(0)
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig)

# Plot price (test)
st.subheader("Actual vs Predicted (price) на тесте")
fig = plt.figure(figsize=(12, 4))
plt.plot(res["y_test"].index, res["actual_price"], label="Actual price(t+1)", linewidth=2)
plt.plot(res["y_test"].index, res["pred_price"], label="Pred price(t+1) via returns", linewidth=2)
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig)

# Table preview
st.subheader("Примеры прогнозов (первые 12 строк теста)")
preview = pd.DataFrame({
    "Close(t)": res["X_test"]["Close"].values,
    "Actual_return(t+1)": res["y_test"].values,
    "Pred_return(t+1)": res["pred_ret"],
    "Actual_price(t+1)": res["actual_price"],
    "Pred_price(t+1)": res["pred_price"],
}, index=res["y_test"].index)
st.dataframe(preview.head(12), use_container_width=True)


# ---- Conclusion
st.header("Главный вывод")
st.success("""
Дневные log-доходности Bitcoin на горизонте t+1 почти непредсказуемы на основе OHLCV + технических фичей.
LightGBM не превосходит сильный baseline “цена завтра = цена сегодня”, а ранняя остановка срабатывает очень рано.
""")

st.caption("Следующие логичные улучшения: прогноз t+7/t+14, классификация направления, прогноз волатильности.")
