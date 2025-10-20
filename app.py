# app.py
# Streamlit App: Real-time Forecast + Inventory Optimization Simulation (NO full_data.csv)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
from scipy.stats import norm
import tensorflow as tf

# -------------------------------------------------------------
# CLASS: Forecast + Optimization
# -------------------------------------------------------------
class InventoryForecasterOptimizer:
    def __init__(self, model_path: str, annual_holding_rate: float = 0.25):
        with open(model_path, "rb") as f:
            comps = pickle.load(f)
        self.mlp = comps["mlp_model"]
        self.lstm = comps["lstm_model"]
        self.cnn = comps["cnn_model"]
        self.meta = comps["meta_model"]
        self.scaler = comps["scaler"]
        self.seq_len = comps["sequence_length"]
        self.feature_cols = comps["feature_columns"]
        self.annual_holding_rate = annual_holding_rate

    def predict_next_day(self, df_last):
        df_last = df_last[self.feature_cols]
        scaled_seq = self.scaler.transform(df_last.values)
        X_seq = np.expand_dims(scaled_seq, axis=0)
        mlp_pred = float(self.mlp.predict(X_seq, verbose=0).flatten()[0])
        lstm_pred = float(self.lstm.predict(X_seq, verbose=0).flatten()[0])
        cnn_pred = float(self.cnn.predict(X_seq, verbose=0).flatten()[0])
        meta_in = np.array([[mlp_pred, lstm_pred, cnn_pred]])
        return float(self.meta.predict(meta_in, verbose=0).flatten()[0])

    def compute_costs_rates(self, unit_cost, unit_price):
        h_rate = self.annual_holding_rate
        holding_per_unit_per_day = h_rate * unit_cost / 365.0
        stockout_per_unit = unit_price - unit_cost
        return holding_per_unit_per_day, stockout_per_unit

    def orderup_to_level(self, mu_LT, sigma_LT, h_cost_unit, p_cost_unit):
        eps = 1e-9
        alpha = p_cost_unit / (p_cost_unit + h_cost_unit + eps)
        alpha = np.clip(alpha, 0.5001, 0.9999)
        z = norm.ppf(alpha)
        return float(max(0.0, mu_LT + z * sigma_LT))

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Smart Inventory Simulator")
st.title("📦 Smart Inventory Optimization Simulation")
st.caption("Forecast-driven order-up-to policy simulation — no full data file required.")

st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path (.pkl)", "ensemble_forecast_model.pkl")
last30_path = st.sidebar.text_input("Last 30 Days CSV", "sku1_last_30days.csv")
horizon = st.sidebar.number_input("Simulation Horizon (days)", value=90, min_value=1, max_value=365)
lead_time = st.sidebar.number_input("Supplier Lead Time (days)", value=7, min_value=1, max_value=30)
annual_holding_rate = st.sidebar.slider("Annual Holding Cost Rate", 0.01, 0.5, 0.25, 0.01)
noise_scale = st.sidebar.slider("Demand Noise (fraction of forecast)", 0.0, 1.0, 0.1, 0.05)
demo_speed = st.sidebar.slider("Speed (sec per day)", 0.05, 2.0, 0.3, 0.05)

start_button = st.button("🚀 Start Simulation")

if start_button:
    # Load model and last 30 days data
    inv = InventoryForecasterOptimizer(model_path, annual_holding_rate)
    last30 = pd.read_csv(last30_path)
    seq_df = last30.tail(inv.seq_len).copy()

    # Default assumptions
    unit_cost = seq_df["Unit_Cost"].iloc[-1] if "Unit_Cost" in seq_df else 20.0
    unit_price = seq_df["Unit_Price"].iloc[-1] if "Unit_Price" in seq_df else 40.0
    current_inventory = seq_df["Inventory_Level"].iloc[-1] if "Inventory_Level" in seq_df else 100.0
    h_cost_unit, p_cost_unit = inv.compute_costs_rates(unit_cost, unit_price)

    # State variables
    scheduled = {}
    history = {"day": [], "forecast": [], "actual": [], "inventory": [], "order": [], "S": [], "cost": []}
    cumulative_cost = 0.0

    # Streamlit placeholders
    chart_ph = st.empty()
    metric_ph = st.empty()
    log_ph = st.empty()

    for t in range(horizon):
        # 1️⃣ Predict demand
        pred = inv.predict_next_day(seq_df)

        # 2️⃣ Rolling mean/std for lead time
        preds_LT = history["forecast"][-(lead_time - 1):] if len(history["forecast"]) >= (lead_time - 1) else history["forecast"]
        mu_LT = np.sum(preds_LT + [pred])
        sigma_LT = np.std(preds_LT + [pred], ddof=0)

        # 3️⃣ Compute Order-up-to Level
        S = inv.orderup_to_level(mu_LT, sigma_LT, h_cost_unit, p_cost_unit)

        # 4️⃣ Compute recommended order (consider pipeline)
        on_order = sum(q for day_idx, q in scheduled.items() if day_idx > t)
        inventory_position = current_inventory + on_order
        Q = max(0.0, S - inventory_position)
        Q = float(np.round(Q))
        if Q > 0:
            scheduled[t + lead_time] = scheduled.get(t + lead_time, 0.0) + Q

        # 5️⃣ Simulate actual demand (add stochasticity)
        actual_demand = max(0.0, pred + np.random.normal(0, noise_scale * max(1.0, pred)))

        # 6️⃣ Process arrivals
        arrivals_today = scheduled.pop(t, 0.0) if t in scheduled else 0.0
        current_inventory += arrivals_today

        # 7️⃣ Fulfill demand
        fulfilled = min(current_inventory, actual_demand)
        unmet = max(0.0, actual_demand - fulfilled)
        current_inventory -= fulfilled

        # 8️⃣ Compute costs
        hold_cost_today = current_inventory * h_cost_unit
        stockout_cost_today = unmet * p_cost_unit
        total_cost_today = hold_cost_today + stockout_cost_today
        cumulative_cost += total_cost_today

        # 9️⃣ Log state
        history["day"].append(t)
        history["forecast"].append(pred)
        history["actual"].append(actual_demand)
        history["inventory"].append(current_inventory)
        history["order"].append(Q)
        history["S"].append(S)
        history["cost"].append(cumulative_cost)

        # 10️⃣ Visualization update
        df_plot = pd.DataFrame(history)
        with chart_ph.container():
            st.line_chart(
                df_plot[["forecast", "actual", "inventory"]].rename(
                    columns={"forecast": "Forecast", "actual": "Actual", "inventory": "Inventory"}
                )
            )
            st.bar_chart(
                df_plot[["order", "S"]].rename(columns={"order": "Order Qty", "S": "Order-Up-To Level"}).tail(30)
            )

        with metric_ph.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Inventory", f"{current_inventory:.0f}")
            c2.metric("Outstanding Orders", f"{on_order:.0f}")
            c3.metric("Cumulative Cost", f"${cumulative_cost:,.2f}")
            c4.metric("Order Placed", f"{Q:.0f}")

        with log_ph:
            st.write(
                f"**Day {t+1}/{horizon}:** Pred={pred:.1f}, Actual={actual_demand:.1f}, "
                f"Order={Q:.0f}, Arrived={arrivals_today:.0f}, Inv={current_inventory:.1f}"
            )

        # 11️⃣ Update rolling window for next forecast
        next_row = seq_df.iloc[-1:].copy()
        next_row["Predicted_Units_Sold"] = pred
        next_row["Units_Sold"] = actual_demand
        next_row["Inventory_Level"] = current_inventory
        seq_df = pd.concat([seq_df, next_row], ignore_index=True).tail(inv.seq_len)

        time.sleep(demo_speed)

    st.success("✅ Simulation completed!")
    st.balloons()
