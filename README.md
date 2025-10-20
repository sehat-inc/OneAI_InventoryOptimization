📦 Smart Inventory Optimization Simulator

## Overview

This project is a Streamlit-based real-time simulation app for smart inventory optimization in manufacturing. It integrates AI-powered demand forecasting (via an ensemble model) with deterministic order-up-to inventory control to dynamically maintain optimal stock levels.

Designed for short hackathons and production demos, the app shows how an enterprise like AuraTech Electronics can minimize total cost by balancing holding costs and stockout penalties, powered by machine learning predictions.

## 🧠 Core Concept

In modern manufacturing, managing critical components means finding a balance between:

- 📈 Overstocking → High holding & obsolescence costs
- 📉 Understocking → Lost sales & production delays

This app simulates that tradeoff in real time:

- Uses a trained ensemble model (MLP + LSTM + CNN + Meta-learner) to forecast daily demand.
- Computes an optimal inventory level (S) and order quantity (Q) using a cost-driven deterministic rule.
- Streams an animated simulation showing:
	- Predicted vs actual demand
	- Inventory levels over time
	- Order decisions
	- Accumulated cost metrics

## ⚙️ Features

- ✅ Forecasts daily demand using an ensemble ML model (.pkl)
- ✅ Implements order-up-to (S, Q) inventory policy
- ✅ Calculates holding cost and stockout cost dynamically
- ✅ Real-time simulation & visualization with animated charts
- ✅ Fully interactive Streamlit UI
- ✅ No need for full history data — uses only last 30 days CSV

## 🧩 App Workflow

### Load Model
The .pkl ensemble includes:

- `mlp_model`, `lstm_model`, `cnn_model`, `meta_model`
- `scaler`, `sequence_length`, and `feature_columns`

### Load Data
The app reads your `sku1_last_30days.csv` as the rolling window for context.

### Forecast + Optimize

Forecasts next-day demand using ensemble.

Calculates order-up-to level:

$$ S = \mu_{LT} + z\,\sigma_{LT} $$

where

$$ z = \Phi^{-1}\!\left( \frac{p}{p + h} \right), \quad p = \text{stockout cost per unit}, \; h = \text{holding cost per unit per day}. $$

### Simulate

Each day, the app updates inventory, places orders (with supplier lead time), and computes total cost.

Visualization updates dynamically for each step.

## 📁 Directory Structure

```
📦 OneAI_InventoryOptimization
├── app.py                         # Main Streamlit app
├── ensemble_forecast_model.pkl    # Trained model (root copy)
├── sku1_last_30days.csv           # Last 30 days data (root copy)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── comparsion/                    # Model comparison notebooks & graphs
│   ├── arima.ipynb
│   ├── prophet.ipynb
│   ├── arima_graph.png
│   └── prophet_graph.png
├── inference_files/               # Inference-ready assets
│   ├── ensemble_forecast_model.pkl
│   ├── prediction.py
│   └── sku1_last_30days.csv
└── Research/                      # Research notes and notebooks
	├── optimal-inventory.ipynb
	├── README.md
	└── research_mind_map.png
```

## 🧰 Installation Guide

1️⃣ Clone the Repository

```
git clone https://github.com/sehat-inc/OneAI_InventoryOptimization
cd OneAI_InventoryOptimization
```

2️⃣ Install Dependencies

You can install the required packages using pip:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install streamlit pandas numpy scipy tensorflow scikit-learn matplotlib
```

(You can add `pickle-mixin` or `keras` if required by your model.)

## 🚀 Running the App

1️⃣ Place your files:

- `ensemble_forecast_model.pkl` — trained ensemble model
- `sku1_last_30days.csv` — recent 30-day SKU data

2️⃣ Run Streamlit:

```
streamlit run app.py
```

3️⃣ Configure in the Sidebar:

- Model Path: `ensemble_forecast_model.pkl`
- Last 30 Days CSV: `sku1_last_30days.csv`
- Set lead time, horizon, and cost parameters
- Hit “🚀 Start Simulation”

## 📊 Outputs

- Line Chart: Displays forecasted vs actual demand and inventory levels over time.
- Bar Chart: Shows order quantities and target inventory (S-level).
- KPIs: Real-time metrics for inventory, open orders, total cost, and order decisions.
- Event Log: Daily trace of forecasts, arrivals, and inventory actions.

## 💡 Use Case Examples

- Manufacturing plants forecasting critical components
- Retail chains managing warehouse stock
- Hackathons or research projects in supply chain optimization
- Teaching demos for AI-driven operations management

## 📚 Technical Notes

Deterministic (s, S) control policy

Per-day cost function:

$$ C_t = h \times I_t + p \times \text{UnmetDemand}_t $$

Order decision:

$$ Q_t = \max\bigl(0,\; S - (\text{Inventory} + \text{OnOrder})\bigr) $$

Forecast models can be replaced with any `.pkl` returning a single-day prediction.

## 🧑‍💻 Authors

Developed by:

- Sehat Gang — GIKI

For: OneAI Hackathon

## 🏁 Example Command Recap

```
# Install dependencies
pip install streamlit pandas numpy scipy tensorflow scikit-learn matplotlib

# Launch the simulation
streamlit run app.py
```