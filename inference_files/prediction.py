import pickle
import numpy as np
import pandas as pd

# ============================
# 1. Load the ensemble model
# ============================
with open('ensemble_forecast_model.pkl', 'rb') as f:
    model_components = pickle.load(f)

mlp_model = model_components['mlp_model']
lstm_model = model_components['lstm_model']
cnn_model = model_components['cnn_model']
meta_model = model_components['meta_model']
scaler = model_components['scaler']
sequence_length = model_components['sequence_length']
feature_columns = model_components['feature_columns']

print("✅ Ensemble model components loaded")

# ============================
# 2. Load the last 30 days CSV
# ============================
df_last30 = pd.read_csv('sku1_last_30days.csv')
print("✅ 30-day context data loaded")
print(df_last30.tail())

# Ensure order of columns is consistent
df_last30 = df_last30[feature_columns]

# ============================
# 3. Get user input for the new day
# ============================
print("\nPlease input numeric values for the next day 👇")

user_values = []
for col in feature_columns:
    val = float(input(f"Enter value for {col}: "))
    user_values.append(val)

user_input = np.array([user_values])  # shape (1,6)
print("\n📥 Your input:", user_input)

# ============================
# 4. Prepare full 30-day + new-day sequence
# ============================
# Drop oldest day, append new one → to simulate real next-day prediction
combined = pd.concat([
    df_last30.iloc[1:],  # keep last 29
    pd.DataFrame(user_input, columns=feature_columns)  # append new input
], ignore_index=True)

# Scale
scaled_sequence = scaler.transform(combined.values)
X_seq = np.expand_dims(scaled_sequence, axis=0)  # shape (1,30,6)

# ============================
# 5. Predict using ensemble
# ============================
mlp_pred = mlp_model.predict(X_seq, verbose=0).flatten()[0]
lstm_pred = lstm_model.predict(X_seq, verbose=0).flatten()[0]
cnn_pred = cnn_model.predict(X_seq, verbose=0).flatten()[0]

meta_input = np.array([[mlp_pred, lstm_pred, cnn_pred]])
ensemble_pred = meta_model.predict(meta_input, verbose=0).flatten()[0]

print(f"\n🎯 Predicted Units Sold for Next Day: {ensemble_pred:.2f}")