import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


url = "https://tatoeba.org/en/downloads?utm_source=chatgpt.com"
data = pd.read_csv(url)
print("Dataset shape:", data.shape)
print(data.head())


X = data.drop(['Class'], axis=1)
y = data['Class']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_normal = X_scaled[y == 0]


autoencoder = Sequential([
    Dense(14, activation='relu', input_shape=(X_normal.shape[1],)),
    Dense(7, activation='relu'),
    Dense(14, activation='relu'),
    Dense(X_normal.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_normal, X_normal, epochs=10, batch_size=256, shuffle=True, verbose=1)

X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)


data['Reconstruction_error'] = mse
mse_normal_data = data[data['Class'] == 0]['Reconstruction_error']
threshold = np.percentile(mse_normal_data, 95)

print(f"\nThreshold for anomaly: {threshold}")

data['Anomaly'] = data['Reconstruction_error'] > threshold


fraud_detected = data[data['Anomaly'] == True]
actual_fraud = data[data['Class'] == 1]

print("\n--- Results ---")
print(f"Total transactions: {len(data)}")
print(f"Actual fraud cases: {len(actual_fraud)}")
print(f"Detected anomalies: {len(fraud_detected)}")

correctly_detected = fraud_detected[fraud_detected['Class'] == 1]
print(f"Correctly detected fraud (True Positives): {len(correctly_detected)}")

if len(actual_fraud) > 0:
    print(f"Recall (Fraud cases detected): {len(correctly_detected) / len(actual_fraud) * 100:.2f}%")
else:
    print("No actual fraud cases in this dataset to calculate recall.")

false_positives = fraud_detected[fraud_detected['Class'] == 0]
print(f"Incorrectly flagged normal (False Positives): {len(false_positives)}")

print("\n--- Sample of Detected Anomalies ---")
print(fraud_detected[['Time', 'Amount', 'Reconstruction_error', 'Class']].head())