# Step 1: Import required libraries
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 2: Download historical stock data for Apple, Amazon, and NVIDIA
companies = ['AAPL', 'AMZN', 'NVDA']
end_date = dt.datetime.now().date()
start_date = end_date - pd.Timedelta(days=365 * 2)  # Last 2 years of data

# Download stock data
data = yf.download(companies, start=start_date, end=end_date)['Close']

# Fill any missing values by forward filling
data.fillna(method='ffill', inplace=True)

# Step 3: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare training data (using the last 60 days as a lookback period)
lookback = 60
X_train, y_train = [], []

for i in range(lookback, len(scaled_data)):
    X_train.append(scaled_data[i-lookback:i])
    y_train.append(scaled_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)

# Step 4: Build and train the LSTM model
model = Sequential()

# Add LSTM layers with Dropout
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=X_train.shape[2]))  # Output for each stock

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Step 5: Make predictions for the current week (5 business days)
latest_data = scaled_data[-lookback:]
latest_data = np.reshape(latest_data, (1, latest_data.shape[0], latest_data.shape[1]))

future_predictions = []
for _ in range(5):  # Predict for the next 5 business days
    pred = model.predict(latest_data)
    future_predictions.append(pred)
    latest_data = np.append(latest_data[:, 1:, :], pred.reshape(1, 1, -1), axis=1)

future_predictions = np.array(future_predictions).reshape(5, -1)
future_predictions = scaler.inverse_transform(future_predictions)

# Create a DataFrame for future predictions
pred_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=5, freq='B')  # Only business days
future_predictions_df = pd.DataFrame(future_predictions, index=pred_dates, columns=companies)

# Step 6: Combine historical and predicted data
historical_data = data.iloc[-60:]  # Use the last 60 days of historical data
combined_df = pd.concat([historical_data, future_predictions_df])

# Add a 'Type' column to differentiate actual and predicted data
combined_df['Type'] = ['Actual'] * len(historical_data) + ['Predicted'] * len(future_predictions_df)

# Reset the index for Power BI compatibility
combined_df.reset_index(inplace=True)
combined_df.rename(columns={'index': 'Date'}, inplace=True)

# Step 7: Model Evaluation - Metrics and Visualizations

# 1. Calculate evaluation metrics for the predicted vs actual
# For simplicity, let's take the last week's actual data and compare it with the predicted data
actual_prices = data[companies].iloc[-10:].values  # Last 10 days of actual prices
predicted_prices = future_predictions_df[companies].values  # Predicted prices for the next 5 days

# Calculate error metrics
mse = mean_squared_error(actual_prices[-5:], predicted_prices)
mae = mean_absolute_error(actual_prices[-5:], predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices[-5:], predicted_prices)

# Print error metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 2. Visualize the actual vs predicted prices for the last 60 days
plt.figure(figsize=(14, 7))
for company in companies:
    plt.plot(historical_data.index, historical_data[company], label=f'{company} Actual', color='blue')
    plt.plot(future_predictions_df.index, future_predictions_df[company], label=f'{company} Predicted', linestyle='--', color='red')
plt.title('Predicted vs Actual Stock Prices (Past 60 Days)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap of Stock Prices
plt.figure(figsize=(8, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Stock Prices')
plt.tight_layout()
plt.show()

# 4. Prediction Error Plot (Actual vs Predicted for the current week)
# Calculate the prediction errors for the past 5 days
error_df = pd.DataFrame(columns=companies)
for company in companies:
    actual_prices = data[company].iloc[-10:].values  # Last week's actual prices
    predicted_prices = future_predictions_df[company].values  # Predicted prices for the current week
    error = predicted_prices - actual_prices[-5:]  # Compare last 5 days
    error_df[company] = error

plt.figure(figsize=(14, 7))
for company in companies:
    plt.plot(error_df.index, error_df[company], label=f'{company} Prediction Error')
plt.title('Prediction Error (Predicted vs Actual)')
plt.xlabel('Date')
plt.ylabel('Prediction Error')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 8: Save the combined DataFrame to a CSV file for Power BI
combined_df.to_csv('stock_predictions_combined.csv', index=False)
print("Combined data saved to 'stock_predictions_combined.csv'")
