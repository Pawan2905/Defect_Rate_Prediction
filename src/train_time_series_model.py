import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import config as cf

# Load and preprocess data
def load_and_resample(filepath):
    data = pd.read_csv(filepath, parse_dates=['DateTime'], index_col='DateTime')
    data = data[['Defect_Rate']]
    data_resampled = data['Defect_Rate'].resample('min').mean().ffill().bfill()
    return data_resampled

# Modified train_test_split for time series
def train_test_split(data, train_size=0.8):
    """
    Split the data into training and testing sets while maintaining temporal order
    
    Parameters:
    data: pandas Series with datetime index
    train_size: float, proportion of data to use for training
    
    Returns:
    train, test: pandas Series
    """
    n = len(data)
    train_size = int(n * train_size)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test

# Prepare data for LSTM model
def prepare_lstm_data(series, seq_length):
    """
    Prepare data for LSTM model by creating sequences
    """
    X, y = [], []
    values = series.values if isinstance(series, pd.Series) else series
    for i in range(len(values) - seq_length):
        X.append(values[i:(i + seq_length)])
        y.append(values[i + seq_length])
    return np.array(X), np.array(y)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Train LSTM model
def train_lstm_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:  # Print loss every 5 epochs
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    return model

# Evaluate model performance
def evaluate_model(model, dataloader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch).detach().numpy()
            predictions.extend(outputs)
            actuals.extend(y_batch.numpy())
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actuals, predictions)
    return predictions, mse, rmse, mape

# Auto ARIMA model training and forecasting
def auto_arima_forecast(train, test):
    """
    Train Auto ARIMA model and generate forecasts
    """
    print("Training ARIMA model...")
    model = auto_arima(train, 
                      seasonal=True,
                      stepwise=True,
                      suppress_warnings=True,
                      error_action="ignore")
    print(f"Best ARIMA model: {model.order}")
    forecast = model.predict(n_periods=len(test))
    return forecast, model

# Plot and save results
def plot_and_save(train, test, lstm_pred, arima_pred, filename):
    """
    Plot actual vs predicted values from both models
    """
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    plt.plot(train.index, train.values, label='Training Data', color='blue', alpha=0.5)
    
    # Plot test data
    plt.plot(test.index, test.values, label='Test Data', color='blue')
    
    # Plot LSTM predictions
    test_indices = test.index[len(test)-len(lstm_pred):]
    plt.plot(test_indices, lstm_pred, label='LSTM Prediction', color='orange')
    
    # Plot ARIMA predictions
    plt.plot(test.index[:len(arima_pred)], arima_pred, label='ARIMA Prediction', color='green')
    
    plt.title('Time Series Prediction Results')
    plt.xlabel('Time')
    plt.ylabel('Defect Rate')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main script
if __name__ == "__main__":
    # Load and resample data
    print("Loading and preprocessing data...")
    filepath = cf.FILEPATH
    data_resampled = load_and_resample(filepath)
    
    # Split data sequentially
    train, test = train_test_split(data_resampled, train_size=0.8)
    print(f"Training set size: {len(train)}, Test set size: {len(test)}")
    
    # Scale the data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    
    # Prepare LSTM data
    seq_length = 10
    X_train, y_train = prepare_lstm_data(train_scaled, seq_length)
    X_test, y_test = prepare_lstm_data(test_scaled, seq_length)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # No shuffling for time series
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train LSTM model
    print("Training LSTM model...")
    model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train_lstm_model(model, train_loader, criterion, optimizer, epochs=20)
    
    # Get LSTM predictions
    lstm_predictions, lstm_mse, lstm_rmse, lstm_mape = evaluate_model(model, test_loader, scaler)
    
    # Get ARIMA predictions
    arima_forecast, arima_model = auto_arima_forecast(train, test)
    
    # Plot results
    plot_and_save(train, test, lstm_predictions, arima_forecast, "../output/model_predictions.png")
    
    # Calculate metrics for both models
    # Ensure we're comparing the same time period
    min_len = min(len(lstm_predictions), len(arima_forecast))
    test_actual = test[seq_length:seq_length + min_len]
    lstm_pred_trimmed = lstm_predictions[:min_len]
    arima_pred_trimmed = arima_forecast[:min_len]
    
    # Calculate and save metrics
    metrics = pd.DataFrame({
        'Model': ['LSTM', 'ARIMA'],
        'MSE': [
            mean_squared_error(test_actual, lstm_pred_trimmed),
            mean_squared_error(test_actual, arima_pred_trimmed)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(test_actual, lstm_pred_trimmed)),
            np.sqrt(mean_squared_error(test_actual, arima_pred_trimmed))
        ],
        'MAPE': [
            mean_absolute_percentage_error(test_actual, lstm_pred_trimmed),
            mean_absolute_percentage_error(test_actual, arima_pred_trimmed)
        ]
    })
    
    metrics.to_csv(cf.OUTPUTPATH + "model_metrics.csv", index=False)
    print("\nMetrics saved to model_metrics.csv")
    print("\nMetrics Summary:")
    print(metrics)