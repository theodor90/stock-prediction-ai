# Stock Price Forecasting with ML.NET

A machine learning application that predicts stock closing prices using ML.NET and C# .NET 9.

## 🚀 Features

- **ML.NET Integration**: Uses Microsoft's ML.NET framework for machine learning
- **FastTree Regression**: Employs gradient boosting algorithm for accurate predictions
- **Data Processing**: Loads and processes stock data from CSV files
- **Performance Metrics**: Displays R-Squared and RMSE for model evaluation
- **Real-time Predictions**: Shows actual vs predicted closing prices

## 🛠️ Technologies Used

- **C# 13.0**
- **.NET 9**
- **ML.NET** - Microsoft's machine learning framework
- **FastTree Regression Algorithm**

## 📊 How It Works

1. **Data Loading**: Reads stock data from CSV file with columns: Date, Open, High, Low, Close
2. **Feature Engineering**: Combines Open, High, and Low prices as input features
3. **Model Training**: Uses 80% of data for training with FastTree regression
4. **Model Evaluation**: Tests on remaining 20% and displays performance metrics
5. **Predictions**: Shows comparison between actual and predicted closing prices

## 🏃‍♂️ Getting Started

### Prerequisites
- .NET 9 SDK
- Visual Studio 2022 or VS Code
- Stock data CSV file

### Installation
1. Clone the repository
