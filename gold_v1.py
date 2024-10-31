import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GoldPricePredictor:
    def __init__(self, ticker="GLD", lookback_days=365):
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.model = None
        self.last_predictions = None
        
    def fetch_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                         end=end_date.strftime("%Y-%m-%d"), 
                         interval='1d')
        return df[['Close']].dropna()
    
    def prepare_features(self, df):
        df['S_3'] = df['Close'].rolling(window=3).mean()
        df['S_9'] = df['Close'].rolling(window=9).mean()
        df['next_day_price'] = df['Close'].shift(-1)
        df['volatility'] = df['Close'].rolling(window=5).std()
        
        return df.dropna()
    
    def train_model(self, train_size=0.8):
        df = self.fetch_data()
        df = self.prepare_features(df)
        
        X = df[['S_3', 'S_9', 'volatility']]
        y = df['next_day_price']
        
        split_idx = int(train_size * len(df))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        test_predictions = self.model.predict(X_test)
        self.last_predictions = pd.DataFrame(test_predictions, 
                                          index=y_test.index, 
                                          columns=['predicted_price'])
        
        metrics = {
            'MSE': mean_squared_error(y_test, test_predictions),
            'MAE': mean_absolute_error(y_test, test_predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, test_predictions)),
            'R²': self.model.score(X_test, y_test)
        }
        
        return metrics, df['Close'][-len(self.last_predictions):], self.last_predictions

    def generate_trading_signals(self):
        if self.last_predictions is None:
            raise ValueError("Must train model first")
            
        signals = pd.DataFrame(index=self.last_predictions.index)
        signals['predicted_price'] = self.last_predictions['predicted_price']
        signals['signal'] = np.where(
            signals.predicted_price.shift(1) < signals.predicted_price,
            "Buy", "No Position"
        )
        signals['predicted_return'] = signals['predicted_price'].pct_change() * 100
        
        return signals

def main():
    st.set_page_config(page_title="Gold ETF Price Prediction", layout="wide")
    
    # Title and description
    st.title("Gold ETF Price Prediction Dashboard")
    st.markdown("This app predicts Gold ETF (GLD) prices using a linear regression model based on moving averages.")
    
    # Set default values instead of using sliders
    lookback_days = 365
    train_size = 0.8
    
    # Initialize predictor
    predictor = GoldPricePredictor(lookback_days=lookback_days)
    
    # Plot predictions vs actual prices
    with st.spinner("Training model..."):
        metrics, actual_prices, predictions = predictor.train_model(train_size=train_size)
        
    # Create and display the main chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_prices.index,
        y=actual_prices.values,
        name="Actual Price",
        line=dict(color="blue")
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions['predicted_price'].values,
        name="Predicted Price",
        line=dict(color="red")
    ))
    
    fig.update_layout(
        title="Gold ETF Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        height=400  # Reduced height for better layout
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics in columns
    st.header("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
    col2.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
    col3.metric("Root Mean Squared Error", f"{metrics['RMSE']:.4f}")
    col4.metric("R² Score", f"{metrics['R²']:.4f}")
    
    # Generate and display trading signals
    st.header("Latest Trading Signals")
    signals = predictor.generate_trading_signals()
    latest_signals = signals.tail(5)
    
    def color_signals(val):
        color = 'green' if val == 'Buy' else 'red'
        return f'color: {color}'
    
    styled_signals = latest_signals.style.applymap(
        color_signals, subset=['signal']
    ).format({
        'predicted_price': '{:.2f}',
        'predicted_return': '{:.2f}%'
    })
    
    st.dataframe(styled_signals)
    
    # Add download button for full results
    full_results = pd.concat([
        actual_prices, 
        predictions, 
        signals['signal'], 
        signals['predicted_return']
    ], axis=1)
    full_results.columns = ['Actual Price', 'Predicted Price', 'Signal', 'Predicted Return (%)']
    
    st.download_button(
        label="Download Full Results",
        data=full_results.to_csv().encode('utf-8'),
        file_name='gold_predictions.csv',
        mime='text/csv'
    )
    
    # Footer
    st.markdown("<div style='text-align: center; color: black; padding: 24px;'>Developed by Anupam Kabade, +91 9008816799</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
