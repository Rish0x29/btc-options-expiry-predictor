import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests

# Bitcoin Options Expiry-Based Price Prediction Algorithm

class BTCExpiryPredictor:
    def __init__(self):
        self.btc_price_history = []
        self.expiry_dates = []
        self.prediction_window = 30  # days
        
    def fetch_btc_price(self, days=90):
        """Fetch historical BTC price data"""
        # Using CoinGecko API as example
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
        try:
            response = requests.get(url)
            data = response.json()
            prices = data['prices']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['date', 'price']]
            
            self.btc_price_history = df
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def set_expiry_dates(self, dates):
        """Set known options expiry dates"""
        self.expiry_dates = [pd.to_datetime(date) for date in dates]
        
    def calculate_expiry_impact(self, current_date, expiry_date, days_before=7):
        """Calculate impact score based on proximity to expiry"""
        days_to_expiry = (expiry_date - current_date).days
        
        if days_to_expiry < 0:
            return 0
        elif days_to_expiry <= days_before:
            # Higher impact as we approach expiry
            return (days_before - days_to_expiry) / days_before
        else:
            return 0
            
    def analyze_historical_expiry_patterns(self):
        """Analyze price behavior around historical expiry dates"""
        patterns = []
        
        for expiry in self.expiry_dates:
            # Get price data around expiry
            start_date = expiry - timedelta(days=14)
            end_date = expiry + timedelta(days=7)
            
            mask = (self.btc_price_history['date'] >= start_date) & \
                   (self.btc_price_history['date'] <= end_date)
            expiry_data = self.btc_price_history[mask]
            
            if len(expiry_data) > 0:
                pre_expiry_price = expiry_data[expiry_data['date'] < expiry]['price'].mean()
                post_expiry_price = expiry_data[expiry_data['date'] >= expiry]['price'].mean()
                
                if pre_expiry_price > 0:
                    price_change_pct = ((post_expiry_price - pre_expiry_price) / pre_expiry_price) * 100
                    patterns.append({
                        'expiry_date': expiry,
                        'pre_expiry_price': pre_expiry_price,
                        'post_expiry_price': post_expiry_price,
                        'change_pct': price_change_pct
                    })
                    
        return pd.DataFrame(patterns)
    
    def predict_price(self, target_date, next_expiry_date):
        """Predict BTC price for target date based on expiry analysis"""
        if len(self.btc_price_history) == 0:
            return None
            
        # Get current price
        current_price = self.btc_price_history['price'].iloc[-1]
        current_date = self.btc_price_history['date'].iloc[-1]
        
        # Calculate impact of upcoming expiry
        expiry_impact = self.calculate_expiry_impact(target_date, next_expiry_date)
        
        # Analyze historical patterns
        historical_patterns = self.analyze_historical_expiry_patterns()
        
        if len(historical_patterns) > 0:
            avg_change = historical_patterns['change_pct'].mean()
            std_change = historical_patterns['change_pct'].std()
        else:
            avg_change = 0
            std_change = 5
        
        # Calculate predicted price with expiry influence
        base_prediction = current_price
        expiry_adjustment = (avg_change / 100) * expiry_impact
        predicted_price = base_prediction * (1 + expiry_adjustment)
        
        # Calculate confidence bounds
        upper_bound = predicted_price * (1 + (std_change / 100))
        lower_bound = predicted_price * (1 - (std_change / 100))
        
        return {
            'target_date': target_date,
            'predicted_price': predicted_price,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'expiry_impact': expiry_impact,
            'confidence': 1 - (expiry_impact * 0.5)  # Lower confidence near expiry
        }
    
    def plot_predictions(self, predictions):
        """Plot historical prices and predictions"""
        plt.figure(figsize=(14, 8))
        
        # Plot historical prices
        plt.subplot(2, 1, 1)
        plt.plot(self.btc_price_history['date'], self.btc_price_history['price'], 
                label='Historical Price', color='blue', linewidth=2)
        
        # Mark expiry dates
        for expiry in self.expiry_dates:
            plt.axvline(x=expiry, color='red', linestyle='--', alpha=0.5, label='Expiry Date')
        
        plt.title('Bitcoin Price History with Options Expiry Dates', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot predictions
        plt.subplot(2, 1, 2)
        dates = [p['target_date'] for p in predictions]
        prices = [p['predicted_price'] for p in predictions]
        upper = [p['upper_bound'] for p in predictions]
        lower = [p['lower_bound'] for p in predictions]
        
        plt.plot(dates, prices, label='Predicted Price', color='green', linewidth=2, marker='o')
        plt.fill_between(dates, lower, upper, alpha=0.3, color='green', label='Confidence Interval')
        
        plt.title('BTC Price Predictions Based on Options Expiry Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('btc_expiry_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plot saved as 'btc_expiry_predictions.png'")


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BTCExpiryPredictor()
    
    # Fetch historical BTC data
    print("Fetching BTC price data...")
    predictor.fetch_btc_price(days=90)
    
    # Set known expiry dates (example dates - replace with actual expiry dates)
    expiry_dates = [
        '2024-12-27',  # End of year expiry
        '2025-01-31',  # End of month expiry
        '2025-02-28',  # Next month expiry
    ]
    predictor.set_expiry_dates(expiry_dates)
    
    # Generate predictions for next 30 days
    print("\nGenerating predictions...")
    predictions = []
    start_date = datetime.now()
    next_expiry = pd.to_datetime('2025-01-31')
    
    for i in range(30):
        target_date = start_date + timedelta(days=i)
        prediction = predictor.predict_price(target_date, next_expiry)
        if prediction:
            predictions.append(prediction)
            print(f"Date: {prediction['target_date'].strftime('%Y-%m-%d')} | "
                  f"Predicted Price: ${prediction['predicted_price']:,.2f} | "
                  f"Range: ${prediction['lower_bound']:,.2f} - ${prediction['upper_bound']:,.2f}")
    
    # Plot results
    print("\nGenerating plots...")
    predictor.plot_predictions(predictions)
    
    print("\nAnalysis complete!")
