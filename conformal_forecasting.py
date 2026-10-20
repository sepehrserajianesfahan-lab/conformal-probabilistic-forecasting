"""
Conformal Probabilistic Time-Series Forecasting Pipeline 
Fetches real meteorological wind telemetry from a verified online repository, 
converts wind speed to electrical power using a standard turbine power curve, 
and applies Split Conformal Prediction for valid uncertainty quantification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def wind_speed_to_power(v: np.ndarray) -> np.ndarray:
    """
    Applies a standard physical wind turbine power curve.
    - Cut-in speed: 3.0 m/s (turbine starts generating power)
    - Rated speed: 12.0 m/s (reaches maximum rated power)
    - Cut-out speed: 25.0 m/s (shut down for safety)
    """
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    power = np.zeros_like(v, dtype=float)
    
    # Cubic region between cut-in and rated speed (P proportional to v^3)
    active_mask = (v >= cut_in) & (v <= rated)
    power[active_mask] = 100.0 * ((v[active_mask] - cut_in) / (rated - cut_in)) ** 3
    
    # Flat rated power region between rated and cut-out speed
    rated_mask = (v > rated) & (v <= cut_out)
    power[rated_mask] = 100.0
    
    return power


def load_and_convert_wind_data() -> pd.DataFrame:
    """
    Loads real open-source weather records and maps wind speed 
    to a physical wind power generation target.
    """
    url = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv"
    df = pd.read_csv(url)
    
    if 'wind' in df.columns:
        # Extract real measured wind speed telemetry
        wind_speed = df['wind'].values
        
        # Convert wind speed to physical power output (MW) using the power curve
        ideal_power = wind_speed_to_power(wind_speed)
        
        # Add realistic micro-variability / operational noise
        np.random.seed(42)
        power_output = np.clip(ideal_power + np.random.normal(0, 3.5, len(wind_speed)), 0, 100.0)
        
        # Construct structured feature DataFrame
        processed_df = pd.DataFrame({
            'wind_speed': wind_speed,
            'power_target': power_output
        })
        
        # Create temporal lag features for time-series forecasting
        processed_df['wind_speed_lag1'] = processed_df['wind_speed'].shift(1)
        processed_df['wind_speed_lag2'] = processed_df['wind_speed'].shift(2)
        processed_df = processed_df.dropna()
        
        return processed_df[['wind_speed', 'wind_speed_lag1', 'wind_speed_lag2', 'power_target']]
    
    raise ValueError("The online repository structure did not contain the 'wind' column.")


class ConformalRegressor:
    """A wrapper implementing Split Conformal Prediction for valid uncertainty intervals."""
    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha  # Significance level (e.g., 0.1 for 90% coverage)
        self.q_hat = 0.0

    def fit(self, X_train, y_train, X_calib, y_calib):
        # 1. Fit underlying point forecasting model
        self.model.fit(X_train, y_train)
        
        # 2. Compute absolute residuals on the calibration set
        calib_preds = self.model.predict(X_calib)
        residuals = np.abs(y_calib - calib_preds)
        
        # 3. Find the (1 - alpha) quantile of absolute residuals with finite-sample correction
        n = len(calib_preds)
        quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.q_hat = np.quantile(residuals, np.clip(quantile_level, 0, 1))

    def predict_interval(self, X):
        preds = self.model.predict(X)
        lower_bound = preds - self.q_hat
        upper_bound = preds + self.q_hat
        return preds, lower_bound, upper_bound


if __name__ == "__main__":
    print("Fetching real weather telemetry and applying physical wind turbine power curve...")
    
    # Load and physically transform dataset
    data = load_and_convert_wind_data()
    X = data[['wind_speed', 'wind_speed_lag1', 'wind_speed_lag2']]
    y = data['power_target']
    
    # Split data: Train (50%), Calibration (25%), Test (25%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_full, y_train_full, test_size=0.33, shuffle=False)
    
    # Initialize and train model with Conformal Prediction wrapper (90% target coverage)
    base_model = GradientBoostingRegressor(random_state=42, n_estimators=100)
    conformal_model = ConformalRegressor(base_model, alpha=0.1)
    
    print("Training Gradient Boosting forecaster and calibrating statistical bounds...")
    conformal_model.fit(X_train, y_train, X_calib, y_calib)
    
    # Evaluate performance on the unseen test set
    preds, lower, upper = conformal_model.predict_interval(X_test)
    
    # Calculate Empirical Metrics
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    mean_width = np.mean(upper - lower)
    
    print(f"\n--- Physical Power Conformal Evaluation Results ---")
    print(f"Target Coverage Level: 90.0%")
    print(f"Empirical Coverage Achieved: {coverage * 100:.2f}%")
    print(f"Average Prediction Interval Width: {mean_width:.2f} MW")
    print(f"Conformal Residual Offset (q_hat): {conformal_model.q_hat:.2f} MW")
