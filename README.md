# Conformal Probabilistic Time-Series Forecasting for Wind Energy

A clean, modular Python implementation of **Probabilistic Time-Series Forecasting** combined with **Split Conformal Prediction** for renewable energy generation. 

This repository demonstrates rigorous uncertainty quantification (UQ) to provide valid statistical coverage guarantees for safe operational decision-making in future energy grids.

## Overview
Standard machine learning models output point forecasts, which are insufficient for managing uncertainty and flexibility in multi-energy systems. This repository integrates real meteorological telemetry with a **physical wind turbine power curve** and wraps a machine learning regressor with **Split Conformal Prediction** to construct distribution-free prediction intervals with guaranteed empirical coverage.

## Key Features
* **Physical Power Curve Modeling:** Translates raw wind speed telemetry ($v$) into electrical power ($P$) using a standard cubic turbine model with cut-in, rated, and cut-out parameters.
* **Uncertainty Quantification (UQ):** Generates symmetric upper and lower confidence bounds alongside point predictions.
* **Finite-Sample Statistical Guarantees:** Implements split conformal inference with finite-sample corrections to guarantee valid empirical coverage ($\ge 1 - \alpha$).
* **Reproducible Pipeline:** Fetches verified public weather records directly via automated online data structures.

## Methodology

1. **Physical Transformation:** Raw meteorological wind speed ($v$) is mapped to power generation using:
   $$P(v) = \begin{cases} 0 & v < v_{in} \text{ or } v > v_{out} \\ P_{rated} \left(\frac{v - v_{in}}{v_{rated} - v_{in}}\right)^3 & v_{in} \le v \le v_{rated} \\ P_{rated} & v_{rated} < v \le v_{out} \end{cases}$$
2. **Point Forecasting:** Trains a Gradient Boosting regressor on temporal lag features to predict power output.
3. **Conformal Calibration:** Computes absolute residuals on a held-out calibration set and derives the adjusted $(1 - \alpha)$ quantile threshold ($\hat{q}$) to bound future predictions symmetrically.

## Getting Started

### Prerequisites
* Python 3.8+
* NumPy, Pandas, Scikit-Learn

### Installation
```bash
pip install numpy pandas scikit-learn
