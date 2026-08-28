# Logistics Predictive Modeling & Optimization

This project is Task 4 of a logistics data-science assignment.

## Project objective

Predict shipment **Delivery_Time_Hours** and use predictive insights to support transport-capacity optimization.

## Files

- `logistics_shipments.csv` — 1,500-row synthetic shipment dataset.
- `logistics_prediction_optimization.py` — complete Python analysis script.
- `outputs/` — generated model comparison, predictions and optimization results after running the script.

## Dataset columns

| Column | Description |
|---|---|
| Date | Shipment date |
| Region | Shipment region |
| Mode | Road, Air or Rail |
| Distance_km | Shipment distance |
| Shipment_Volume | Number of units in shipment |
| Traffic_Index | Simulated congestion level, 0–1 |
| Weather_Disruption | 0/1 disruption indicator |
| Handling_Hours | Simulated warehouse/handling time |
| Peak_Season | 0/1 November–December indicator |
| DayOfWeek | Day of week |
| Delivery_Time_Hours | Prediction target |
| Transport_Cost | Simulated transport cost |

## Models

1. Ridge Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Models are evaluated using:

- MAE
- RMSE
- R²

A chronological train/test design and `TimeSeriesSplit` are used to reduce temporal leakage.

## Optimization

The script also demonstrates a Mixed-Integer Linear Programming (MILP) allocation problem. It allocates a required shipment volume across Road, Rail and Air options while considering:

- transport cost
- service rate
- capacity
- service-level target

## Installation

```bash
pip install pandas numpy scikit-learn scipy
```

## Run

```bash
python logistics_prediction_optimization.py
```

