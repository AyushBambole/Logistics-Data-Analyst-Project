"""
Logistics Predictive Modeling & Optimization
=============================================
Task 4 project script.

Input:
    logistics_shipments.csv

What this script does:
1. Loads and profiles the simulated logistics dataset.
2. Performs a chronological train/validation/test split.
3. Trains Ridge Regression, Random Forest and Gradient Boosting models.
4. Evaluates models using MAE, RMSE and R².
5. Tunes Gradient Boosting using TimeSeriesSplit.
6. Runs a simple MILP transport-allocation optimization.
7. Saves model comparison and test predictions.

Install:
    pip install pandas numpy scikit-learn scipy

Run:
    python logistics_prediction_optimization.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.optimize import milp, LinearConstraint, Bounds


DATA_FILE = Path("logistics_shipments.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print("\nDataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nSummary:")
    print(df.describe(include="all").transpose())

    return df


def split_data(df):
    """Chronological split: 70% train, 15% validation, 15% test."""
    cut1 = df["Date"].quantile(0.70)
    cut2 = df["Date"].quantile(0.85)

    train = df[df["Date"] <= cut1].copy()
    valid = df[(df["Date"] > cut1) & (df["Date"] <= cut2)].copy()
    test = df[df["Date"] > cut2].copy()

    print("\nSplit dates:")
    print("Train through:", cut1.date())
    print("Validation:", cut1.date(), "to", cut2.date())
    print("Test after:", cut2.date())
    print("\nRows:", len(train), len(valid), len(test))

    return train, valid, test


def build_preprocessor():
    features = [
        "Region",
        "Mode",
        "Distance_km",
        "Shipment_Volume",
        "Traffic_Index",
        "Weather_Disruption",
        "Handling_Hours",
        "Peak_Season",
        "DayOfWeek",
    ]
    categorical = ["Region", "Mode"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            )
        ],
        remainder="passthrough",
    )

    return features, preprocess


def evaluate_models(train, test, features, preprocess):
    models = {
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            loss="huber",
            random_state=42,
        ),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        pipeline.fit(
            train[features],
            train["Delivery_Time_Hours"],
        )

        pred = pipeline.predict(test[features])

        mae = mean_absolute_error(
            test["Delivery_Time_Hours"], pred
        )
        rmse = mean_squared_error(
            test["Delivery_Time_Hours"], pred
        ) ** 0.5
        r2 = r2_score(
            test["Delivery_Time_Hours"], pred
        )

        results.append(
            {
                "Model": name,
                "MAE_hours": mae,
                "RMSE_hours": rmse,
                "R2": r2,
            }
        )
        fitted[name] = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE_hours")
        .reset_index(drop=True)
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    return results_df, fitted


def tune_gradient_boosting(train, test, features, preprocess):
    """Time-aware hyperparameter search."""
    tscv = TimeSeriesSplit(n_splits=4)

    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=42
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 180],
        "model__learning_rate": [0.03, 0.05],
        "model__max_depth": [2, 3],
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    search.fit(
        train[features],
        train["Delivery_Time_Hours"],
    )

    pred = search.predict(test[features])

    metrics = {
        "MAE_hours": mean_absolute_error(
            test["Delivery_Time_Hours"], pred
        ),
        "RMSE_hours": mean_squared_error(
            test["Delivery_Time_Hours"], pred
        ) ** 0.5,
        "R2": r2_score(
            test["Delivery_Time_Hours"], pred
        ),
    }

    predictions = test[
        ["Date", "Region", "Mode", "Delivery_Time_Hours"]
    ].copy()
    predictions["Predicted_Delivery_Time_Hours"] = pred
    predictions["Absolute_Error_Hours"] = (
        predictions["Delivery_Time_Hours"]
        - predictions["Predicted_Delivery_Time_Hours"]
    ).abs()

    predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\nBest Gradient Boosting parameters:")
    print(search.best_params_)
    print("\nTuned Gradient Boosting metrics:")
    print(metrics)

    return search, metrics


def optimize_transport_allocation():
    """
    Simple MILP example.

    Decision variables:
        Number of shipments allocated to each option.

    Objective:
        Minimize transport cost plus a penalty for service
        options below the target service level.
    """
    options = [
        "Road standard",
        "Road priority",
        "Rail standard",
        "Air priority",
    ]

    unit_cost = np.array(
        [115, 145, 98, 205], dtype=float
    )

    service_rate = np.array(
        [0.91, 0.96, 0.88, 0.985], dtype=float
    )

    capacity = np.array(
        [420, 300, 500, 180], dtype=float
    )

    required_shipments = 900
    service_target = 0.93
    penalty = 180

    objective = (
        unit_cost
        + penalty
        * np.maximum(
            0,
            service_target - service_rate,
        )
    )

    # First constraint: allocate exactly required shipments.
    # Second: weighted service capacity must reach the target.
    A = np.array(
        [
            [1, 1, 1, 1],
            [
                1 / service_rate[0],
                1 / service_rate[1],
                1 / service_rate[2],
                1 / service_rate[3],
            ],
        ],
        dtype=float,
    )

    constraints = LinearConstraint(
        A,
        [required_shipments, -np.inf],
        [required_shipments, 1050],
    )

    result = milp(
        c=objective,
        integrality=np.ones(len(options)),
        bounds=Bounds(0, capacity),
        constraints=constraints,
    )

    if not result.success:
        print("\nOptimization failed:")
        print(result.message)
        return None

    allocation = np.rint(result.x).astype(int)

    allocation_df = pd.DataFrame(
        {
            "Transport_Option": options,
            "Allocated_Shipments": allocation,
            "Unit_Cost": unit_cost,
            "Service_Rate": service_rate,
            "Capacity": capacity.astype(int),
        }
    )

    allocation_df["Estimated_Cost"] = (
        allocation_df["Allocated_Shipments"]
        * allocation_df["Unit_Cost"]
    )

    allocation_df.to_csv(
        OUTPUT_DIR / "optimized_transport_allocation.csv",
        index=False,
    )

    print("\nOptimized allocation:")
    print(allocation_df.to_string(index=False))
    print(
        "\nTotal optimized cost:",
        allocation_df["Estimated_Cost"].sum(),
    )

    return allocation_df


def main():
    df = load_data()

    features, preprocess = build_preprocessor()
    train, valid, test = split_data(df)

    results, fitted = evaluate_models(
        train,
        test,
        features,
        preprocess,
    )

    print("\nBaseline model comparison:")
    print(results.to_string(index=False))

    best_model = results.iloc[0]["Model"]
    print("\nBest baseline model:", best_model)

    tuned_model, tuned_metrics = tune_gradient_boosting(
        train,
        test,
        features,
        preprocess,
    )

    optimize_transport_allocation()

    print("\nCompleted.")
    print("Files saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
