import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_features(df, target_col='yield_per_hectare'):
    """
    Splits the data into features (X) and the target (y).
    It also handles one-hot encoding for any categorical columns left in the data.
    """
    df_encoded = df.copy()

    # Turn categorical text columns into numbers
    cat_cols = df_encoded.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    # Convert the boolean columns to 0/1 integers for better compatibility
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    Default is an 80/20 split with a fixed seed so we get the same results every time.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, model_type='random_forest', random_state=42):
    """
    Trains either a Linear Regression or Random Forest model inside a pipeline.
    """
    if model_type == 'linear_regression':
        regressor = LinearRegression()
    elif model_type == 'random_forest':
        regressor = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Whoops, {model_type} isn't supported. Use 'linear_regression' or 'random_forest'.")

    pipeline = Pipeline([('regressor', regressor)])
    pipeline.fit(X_train, y_train)

    print(f'Done training: {model_type}')
    return pipeline


def tune_random_forest(X_train, y_train, random_state=42):
    """
    Uses GridSearch to find the best settings for the Random Forest model.
    It tests different tree counts and depths using 5-fold cross-validation.
    """
    pipeline = Pipeline([
        ('regressor', RandomForestRegressor(random_state=random_state))
    ])

    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [5, 10, None]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f'Best settings found: {grid_search.best_params_}')
    print(f'Best R2 score from CV: {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Tests the model on the hold-out set and returns MAE, RMSE, and R2 metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'Model': model_name,
        'MAE': round(mean_absolute_error(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'R2': round(r2_score(y_test, y_pred), 4)
    }

    # Print a nice summary line
    print(f'{model_name:30s} | MAE={metrics["MAE"]:.4f} | RMSE={metrics["RMSE"]:.4f} | R2={metrics["R2"]:.4f}')
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='r2'):
    """
    Runs cross-validation to see how stable the model is across different data splits.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f'CV scores: {scores.round(4)}')
    print(f'Average: {scores.mean():.4f} | Std Dev: {scores.std():.4f}')
    return scores
