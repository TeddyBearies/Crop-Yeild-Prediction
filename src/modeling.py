import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_features(df, target_col='yield_per_hectare'):
    """
    Splits a fully prepared DataFrame into feature matrix X and target vector y.
    Encodes any remaining categorical columns using one-hot encoding.
    Returns X, y.
    """
    df_encoded = df.copy()

    # Encode any remaining object columns
    cat_cols = df_encoded.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    # Convert boolean dummies to int for clean sklearn compatibility
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets.
    Uses an 80/20 split by default with a fixed random seed for reproducibility.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, model_type='random_forest', random_state=42):
    """
    Trains a regression model inside a sklearn Pipeline.
    Supported model_types: 'linear_regression', 'random_forest'.
    Returns the fitted Pipeline.
    """
    if model_type == 'linear_regression':
        regressor = LinearRegression()
    elif model_type == 'random_forest':
        regressor = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'linear_regression' or 'random_forest'.")

    pipeline = Pipeline([('regressor', regressor)])
    pipeline.fit(X_train, y_train)

    print(f'Model trained: {model_type}')
    return pipeline


def tune_random_forest(X_train, y_train, random_state=42):
    """
    Performs hyperparameter tuning on a Random Forest Pipeline using GridSearchCV.
    Searches over n_estimators and max_depth with 5-fold cross-validation.
    Returns the best estimator Pipeline.
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

    print(f'Best params: {grid_search.best_params_}')
    print(f'Best CV R2 : {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluates a fitted model on the test set.
    Returns a dict with MAE, RMSE, and R2 metrics.
    Prints a formatted summary.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'Model': model_name,
        'MAE': round(mean_absolute_error(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'R2': round(r2_score(y_test, y_pred), 4)
    }

    print(f'{model_name:30s} | MAE={metrics["MAE"]:.4f} | RMSE={metrics["RMSE"]:.4f} | R2={metrics["R2"]:.4f}')
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='r2'):
    """
    Runs k-fold cross-validation on a model.
    Returns mean and standard deviation of the chosen scoring metric.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f'CV {scoring} scores: {scores.round(4)}')
    print(f'Mean: {scores.mean():.4f} | Std: {scores.std():.4f}')
    return scores
