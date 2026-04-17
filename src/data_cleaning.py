import pandas as pd

def filter_target_crops(df):
    # Focus analysis exclusively on Rice and Wheat crops to address the business question
    target_crops = ['Rice', 'Wheat']

    # Filter dataset based on target crops
    df_filtered = df[df['Crop'].isin(target_crops)].copy()

    # Retain only the relevant columns that actually exist in the raw dataset
    # Note: The raw dataset uses 'Annual_Rainfall' not 'Rainfall', and has no Temperature or District column
    keep_cols = ['State', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

    # Silently skip any column not found so the function is robust to dataset variations
    final_cols = [col for col in keep_cols if col in df_filtered.columns]

    return df_filtered[final_cols]


def remove_invalid_rows(df):
    # Remove anomalous records where area or production are zero or negative
    df = df[df['Area'] > 0].copy()
    df = df[df['Production'] >= 0].copy()

    return df


def handle_missing_values(df):
    # Drop records missing key variables required to calculate yield
    df_clean = df.dropna(subset=['Area', 'Production']).copy()

    # Impute missing climate and input data using the median to maintain dataset volume
    # Note: column is Annual_Rainfall in the raw dataset, not Rainfall
    impute_cols = ['Annual_Rainfall', 'Fertilizer', 'Pesticide']
    for col in impute_cols:
        if col in df_clean.columns:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)

    return df_clean


def cap_rainfall_outliers(df):
    # Cap extreme Annual_Rainfall outliers using the IQR method to improve model stability
    if 'Annual_Rainfall' in df.columns:
        q1 = df['Annual_Rainfall'].quantile(0.25)
        q3 = df['Annual_Rainfall'].quantile(0.75)
        iqr = q3 - q1
        ceiling = q3 + 1.5 * iqr

        # Apply the upper bound cap
        df['Annual_Rainfall'] = df['Annual_Rainfall'].clip(upper=ceiling)
        print(f'Outliers in Annual_Rainfall capped at {ceiling:.1f} mm.')

    return df
