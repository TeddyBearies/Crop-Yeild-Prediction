import pandas as pd

def filter_target_crops(df):
    # Filter the data for Rice and Wheat, as specified in the project scope.
    target_crops = ['Rice', 'Wheat']
    df_filtered = df[df['Crop'].isin(target_crops)].copy()

    # Pick the columns we need. We use 'Annual_Rainfall' because that's the 
    # correct name in the raw data.
    keep_cols = ['State', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

    # Make sure we only pull columns that actually exist in this specific dataset.
    final_cols = [col for col in keep_cols if col in df_filtered.columns]

    return df_filtered[final_cols]


def remove_invalid_rows(df):
    # Drop rows with zero or negative area/production values since they aren't realistic.
    df = df[df['Area'] > 0].copy()
    df = df[df['Production'] >= 0].copy()

    return df


def handle_missing_values(df):
    # If we're missing Area or Production, we can't calculate yield, so drop those rows.
    df_clean = df.dropna(subset=['Area', 'Production']).copy()

    # Fill in missing weather and fertilizer data using the median value 
    # to keep the dataset size consistent.
    impute_cols = ['Annual_Rainfall', 'Fertilizer', 'Pesticide']
    for col in impute_cols:
        if col in df_clean.columns:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)

    return df_clean


def cap_rainfall_outliers(df):
    # Clip extreme rainfall values using the IQR method. 
    # This keeps outliers from messing with the model too much.
    if 'Annual_Rainfall' in df.columns:
        q1 = df['Annual_Rainfall'].quantile(0.25)
        q3 = df['Annual_Rainfall'].quantile(0.75)
        iqr = q3 - q1
        ceiling = q3 + 1.5 * iqr

        df['Annual_Rainfall'] = df['Annual_Rainfall'].clip(upper=ceiling)
        print(f'Rainfall outliers capped at {ceiling:.1f} mm.')

    return df
