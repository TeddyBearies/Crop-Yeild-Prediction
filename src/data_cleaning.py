import pandas as pd

def filter_target_crops(df):
    # Focus analysis exclusively on Rice and Wheat crops
    target_crops = ['Rice', 'Wheat']
    
    # Filter dataset based on target crops
    df_filtered = df[df['Crop'].isin(target_crops)].copy()
    
    # Retain only the relevant columns as specified in the assignment brief
    keep_cols = ['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production', 'Rainfall', 'Temperature', 'Fertilizer']
    
    # Ensure missing columns do not cause execution errors
    final_cols = []
    for col in keep_cols:
        if col in df_filtered.columns:
            final_cols.append(col)
            
    return df_filtered[final_cols]

def remove_invalid_rows(df):
    # Remove anomalous records where area or production are zero or negative
    df = df[df['Area'] > 0].copy()
    df = df[df['Production'] >= 0].copy()
    
    return df

def handle_missing_values(df):
    # Drop records missing key variables required to calculate yield
    df_clean = df.dropna(subset=['Area', 'Production']).copy()
    
    # Impute missing weather data using the median to maintain dataset volume
    weather_cols = ['Rainfall', 'Temperature', 'Fertilizer']
    for col in weather_cols:
        if col in df_clean.columns:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
            
    return df_clean

def cap_rainfall_outliers(df):
    # Cap extreme rainfall outliers using the IQR method to improve model stability
    if 'Rainfall' in df.columns:
        q1 = df['Rainfall'].quantile(0.25)
        q3 = df['Rainfall'].quantile(0.75)
        iqr = q3 - q1
        ceiling = q3 + 1.5 * iqr
        
        # Apply the upper bound cap
        df['Rainfall'] = df['Rainfall'].clip(upper=ceiling)
        print('Outliers in rainfall have been capped.')
        
    return df
