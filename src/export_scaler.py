import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

def export_deployment_artifacts():
    print("Loading integrated data...")
    # Load the pre-scraping data (which was before scaling/encoding)
    df_pre = pd.read_csv('../data/processed/crop_yield_pre_scraping.csv')
    
    # Load wiki data and merge (as in Final_Pipeline)
    from scraping import fetch_wikipedia_tables
    from merge_data import merge_supplementary_data
    df_wiki = fetch_wikipedia_tables()
    df_final = merge_supplementary_data(df_pre, df_wiki)
    
    print("Encoding categorical columns...")
    cat_cols = ['Crop', 'Season', 'State', 'Zone']
    df_encoded = pd.get_dummies(df_final, columns=cat_cols, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Extract features matching what the model expects
    X = df_encoded.drop(columns=['yield_per_hectare'])
    
    # Fit the scaler
    print("Fitting scaler...")
    scaler = StandardScaler()
    num_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    scaler.fit(X[num_cols])
    
    # Save the artifacts
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/scaler.joblib')
    print("Scaler saved to models/scaler.joblib")
    
    # Save the feature columns so the web app knows exactly how to build the row
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, '../models/feature_columns.joblib')
    print("Feature columns saved to models/feature_columns.joblib")
    
if __name__ == "__main__":
    export_deployment_artifacts()
