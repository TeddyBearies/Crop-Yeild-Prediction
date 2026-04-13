import pandas as pd

def create_yield_per_hectare(df):
    # Create the dependent variable (expected yield), defined as production per hectare
    df = df.copy()
    df['yield_per_hectare'] = df['Production'] / df['Area']
    
    print('Feature engineering complete: yield_per_hectare added.')
    return df
