import pandas as pd

def merge_supplementary_data(df_main, df_scraped):
    """
    Merges the primary agricultural dataset with the scraped geographical dataset.
    Performs a left join on the 'State' column to enrich the main data.
    """
    if df_scraped.empty:
        print('Warning: Scraped DataFrame is empty. Returning original DataFrame without merge.')
        return df_main.copy()
        
    print('Merging primary dataset with scraped Wikipedia data on [State]...')
    
    # Strip whitespace from merge keys to ensure clean joins
    df_main_clean = df_main.copy()
    df_main_clean['State'] = df_main_clean['State'].str.strip()
    
    df_scraped_clean = df_scraped.copy()
    df_scraped_clean['State'] = df_scraped_clean['State'].str.strip()
    
    # Perform a left join to keep all original agricultural records
    df_merged = pd.merge(df_main_clean, df_scraped_clean, on='State', how='left')
    
    print(f'Merge complete. New dataset shape: {df_merged.shape}')
    return df_merged
