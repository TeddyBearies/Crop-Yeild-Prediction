import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_wikipedia_tables():
    """
    Requests geographic state data from Wikipedia for data augmentation.
    Returns a Pandas DataFrame containing 'State' and 'Zone' columns.
    """
    url = 'https://en.wikipedia.org/wiki/States_and_union_territories_of_India'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        req = requests.get(url, headers=headers, timeout=10)
        req.raise_for_status()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(req.content, 'html.parser')
        
        # Extract all structural tables containing the 'wikitable' class
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        # Find the specific table containing States and Zones
        for table in tables:
            th_tags = table.find_all('th')
            headers_text = [th.text.strip() for th in th_tags]
            
            # Check if this is the correct table
            if any('State' in h for h in headers_text) and any('Zone' in h for h in headers_text):
                # Extract rows
                rows = []
                for tr in table.find_all('tr'):
                    # Get both headers and data cells for row processing
                    cells = tr.find_all(['td', 'th'])
                    row = [cell.text.strip() for cell in cells]
                    if len(row) > 0:
                        rows.append(row)
                
                # The first row contains the headers
                if rows:
                    columns = rows[0]
                    data = rows[1:]
                    
                    df = pd.DataFrame(data, columns=columns)
                    
                    # Clean the dataframe to only include relevant columns for merging
                    # We look for the exact names but they might have citations like [A]
                    state_col = [c for c in columns if 'State' in c][0]
                    zone_col = [c for c in columns if 'Zone' in c][0]
                    
                    df_clean = df[[state_col, zone_col]].copy()
                    df_clean.rename(columns={state_col: 'State', zone_col: 'Zone'}, inplace=True)
                    
                    print(f"Scraping successful. Extracted {len(df_clean)} states/regions from Wikipedia.")
                    return df_clean
                    
        print('Scraping warning: Could not locate the expected State/Zone table on Wikipedia.')
        return pd.DataFrame()
        
    except Exception as e:
        print('Error encountered during web scraping:', e)
        return pd.DataFrame()
