import pandas as pd
import numpy as np
from itertools import product
import os

def process_raw_data(data_dir='./data/raw', output_dir='./data/processed'):
    print("Loading raw datasets...")
    train = pd.read_csv(f'{data_dir}/train.csv', parse_dates=['date'])
    stores = pd.read_csv(f'{data_dir}/stores.csv')
    oil = pd.read_csv(f'{data_dir}/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv(f'{data_dir}/holidays_events.csv', parse_dates=['date'])

    # 1. Create the Cartesian Product Grid
    print("Generating Cartesian Product Grid...")
    date_range = pd.date_range(start=train['date'].min(), end=train['date'].max(), freq='D')
    unique_stores = train['store_nbr'].unique()
    unique_families = train['family'].unique()
    
    grid = pd.DataFrame(list(product(date_range, unique_stores, unique_families)), 
                        columns=['date', 'store_nbr', 'family'])
    
    # 2. Merge actual sales into the grid
    print("Merging sales into grid...")
    df = grid.merge(train, on=['date', 'store_nbr', 'family'], how='left')
    df['sales'] = df['sales'].fillna(0)
    df['onpromotion'] = df['onpromotion'].fillna(0)

    # 3. Process Exogenous & Metadata (Updated Holiday Logic)
    print("Processing exogenous variables...")
    oil_indexed = oil.set_index('date').reindex(date_range)
    oil_filled = oil_indexed.assign(dcoilwtico=lambda x: x['dcoilwtico'].ffill().bfill()).reset_index().rename(columns={'index': 'date', 'dcoilwtico': 'oil_price'})

    # New Holiday Deduplication Logic
    holidays['date'] = pd.to_datetime(holidays['date'])
    holidays = holidays[holidays['transferred'] == False].copy()
    
    # Deduplicate by taking the first holiday on that date 
    # (Prevents join explosion when multiple holidays fall on the same day)
    holidays_dedup = holidays.groupby('date')[['type', 'locale']].first().reset_index()
    holidays_dedup['is_holiday'] = 1
    holidays_dedup['is_national'] = (holidays_dedup['locale'] == 'National').astype(int)
    holidays_dedup = holidays_dedup.rename(columns={'type': 'holiday_type'})

    # 4. Execute Relational Merges
    print("Merging features...")
    stores = stores.rename(columns={'type': 'store_type'})
    
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil_filled, on='date', how='left')
    df = df.merge(holidays_dedup, on='date', how='left')
    
    df['is_national'] = df['is_national'].fillna(0)
    df['is_holiday'] = df['is_holiday'].fillna(0)
    
    # Fill NaNs for non-holidays so they aren't destroyed by dropna() later
    df['holiday_type'] = df['holiday_type'].fillna('None')
    df['locale'] = df['locale'].fillna('None')

    # 5. Truncate Dead History
    print("Truncating inactive historical rows...")
    first_sales = df[df['sales'] > 0].groupby(['store_nbr', 'family'])['date'].min().reset_index()
    first_sales = first_sales.rename(columns={'date': 'first_sale_date'})
    df = df.merge(first_sales, on=['store_nbr', 'family'], how='left')
    df['first_sale_date'] = df['first_sale_date'].fillna(df['date'].min())
    df = df[df['date'] >= df['first_sale_date']].copy()
    df = df.drop(columns=['first_sale_date', 'id']) # Drop ID as we don't need it for local evaluation

    # 6. Target Transformation
    df['sales_log1p'] = np.log1p(df['sales'])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = f'{output_dir}/train_processed.parquet'
    df.to_parquet(out_path)
    print(f"Data processing complete. Saved to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    process_raw_data()