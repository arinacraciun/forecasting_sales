import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest  
import seaborn as sns

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def run_imputation_experiment(train_final, output_dir='./graphs'):
    """
    Recreates the data disaster experiment to prove Seasonal Interpolation 
    is the superior imputation method, then saves the visualization.
    """
    print("Running imputation experiment...")
    os.makedirs(output_dir, exist_ok=True)

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_final['date']):
        train_final['date'] = pd.to_datetime(train_final['date'])

    # 1. Isolate Store 1 / Bread
    ts = train_final[
        (train_final['store_nbr'] == 1) & 
        (train_final['family'] == 'BREAD/BAKERY')
    ].set_index('date')[['sales']].sort_index()

    # 2. Create Artificial Gap (Feb 1 - Feb 14, 2016)
    ts['sales_clean'] = ts['sales']
    ts['sales_missing'] = ts['sales'].copy()
    missing_window = slice('2016-02-01', '2016-02-14')
    ts.loc[missing_window, 'sales_missing'] = np.nan

    # 3. Method A: Naive (Previous Week)
    ts['impute_naive'] = ts['sales_missing'].fillna(ts['sales_missing'].shift(7)).fillna(ts['sales_missing'].shift(-7))

    # 4. Method B: Day-of-Week Profile
    ts['day_of_week'] = ts.index.dayofweek
    day_profile = ts.groupby('day_of_week')['sales_missing'].mean()

    ts['impute_profile'] = ts['sales_missing'].copy()
    for day in range(7):
        mask = (ts['day_of_week'] == day) & (ts['sales_missing'].isna())
        ts.loc[mask, 'impute_profile'] = day_profile[day]

    # 5. Method C: Seasonal Interpolation
    ts['residuals'] = ts['sales_missing'] - ts['day_of_week'].map(day_profile)
    ts['residuals_interp'] = ts['residuals'].interpolate(method='time')
    ts['impute_seasonal'] = ts['residuals_interp'] + ts['day_of_week'].map(day_profile)

    # Evaluation Output
    methods = {
        'Naive (Prev Week)': ts['impute_naive'],
        'Day Profile': ts['impute_profile'],
        'Seasonal Interp': ts['impute_seasonal']
    }

    print("\n--- Imputation MAE Results ---")
    for name, series in methods.items():
        score = mean_absolute_error(
            ts.loc[missing_window, 'sales_clean'], 
            series.loc[missing_window]
        )
        print(f"{name}: {score:.2f}")

    # 6. Plotting the winner
    plt.figure(figsize=(12, 5))
    plt.plot(ts.loc[missing_window, 'sales_clean'], label='True Sales', linewidth=2, color='silver')
    plt.plot(ts.loc[missing_window, 'impute_seasonal'], label='Seasonal Interpolation', linestyle='--', color='blue')
    plt.title("Winning Method: Seasonal Interpolation Reconstruction")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    
    out_path = f'{output_dir}/imputation_winner.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved winning visualization to {out_path}\n")

# def plot_diagnostics(train_df, output_dir='./graphs'):
#     """Generates ACF, Boxplots, and STL decomposition to justify feature engineering."""
#     print("Generating diagnostic plots...")
#     os.makedirs(output_dir, exist_ok=True)
    
#     if not pd.api.types.is_datetime64_any_dtype(train_df['date']):
#         train_df['date'] = pd.to_datetime(train_df['date'])
        
#     # Aggregate to global daily sales for macro-diagnostics
#     global_sales = train_df.groupby('date')['sales'].sum().reset_index()
#     global_sales.set_index('date', inplace=True)
#     global_sales['day_of_week'] = global_sales.index.dayofweek
#     global_sales['month'] = global_sales.index.month

#     # 1. Seasonality Boxplots
#     fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#     sns.boxplot(data=global_sales, x='day_of_week', y='sales', ax=axes[0])
#     axes[0].set_title('Weekly Seasonality (0=Monday)')
#     sns.boxplot(data=global_sales, x='month', y='sales', ax=axes[1])
#     axes[1].set_title('Yearly Seasonality by Month')
#     plt.savefig(f'{output_dir}/seasonality_boxplots.png')
#     plt.close()

#     # 2. Autocorrelation (ACF)
#     plt.figure(figsize=(12, 4))
#     plot_acf(global_sales['sales'], lags=35, ax=plt.gca())
#     plt.title('Autocorrelation Function (ACF) - Daily Sales')
#     plt.savefig(f'{output_dir}/acf_plot.png')
#     plt.close()

#     # 3. STL Decomposition
#     stl = STL(global_sales['sales'], period=7, robust=True)
#     res = stl.fit()
#     fig = res.plot()
#     fig.set_size_inches(15, 8)
#     plt.suptitle('STL Decomposition of Global Sales (Weekly Period)', y=1.02)
#     plt.savefig(f'{output_dir}/stl_decomposition.png')
#     plt.close()
    
#     print("Diagnostic plots saved successfully.")

def plot_diagnostics(train_df, output_dir='./graphs'):
    """Generates ACF, Boxplots, STL decomposition, and Outlier Comparison plots to justify feature engineering."""
    print("Generating diagnostic plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    if not pd.api.types.is_datetime64_any_dtype(train_df['date']):
        train_df['date'] = pd.to_datetime(train_df['date'])
        
    # --- Aggregation Step (Adjusted for Multivariate Features) ---
    # Sales and promotions are summed across all stores; oil price is a nationwide constant per day, so mean/first works.
    agg_dict = {'sales': 'sum'}
    if 'onpromotion' in train_df.columns:
        agg_dict['onpromotion'] = 'sum'
    if 'dcoilwtico' in train_df.columns:
        agg_dict['dcoilwtico'] = 'mean'

    global_sales = train_df.groupby('date').agg(agg_dict).reset_index()
    global_sales.set_index('date', inplace=True)
    global_sales['day_of_week'] = global_sales.index.dayofweek
    global_sales['month'] = global_sales.index.month

    # 1. Seasonality Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.boxplot(data=global_sales, x='day_of_week', y='sales', ax=axes[0])
    axes[0].set_title('Weekly Seasonality (0=Monday)')
    sns.boxplot(data=global_sales, x='month', y='sales', ax=axes[1])
    axes[1].set_title('Yearly Seasonality by Month')
    plt.savefig(f'{output_dir}/seasonality_boxplots.png')
    plt.close()

    # 2. Autocorrelation (ACF)
    plt.figure(figsize=(12, 4))
    plot_acf(global_sales['sales'], lags=35, ax=plt.gca())
    plt.title('Autocorrelation Function (ACF) - Daily Sales')
    plt.savefig(f'{output_dir}/acf_plot.png')
    plt.close()

    # 3. STL Decomposition
    stl = STL(global_sales['sales'], period=7, robust=True)
    res = stl.fit()
    fig = res.plot()
    fig.set_size_inches(15, 8)
    plt.suptitle('STL Decomposition of Global Sales (Weekly Period)', y=1.02)
    plt.savefig(f'{output_dir}/stl_decomposition.png')
    plt.close()
    
    # 4. Outlier Detection Comparison (New Integrated Section)
    global_sales['stl_residual'] = res.resid

    # Method 1: IQR on STL Residuals (Robust to Seasonality)
    Q1 = global_sales['stl_residual'].quantile(0.25)
    Q3 = global_sales['stl_residual'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    global_sales['outlier_residual_iqr'] = (global_sales['stl_residual'] < lower_bound) | \
                                           (global_sales['stl_residual'] > upper_bound)

    # Method 2: Multivariate Anomaly Detection with Isolation Forest
    features = ['sales', 'onpromotion', 'dcoilwtico']
    # Defensive guardrail to ensure features exist in dataframe
    available_features = [f for f in features if f in global_sales.columns]
    
    if len(available_features) > 0:
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        global_sales['outlier_iso_forest'] = iso_forest.fit_predict(global_sales[available_features])
        global_sales['outlier_iso_forest'] = global_sales['outlier_iso_forest'] == -1

        # Plot and compare results
        plt.figure(figsize=(15, 5))
        plt.plot(global_sales.index, global_sales['sales'], label='Normal Sales', color='blue', alpha=0.6)

        # Plot IQR Residual Outliers
        outliers_iqr = global_sales[global_sales['outlier_residual_iqr']]
        plt.scatter(outliers_iqr.index, outliers_iqr['sales'], color='red', label='Residual IQR Outlier', zorder=5)

        # Plot Isolation Forest Outliers
        outliers_iso = global_sales[global_sales['outlier_iso_forest']]
        plt.scatter(outliers_iso.index, outliers_iso['sales'], color='orange', marker='x', label='Isolation Forest Outlier', zorder=5)

        plt.title('Outlier Detection Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/outlier_detection_comparison.png')
        plt.close()
    else:
        print("Skipping Isolation Forest plot: Required features not found in DataFrame.")
    
    print("Diagnostic plots saved successfully.")

# Stationarity Diagnostics (ADF Test)
def perform_adf_test(series, title=""):
    """Runs the Augmented Dickey-Fuller test for unit roots and prints results."""
    print(f"--- ADF Test Results: {title} ---")
    
    clean_series = series.dropna()
    result = adfuller(clean_series, autolag='AIC')
    
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
        
    if result[1] <= 0.05:
        print("\nConclusion: Strong evidence against the null hypothesis (p <= 0.05).")
        print("The series is likely stationary (no unit root).")
    else:
        print("\nConclusion: Weak evidence against null hypothesis (p > 0.05).")
        print("The series is likely non-stationary (unit root present).")
    print("\n")

def run_stationarity_diagnostics(train_df):
    """Wrapper to run ADF tests on raw and transformed global sales."""
    print("Running stationarity diagnostics (ADF Test)...")
    if not pd.api.types.is_datetime64_any_dtype(train_df['date']):
        train_df['date'] = pd.to_datetime(train_df['date'])
        
    agg_sales = train_df.groupby('date')[['sales', 'sales_log1p']].sum()
    perform_adf_test(agg_sales['sales'], title="Aggregate National Sales (Raw)")
    perform_adf_test(agg_sales['sales_log1p'], title="Aggregate National Sales (Log1p)")

# Log-Transformation Comparison Plot
def plot_target_transformation(train_df, output_dir='./graphs'):
    """Plots the aggregate original sales vs log1p stabilized sales."""
    print("Generating target transformation plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    if not pd.api.types.is_datetime64_any_dtype(train_df['date']):
        train_df['date'] = pd.to_datetime(train_df['date'])
        
    agg_sales = train_df.groupby('date')[['sales', 'sales_log1p']].sum()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(agg_sales.index, agg_sales['sales'], color='blue', linewidth=0.5)
    axes[0].set_title('Aggregate Daily Sales (Original)')
    axes[0].set_ylabel('Total Sales')

    axes[1].plot(agg_sales.index, agg_sales['sales_log1p'], color='green', linewidth=0.5)
    axes[1].set_title('Aggregate Daily Sales (Log1p Transformed)')
    axes[1].set_ylabel('Log(Total Sales + 1)')

    plt.tight_layout()
    out_path = f'{output_dir}/target_transformation.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved to {out_path}")

# Feature Importance Bar Chart
def plot_feature_importances(csv_path='feature_importances.csv', output_dir='./graphs'):
    """Reads the saved feature importances CSV and plots the top 20 features."""
    print("Generating feature importance plot...")
    try:
        importance_df = pd.read_csv(csv_path)
        
        # Ensure it's sorted just in case
        importance_df = importance_df.sort_values(by='importance', ascending=False)

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 8))
        
        sns.barplot(
            data=importance_df.head(20), 
            x='importance', 
            y='feature', 
            hue='feature', 
            palette='viridis', 
            legend=False
        )
        
        plt.title('Top 20 Feature Importances (XGBoost)')
        plt.xlabel('Relative Importance (Gain)')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        out_path = f'{output_dir}/feature_importances.png'
        plt.savefig(out_path)
        plt.close()
        print(f"Saved to {out_path}")
        
    except FileNotFoundError:
        print(f"File not found: {csv_path}. Make sure the training script has generated it.")


if __name__ == "__main__":
    try:
        print("Loading processed data for visualizations...")
        df = pd.read_parquet('./data/processed/train_processed.parquet')
        
        # Original Visualizations
        run_imputation_experiment(df)
        plot_diagnostics(df)
        
        # NEW: Added diagnostic and plotting executions
        run_stationarity_diagnostics(df)
        plot_target_transformation(df)
        plot_feature_importances()
        
        print("All visualizations complete.")
    except FileNotFoundError:
        print("Data file not found. Ensure './data/processed/train_processed.parquet' exists before running.")