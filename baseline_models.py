import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, Theta, MSTL
import os

def run_baseline_comparison(data_path='./data/processed/features_ready.parquet', output_dir='./graphs'):
    print("Loading feature-engineered data...")
    df = pd.read_parquet(data_path)
    
    # Define Horizon and Split Data
    HORIZON = 16
    unique_dates = np.sort(df['date'].unique())
    holdout_start_date = unique_dates[-HORIZON]
    
    train_df = df[df['date'] < holdout_start_date].copy()
    holdout_df = df[df['date'] >= holdout_start_date].copy()
    
    # ---------------------------------------------------------
    # 1. Train and Predict with Final XGBoost Model
    # ---------------------------------------------------------
    print("--- 1. Training Final XGBoost Model ---")
    drop_cols = ['date', 'sales', 'sales_log1p']
    features = [col for col in df.columns if col not in drop_cols]
    
    X_train, y_train = train_df[features], train_df['sales_log1p']
    X_holdout, y_holdout_true = holdout_df[features], holdout_df['sales_log1p']
    
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # Predictions are in log1p scale
    xgb_preds_log1p = np.clip(xgb_model.predict(X_holdout), 0, None)
    
    # Calculate XGBoost RMSLE directly from log1p predictions
    xgb_rmsle = np.sqrt(mean_squared_error(y_holdout_true, xgb_preds_log1p))
    print(f"XGBoost Holdout RMSLE: {xgb_rmsle:.4f}")
    
    # Revert XGB predictions to raw sales scale for the overlay plot
    holdout_df['XGBoost'] = np.expm1(xgb_preds_log1p)

    # ---------------------------------------------------------
    # 2. Train and Predict with Statistical Baselines
    # ---------------------------------------------------------
    print("\n--- 2. Running Statistical Baselines ---")
    
    # Format data for StatsForecast
    train_df['unique_id'] = train_df['store_nbr'].astype(str) + '_' + train_df['family'].astype(str)
    holdout_df['unique_id'] = holdout_df['store_nbr'].astype(str) + '_' + holdout_df['family'].astype(str)
    
    sf_train = train_df[['unique_id', 'date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    models = [
        Naive(),
        SeasonalNaive(season_length=7),
        Theta(season_length=7),
        AutoETS(season_length=7),
        MSTL(season_length=[7], trend_forecaster=Naive())
    ]
    
    sf = StatsForecast(models=models, freq='D', n_jobs=-1)
    print("Generating baseline forecasts... (This may take a minute)")
    forecasts_df = sf.forecast(df=sf_train, h=HORIZON).reset_index()
    
    # Merge baseline predictions into the holdout dataframe
    holdout_df = holdout_df.merge(forecasts_df, left_on=['unique_id', 'date'], right_on=['unique_id', 'ds'], how='left')

    # ---------------------------------------------------------
    # 3. Evaluate All Models (RMSLE)
    # ---------------------------------------------------------
    print("\n--- 3. Evaluating All Models ---")
    baseline_models = ['Naive', 'SeasonalNaive', 'Theta', 'AutoETS', 'MSTL']
    results = {'XGBoost': xgb_rmsle}
    
    for model in baseline_models:
        # Baselines predict raw sales, so we clip to 0, then transform to log1p to calculate RMSLE
        preds_raw = np.clip(holdout_df[model], 0, None)
        preds_log1p = np.log1p(preds_raw)
        
        score = np.sqrt(mean_squared_error(y_holdout_true, preds_log1p))
        results[model] = score
        
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSLE']).sort_values(by='RMSLE')
    print("Final Model Comparison (RMSLE):")
    print(results_df)

    # ---------------------------------------------------------
    # 4. Generate Comparison Visualizations
    # ---------------------------------------------------------
    print("\n--- 4. Generating Comparison Visualizations ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualization A: Bar Chart of RMSLE Scores
    plt.figure(figsize=(10, 6))
    colors = ['#2ca02c' if m == 'XGBoost' else '#1f77b4' for m in results_df.index]
    bars = plt.bar(results_df.index, results_df['RMSLE'], color=colors)
    plt.title('Model Performance Comparison (RMSLE)')
    plt.ylabel('RMSLE (Lower is Better)')
    plt.xticks(rotation=45)
    
    # Annotate bars with scores
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_rmsle.png')
    plt.close()
    
    # Visualization B: Time Series Forecast Overlay for a Sample Series
    sample_id = holdout_df['unique_id'].iloc[0]
    sample_holdout = holdout_df[holdout_df['unique_id'] == sample_id]
    
    # Get the last 28 days of training data for context
    sample_train = train_df[(train_df['unique_id'] == sample_id) & 
                            (train_df['date'] >= holdout_start_date - pd.Timedelta(days=28))]
    
    plt.figure(figsize=(14, 7))
    
    # Plot Historical and Actuals
    plt.plot(sample_train['date'], sample_train['sales'], label='Historical Sales', color='silver')
    plt.plot(sample_holdout['date'], sample_holdout['sales'], label='Actual Sales (Holdout)', color='black', linewidth=2)
    
    # Plot Models
    plt.plot(sample_holdout['date'], sample_holdout['XGBoost'], label='XGBoost', color='green', linewidth=2.5)
    plt.plot(sample_holdout['date'], sample_holdout['AutoETS'], label='AutoETS', linestyle='--', color='blue')
    plt.plot(sample_holdout['date'], sample_holdout['Theta'], label='Theta', linestyle='-.', color='orange')
    
    plt.title(f'Forecast vs Actuals: {sample_id} (XGBoost vs Baselines)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/forecast_overlay_{sample_id}.png')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    run_baseline_comparison()