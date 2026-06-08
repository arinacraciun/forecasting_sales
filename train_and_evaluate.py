import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class PanelTimeSeriesSplit:
    """Splits panel dataset strictly by unique dates."""
    def __init__(self, n_splits=3, test_size=16):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X):
        unique_dates = np.sort(X['date'].unique())
        total_dates = len(unique_dates)
        
        for i in range(self.n_splits, 0, -1):
            val_end_idx = total_dates - ((i - 1) * self.test_size)
            val_start_idx = val_end_idx - self.test_size
            
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx - 1]
            
            train_indices = np.where(X['date'] < val_start_date)[0]
            val_indices = np.where((X['date'] >= val_start_date) & (X['date'] <= val_end_date))[0]
            
            yield train_indices, val_indices

def run_evaluation_pipeline(data_path='./data/processed/features_ready.parquet'):
    print("Loading feature-engineered data...")
    df = pd.read_parquet(data_path)
    
    # 1. Isolate the Final Holdout Set
    HORIZON = 16
    unique_dates = np.sort(df['date'].unique())
    holdout_start_date = unique_dates[-HORIZON]
    
    train_val_df = df[df['date'] < holdout_start_date].copy()
    holdout_df = df[df['date'] >= holdout_start_date].copy()
    
    print(f"Total training/validation data ends on: {train_val_df['date'].max().date()}")
    print(f"Local Holdout evaluation starts on: {holdout_df['date'].min().date()}")

    # Define Features
    drop_cols = ['date', 'sales', 'sales_log1p']
    features = [col for col in df.columns if col not in drop_cols]

    # 2. Cross-Validation on the training subset
    print("\nStarting Time-Series Cross Validation...")
    tscv = PanelTimeSeriesSplit(n_splits=3, test_size=HORIZON)
    cv_scores = []
    
    model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)

    fold = 1
    for train_idx, val_idx in tscv.split(train_val_df):
        X_train, y_train = train_val_df.iloc[train_idx][features], train_val_df.iloc[train_idx]['sales_log1p']
        X_val, y_val = train_val_df.iloc[val_idx][features], train_val_df.iloc[val_idx]['sales_log1p']
        
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_val), 0, None)
        
        fold_rmsle = np.sqrt(mean_squared_error(y_val, preds))
        cv_scores.append(fold_rmsle)
        print(f"Fold {fold} RMSLE: {fold_rmsle:.4f}")
        fold += 1

    print(f"Average CV RMSLE: {np.mean(cv_scores):.4f}")

    # 3. Final Local Evaluation
    print("\nTraining final model on all historical data and testing on sequestered holdout...")
    X_full_train = train_val_df[features]
    y_full_train = train_val_df['sales_log1p']
    
    X_holdout = holdout_df[features]
    y_holdout_true = holdout_df['sales_log1p']
    
    final_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)
    final_model.fit(X_full_train, y_full_train)

    # Save feature importances for analysis
    importances = final_model.feature_importances_
    feature_names = X_full_train.columns
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True) # Sort by importance so it's ready for easy plotting later
    feature_importance_df.to_csv("feature_importances.csv", index=False)
    print("Feature importances successfully saved to 'feature_importances.csv'!")

    holdout_preds_log1p = np.clip(final_model.predict(X_holdout), 0, None)
    
    # Calculate final metric
    final_rmsle = np.sqrt(mean_squared_error(y_holdout_true, holdout_preds_log1p))
    
    print("\n" + "="*40)
    print("FINAL LOCAL HOLDOUT PERFORMANCE")
    print(f"Model: XGBoost")
    print(f"Forecast Horizon: {HORIZON} Days")
    print(f"Holdout RMSLE: {final_rmsle:.4f}")
    print("="*40)
    
if __name__ == "__main__":
    run_evaluation_pipeline()