"""
Create walk-forward data splits for time series cross-validation.
"""
import pandas as pd
import numpy as np
import os
import sys

def create_walkforward_splits(data_path, output_dir, n_folds=4):
    """
    Create walk-forward splits ensuring no data leakage.
    
    For each fold:
    - Train: All data up to train_end
    - Validation: Data from train_end to val_end (for early stopping)
    - Test: Data from val_end to test_end (true out-of-sample)
    """
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    print(f"Total data points: {n}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Define fold boundaries
    # Reserve last 20% for final testing, split remaining into folds
    test_ratio_per_fold = 0.10  # Each test fold is ~10% of data
    val_ratio = 0.05  # Validation is ~5%
    
    splits = []
    
    for fold in range(n_folds):
        # Calculate boundaries
        # Each fold's test period starts later
        test_start_ratio = 0.6 + fold * 0.1  # 60%, 70%, 80%, 90%
        test_end_ratio = test_start_ratio + test_ratio_per_fold
        val_start_ratio = test_start_ratio - val_ratio
        
        train_end = int(n * val_start_ratio)
        val_end = int(n * test_start_ratio)
        test_end = min(int(n * test_end_ratio), n)
        
        # Create split dataframes
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[:val_end].copy()  # Include train for context
        test_df = df.iloc[:test_end].copy()  # Include all prior for context
        
        # Save splits
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # For TSLib, we save the full data but it will use the ratios
        # Actually, let's save separate files and modify data loading
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
        
        # Also save the full file for TSLib (it handles splitting internally)
        # But we'll use custom split ratios
        full_df = df.iloc[:test_end].copy()
        full_df.to_csv(os.path.join(fold_dir, f'WTI_fold{fold}.csv'), index=False)
        
        split_info = {
            'fold': fold,
            'train_end': train_end,
            'val_end': val_end,
            'test_end': test_end,
            'train_period': f"{df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[train_end-1].strftime('%Y-%m-%d')}",
            'val_period': f"{df['date'].iloc[train_end].strftime('%Y-%m-%d')} to {df['date'].iloc[val_end-1].strftime('%Y-%m-%d')}",
            'test_period': f"{df['date'].iloc[val_end].strftime('%Y-%m-%d')} to {df['date'].iloc[test_end-1].strftime('%Y-%m-%d')}",
            'train_size': train_end,
            'val_size': val_end - train_end,
            'test_size': test_end - val_end,
            'total_size': test_end
        }
        splits.append(split_info)
        
        print(f"\nFold {fold}:")
        print(f"  Train: {split_info['train_period']} ({split_info['train_size']} samples)")
        print(f"  Val:   {split_info['val_period']} ({split_info['val_size']} samples)")
        print(f"  Test:  {split_info['test_period']} ({split_info['test_size']} samples)")
    
    # Save split info
    split_df = pd.DataFrame(splits)
    split_df.to_csv(os.path.join(output_dir, 'split_info.csv'), index=False)
    
    return splits

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    n_folds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    create_walkforward_splits(data_path, output_dir, n_folds)
