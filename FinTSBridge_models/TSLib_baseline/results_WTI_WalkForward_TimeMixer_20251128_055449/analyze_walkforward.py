"""
Analyze walk-forward validation results.
"""
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats

def analyze_walkforward_results(output_dir):
    """
    Aggregate and analyze results from all folds.
    """
    print("=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)
    
    # Find all result directories in predictions folder
    predictions_dir = os.path.join(output_dir, 'predictions')
    result_dirs = glob.glob(os.path.join(predictions_dir, 'long_term_forecast_*'))
    
    # Also check the main output directory
    if not result_dirs:
        result_dirs = glob.glob(os.path.join(output_dir, 'long_term_forecast_*'))
    
    if not result_dirs:
        print(f"No result directories found in {predictions_dir}!")
        print("Checking for results...")
        for root, dirs, files in os.walk(output_dir):
            for d in dirs:
                if d.startswith('long_term_forecast'):
                    print(f"  Found: {os.path.join(root, d)}")
        return None
    
    print(f"Found {len(result_dirs)} result directories")
    
    all_results = []
    
    for result_dir in sorted(result_dirs):
        pred_path = os.path.join(result_dir, 'pred.npy')
        true_path = os.path.join(result_dir, 'true.npy')
        
        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            print(f"  Missing pred/true files in {result_dir}")
            continue
        
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        print(f"\nProcessing: {os.path.basename(result_dir)}")
        print(f"  Predictions shape: {pred.shape}")
        print(f"  True shape: {true.shape}")
        
        # Extract fold number from directory name
        dir_name = os.path.basename(result_dir)
        fold_num = 'unknown'
        for part in dir_name.split('_'):
            if 'fold' in part.lower():
                fold_num = part
                break
        
        # Calculate metrics for each horizon
        for h in range(pred.shape[1]):
            pred_h = pred[:, h, :].flatten()
            true_h = true[:, h, :].flatten()
            
            # Remove NaN
            valid = ~(np.isnan(pred_h) | np.isnan(true_h))
            pred_h = pred_h[valid]
            true_h = true_h[valid]
            
            if len(pred_h) < 10:
                continue
            
            # Metrics
            mse = np.mean((pred_h - true_h) ** 2)
            mae = np.mean(np.abs(pred_h - true_h))
            
            # Correlation with error handling
            if np.std(pred_h) > 1e-10 and np.std(true_h) > 1e-10:
                corr = np.corrcoef(pred_h, true_h)[0, 1]
            else:
                corr = 0
            
            # Direction accuracy
            pred_dir = np.sign(pred_h)
            true_dir = np.sign(true_h)
            nonzero = true_h != 0
            dir_acc = np.mean(pred_dir[nonzero] == true_dir[nonzero]) if nonzero.sum() > 0 else 0.5
            
            all_results.append({
                'fold': fold_num,
                'horizon': h,
                'n_samples': len(pred_h),
                'mse': mse,
                'mae': mae,
                'correlation': corr,
                'direction_accuracy': dir_acc
            })
    
    if not all_results:
        print("No valid results found!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Print results by fold
    print("\n" + "=" * 80)
    print("RESULTS BY FOLD")
    print("=" * 80)
    
    for fold in sorted(results_df['fold'].unique()):
        fold_data = results_df[results_df['fold'] == fold]
        print(f"\n{fold}:")
        print(fold_data[['horizon', 'n_samples', 'mse', 'correlation', 'direction_accuracy']].to_string(index=False))
    
    # Aggregate across folds
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (Mean ± Std across folds)")
    print("=" * 80)
    
    agg = results_df.groupby('horizon').agg({
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'correlation': ['mean', 'std'],
        'direction_accuracy': ['mean', 'std'],
        'n_samples': 'sum'
    }).round(4)
    
    print(agg)
    
    # Summary statistics for horizon 0 (most important)
    h0 = results_df[results_df['horizon'] == 0]
    
    print("\n" + "=" * 80)
    print("HORIZON 0 (Next-Day) SUMMARY")
    print("=" * 80)
    
    if len(h0) > 0:
        print(f"  Number of folds: {len(h0)}")
        print(f"  Total samples: {h0['n_samples'].sum()}")
        print(f"  Direction Accuracy: {h0['direction_accuracy'].mean():.1%} ± {h0['direction_accuracy'].std():.1%}")
        print(f"  Correlation (IC):   {h0['correlation'].mean():.4f} ± {h0['correlation'].std():.4f}")
        print(f"  MSE:                {h0['mse'].mean():.6f} ± {h0['mse'].std():.6f}")
        
        # Statistical significance test
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 80)
        
        if len(h0) >= 2:
            # Is direction accuracy significantly > 50%?
            dir_accs = h0['direction_accuracy'].values
            t_stat, p_value = stats.ttest_1samp(dir_accs, 0.5)
            print(f"\nTest 1: Direction Accuracy vs 50% (random)")
            print(f"  Mean: {np.mean(dir_accs):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT at 5% level - Model beats random!")
            else:
                print(f"  ✗ NOT significant - Cannot reject that model is random")
            
            # Is correlation significantly > 0?
            corrs = h0['correlation'].values
            t_stat, p_value = stats.ttest_1samp(corrs, 0)
            print(f"\nTest 2: Correlation vs 0 (no predictive power)")
            print(f"  Mean: {np.mean(corrs):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT at 5% level - Model has predictive power!")
            else:
                print(f"  ✗ NOT significant - Cannot reject that model has no predictive power")
        else:
            print("\nNot enough folds for statistical testing (need at least 2)")
    
    # Save results
    results_path = os.path.join(output_dir, 'walkforward_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Also save summary
    summary_path = os.path.join(output_dir, 'walkforward_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("WALK-FORWARD VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        if len(h0) > 0:
            f.write(f"Horizon 0 (Next-Day) Results:\n")
            f.write(f"  Direction Accuracy: {h0['direction_accuracy'].mean():.1%} ± {h0['direction_accuracy'].std():.1%}\n")
            f.write(f"  Correlation (IC):   {h0['correlation'].mean():.4f} ± {h0['correlation'].std():.4f}\n")
            f.write(f"  MSE:                {h0['mse'].mean():.6f} ± {h0['mse'].std():.6f}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return results_df

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    analyze_walkforward_results(output_dir)
