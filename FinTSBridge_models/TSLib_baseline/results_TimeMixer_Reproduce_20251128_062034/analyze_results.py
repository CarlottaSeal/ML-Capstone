"""
Analyze reproduction results to check if good performance is reproducible.
"""
import numpy as np
import pandas as pd
import os
import glob
from scipy import stats

def analyze_model_results(results_dir):
    """Analyze all model results in the directory."""
    
    print("=" * 80)
    print("REPRODUCTION EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Find all result directories
    result_dirs = glob.glob(os.path.join(results_dir, 'long_term_forecast_*'))
    
    if not result_dirs:
        print(f"No results found in {results_dir}")
        return
    
    all_results = []
    
    for result_dir in sorted(result_dirs):
        pred_path = os.path.join(result_dir, 'pred.npy')
        true_path = os.path.join(result_dir, 'true.npy')
        
        if not os.path.exists(pred_path):
            continue
            
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        dir_name = os.path.basename(result_dir)
        
        # Extract iteration number
        parts = dir_name.split('_')
        itr = parts[-1] if parts[-1].isdigit() else '0'
        
        # Determine model type
        if 'BadParams' in dir_name:
            model_type = 'Bad (dm32, df64, pl5)'
        elif 'Reproduce' in dir_name:
            model_type = 'Good (dm16, df32, pl6)'
        else:
            model_type = 'Unknown'
        
        # Calculate metrics for horizon 0
        pred_h0 = pred[:, 0, :].flatten()
        true_h0 = true[:, 0, :].flatten()
        
        valid = ~(np.isnan(pred_h0) | np.isnan(true_h0))
        pred_h0 = pred_h0[valid]
        true_h0 = true_h0[valid]
        
        # Metrics
        mse = np.mean((pred_h0 - true_h0) ** 2)
        
        # Direction accuracy
        dir_acc = np.mean(np.sign(pred_h0) == np.sign(true_h0))
        
        # IC (correlation)
        ic = np.corrcoef(pred_h0, true_h0)[0, 1] if np.std(pred_h0) > 1e-10 else 0
        
        # Simple Sharpe (direction-based strategy)
        returns = true_h0
        positions = np.sign(pred_h0)
        strategy_returns = positions * returns
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        all_results.append({
            'model_type': model_type,
            'iteration': itr,
            'n_samples': len(pred_h0),
            'mse': mse,
            'direction_accuracy': dir_acc,
            'ic': ic,
            'sharpe': sharpe,
            'dir_name': dir_name
        })
    
    if not all_results:
        print("No valid results found!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Print results by model type
    print("\n" + "-" * 80)
    print("DETAILED RESULTS BY ITERATION")
    print("-" * 80)
    
    for model_type in df['model_type'].unique():
        print(f"\n{model_type}:")
        subset = df[df['model_type'] == model_type]
        print(subset[['iteration', 'n_samples', 'direction_accuracy', 'ic', 'sharpe']].to_string(index=False))
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    summary = df.groupby('model_type').agg({
        'direction_accuracy': ['mean', 'std', 'min', 'max'],
        'ic': ['mean', 'std', 'min', 'max'],
        'sharpe': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(summary)
    
    # Statistical comparison
    print("\n" + "-" * 80)
    print("STATISTICAL TESTS")
    print("-" * 80)
    
    good_results = df[df['model_type'].str.contains('Good')]['direction_accuracy'].values
    bad_results = df[df['model_type'].str.contains('Bad')]['direction_accuracy'].values
    
    if len(good_results) >= 2:
        # Test if "good" params direction accuracy > 50%
        t_stat, p_value = stats.ttest_1samp(good_results, 0.5)
        print(f"\n'Good' params vs 50% (random):")
        print(f"  Mean: {np.mean(good_results):.4f}")
        print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05 and np.mean(good_results) > 0.5:
            print("  ✓ Significantly better than random!")
        else:
            print("  ✗ NOT significantly better than random")
    
    if len(good_results) >= 2 and len(bad_results) >= 2:
        # Compare good vs bad
        t_stat, p_value = stats.ttest_ind(good_results, bad_results)
        print(f"\n'Good' vs 'Bad' params comparison:")
        print(f"  Good mean: {np.mean(good_results):.4f}")
        print(f"  Bad mean: {np.mean(bad_results):.4f}")
        print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  ✓ Significant difference between parameter sets!")
        else:
            print("  ✗ NO significant difference")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    good_mean = np.mean(good_results) if len(good_results) > 0 else 0
    good_std = np.std(good_results) if len(good_results) > 1 else 0
    
    print(f"""
'Good' parameters (dm16, df32, pl6) results:
  Direction Accuracy: {good_mean:.1%} ± {good_std:.1%}

If results are consistent (~77% across all iterations):
  → The good performance is REPRODUCIBLE
  → Parameters matter for this specific test set
  → But still need Walk-Forward to confirm real predictive power

If results vary widely (some ~77%, some ~50%):
  → The good performance is due to RANDOM INITIALIZATION
  → Not reproducible even with same parameters

If all results are ~50%:
  → Original result may have been a FLUKE
  → Or there's some other factor we're missing
""")
    
    # Save results
    df.to_csv(os.path.join(results_dir, 'reproduction_results.csv'), index=False)
    print(f"\nResults saved to: {os.path.join(results_dir, 'reproduction_results.csv')}")

if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    analyze_model_results(results_dir)
