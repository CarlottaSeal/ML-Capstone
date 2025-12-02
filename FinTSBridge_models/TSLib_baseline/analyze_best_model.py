"""
Model Selection Analysis Based on Paper Methodology
====================================================

According to the paper, model selection should be based on:

1. IC (Information Coefficient) - Correlation between predictions and actual returns
2. Rank IC - Spearman rank correlation  
3. ICIR (IC Information Ratio) = IC_mean / IC_std - Stability of IC over time

NOT Sharpe Ratio (which is backtesting result and prone to overfitting)

Key Metrics:
- Higher IC = Better predictive power
- Higher Rank IC = Better ranking ability
- Higher ICIR = More stable/consistent predictions
- IC should be positive in both first and second half (consistency check)
"""

import pandas as pd
import numpy as np
import argparse
import os


def load_rankings(csv_path: str) -> pd.DataFrame:
    """Load the rankings CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} models from {csv_path}")
    return df


def analyze_models(df: pd.DataFrame, df_path: str) -> None:
    """Perform analysis based on paper methodology: IC, Rank IC, ICIR."""
    
    print("\n" + "="*100)
    print("MODEL SELECTION ANALYSIS (Paper Methodology)")
    print("="*100)
    print("\nSelection Criteria (from paper):")
    print("  1. IC (Information Coefficient) - prediction-return correlation")
    print("  2. Rank IC - Spearman rank correlation")
    print("  3. ICIR = IC_mean / IC_std - stability of IC")
    print("  4. IC consistency across time (first half vs second half)")
    print("\n  ⚠️  Sharpe Ratio is NOT used for selection (prone to overfitting)")
    
    df = df.copy()
    
    # =========================================================================
    # Calculate ICIR proxy using ic_first_half and ic_second_half
    # ICIR = IC_mean / IC_std
    # We approximate std using the difference between halves
    # =========================================================================
    
    # IC mean (average of first and second half)
    df['ic_mean'] = (df['ic_first_half'] + df['ic_second_half']) / 2
    
    # IC std approximation (based on difference between halves)
    df['ic_std_approx'] = np.abs(df['ic_first_half'] - df['ic_second_half']) / 2
    
    # ICIR = IC_mean / IC_std (higher is better)
    # Add small epsilon to avoid division by zero
    df['icir'] = df['ic_mean'] / (df['ic_std_approx'] + 0.001)
    
    # =========================================================================
    # 1. TOP MODELS BY IC (Primary Metric)
    # =========================================================================
    print("\n" + "-"*100)
    print("1. TOP 10 BY IC (Information Coefficient)")
    print("   Primary metric: Higher IC = Better predictive power")
    print("-"*100)
    
    top_ic = df.nlargest(10, 'ic')[
        ['model_id', 'ic', 'rank_ic', 'icir', 'ic_first_half', 'ic_second_half', 'direction_accuracy']
    ].copy()
    
    top_ic['model_short'] = top_ic['model_id'].apply(
        lambda x: x.replace('long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')[:55]
    )
    
    print(f"\n{'Rank':<5} {'Model':<57} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'IC_1st':>8} {'IC_2nd':>8} {'DirAcc':>8}")
    print("-"*120)
    
    for i, (_, row) in enumerate(top_ic.iterrows(), 1):
        print(f"{i:<5} {row['model_short']:<57} {row['ic']:>8.4f} {row['rank_ic']:>8.4f} "
              f"{row['icir']:>8.2f} {row['ic_first_half']:>8.4f} {row['ic_second_half']:>8.4f} "
              f"{row['direction_accuracy']:>7.1%}")
    
    # =========================================================================
    # 2. TOP MODELS BY RANK IC
    # =========================================================================
    print("\n" + "-"*100)
    print("2. TOP 10 BY RANK IC (Spearman Rank Correlation)")
    print("   Measures ranking ability of predictions")
    print("-"*100)
    
    top_rank_ic = df.nlargest(10, 'rank_ic')[
        ['model_id', 'rank_ic', 'ic', 'icir', 'ic_first_half', 'ic_second_half', 'direction_accuracy']
    ].copy()
    
    top_rank_ic['model_short'] = top_rank_ic['model_id'].apply(
        lambda x: x.replace('long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')[:55]
    )
    
    print(f"\n{'Rank':<5} {'Model':<57} {'RankIC':>8} {'IC':>8} {'ICIR':>8} {'IC_1st':>8} {'IC_2nd':>8} {'DirAcc':>8}")
    print("-"*120)
    
    for i, (_, row) in enumerate(top_rank_ic.iterrows(), 1):
        print(f"{i:<5} {row['model_short']:<57} {row['rank_ic']:>8.4f} {row['ic']:>8.4f} "
              f"{row['icir']:>8.2f} {row['ic_first_half']:>8.4f} {row['ic_second_half']:>8.4f} "
              f"{row['direction_accuracy']:>7.1%}")
    
    # =========================================================================
    # 3. TOP MODELS BY ICIR (IC Information Ratio)
    # =========================================================================
    print("\n" + "-"*100)
    print("3. TOP 10 BY ICIR (IC Information Ratio = IC_mean / IC_std)")
    print("   Measures stability/consistency of predictions")
    print("-"*100)
    
    # Only consider models with positive IC for ICIR ranking
    positive_ic = df[df['ic'] > 0]
    
    if len(positive_ic) > 0:
        top_icir = positive_ic.nlargest(10, 'icir')[
            ['model_id', 'icir', 'ic', 'rank_ic', 'ic_first_half', 'ic_second_half', 'direction_accuracy']
        ].copy()
        
        top_icir['model_short'] = top_icir['model_id'].apply(
            lambda x: x.replace('long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')[:55]
        )
        
        print(f"\n{'Rank':<5} {'Model':<57} {'ICIR':>8} {'IC':>8} {'RankIC':>8} {'IC_1st':>8} {'IC_2nd':>8} {'DirAcc':>8}")
        print("-"*120)
        
        for i, (_, row) in enumerate(top_icir.iterrows(), 1):
            print(f"{i:<5} {row['model_short']:<57} {row['icir']:>8.2f} {row['ic']:>8.4f} "
                  f"{row['rank_ic']:>8.4f} {row['ic_first_half']:>8.4f} {row['ic_second_half']:>8.4f} "
                  f"{row['direction_accuracy']:>7.1%}")
    else:
        print("\n  No models with positive IC found")
    
    # =========================================================================
    # 4. MODELS WITH CONSISTENT IC (Both Halves Positive)
    # =========================================================================
    print("\n" + "-"*100)
    print("4. MODELS WITH CONSISTENT IC (Positive in Both Halves)")
    print("   Most reliable: Shows consistent predictive power over time")
    print("-"*100)
    
    consistent = df[(df['ic_first_half'] > 0) & (df['ic_second_half'] > 0)].nlargest(10, 'ic')[
        ['model_id', 'ic', 'rank_ic', 'icir', 'ic_first_half', 'ic_second_half', 'direction_accuracy']
    ].copy()
    
    if len(consistent) > 0:
        consistent['model_short'] = consistent['model_id'].apply(
            lambda x: x.replace('long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')[:55]
        )
        
        print(f"\n{'Rank':<5} {'Model':<57} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'IC_1st':>8} {'IC_2nd':>8} {'DirAcc':>8}")
        print("-"*120)
        
        for i, (_, row) in enumerate(consistent.iterrows(), 1):
            print(f"{i:<5} {row['model_short']:<57} {row['ic']:>8.4f} {row['rank_ic']:>8.4f} "
                  f"{row['icir']:>8.2f} {row['ic_first_half']:>8.4f} {row['ic_second_half']:>8.4f} "
                  f"{row['direction_accuracy']:>7.1%}")
    else:
        print("\n  ⚠️ No models found with positive IC in both halves!")
        print("     This suggests no model has consistent predictive power.")
    
    # =========================================================================
    # 5. SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "-"*100)
    print("5. SUMMARY STATISTICS")
    print("-"*100)
    
    print(f"""
    Total models: {len(df)}
    
    IC (Information Coefficient):
      Mean:  {df['ic'].mean():.4f}
      Max:   {df['ic'].max():.4f}
      Min:   {df['ic'].min():.4f}
      Models with IC > 0:    {(df['ic'] > 0).sum()}
      Models with IC > 0.02: {(df['ic'] > 0.02).sum()}
      Models with IC > 0.05: {(df['ic'] > 0.05).sum()}
    
    Rank IC:
      Mean:  {df['rank_ic'].mean():.4f}
      Max:   {df['rank_ic'].max():.4f}
      Models with Rank IC > 0:    {(df['rank_ic'] > 0).sum()}
      Models with Rank IC > 0.02: {(df['rank_ic'] > 0.02).sum()}
    
    IC Consistency:
      Models with positive IC in both halves: {((df['ic_first_half'] > 0) & (df['ic_second_half'] > 0)).sum()}
      Models with IC_first > 0:  {(df['ic_first_half'] > 0).sum()}
      Models with IC_second > 0: {(df['ic_second_half'] > 0).sum()}
    
    Direction Accuracy:
      Mean:  {df['direction_accuracy'].mean():.1%}
      Max:   {df['direction_accuracy'].max():.1%}
      Models > 52%: {(df['direction_accuracy'] > 0.52).sum()}
    """)
    
    # =========================================================================
    # 6. FINAL RECOMMENDATION
    # =========================================================================
    print("\n" + "="*100)
    print("FINAL RECOMMENDATION (Based on Paper Methodology)")
    print("="*100)
    
    # Best by IC
    best_ic = df.loc[df['ic'].idxmax()]
    best_ic_name = best_ic['model_id'].replace(
        'long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')
    
    # Best by Rank IC
    best_rank_ic = df.loc[df['rank_ic'].idxmax()]
    best_rank_ic_name = best_rank_ic['model_id'].replace(
        'long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')
    
    # Best consistent model (positive IC in both halves, highest IC)
    consistent_models = df[(df['ic_first_half'] > 0) & (df['ic_second_half'] > 0)]
    
    print(f"""
    BEST BY IC:
    -----------
    Model: {best_ic_name}
    - IC:           {best_ic['ic']:.4f}
    - Rank IC:      {best_ic['rank_ic']:.4f}
    - IC 1st Half:  {best_ic['ic_first_half']:.4f}
    - IC 2nd Half:  {best_ic['ic_second_half']:.4f}
    - Direction Acc: {best_ic['direction_accuracy']:.1%}
    - Sharpe (ref):  {best_ic['sharpe_ratio']:.3f}
    
    BEST BY RANK IC:
    ----------------
    Model: {best_rank_ic_name}
    - Rank IC:      {best_rank_ic['rank_ic']:.4f}
    - IC:           {best_rank_ic['ic']:.4f}
    - IC 1st Half:  {best_rank_ic['ic_first_half']:.4f}
    - IC 2nd Half:  {best_rank_ic['ic_second_half']:.4f}
    - Direction Acc: {best_rank_ic['direction_accuracy']:.1%}
    - Sharpe (ref):  {best_rank_ic['sharpe_ratio']:.3f}
    """)
    
    if len(consistent_models) > 0:
        best_consistent = consistent_models.loc[consistent_models['ic'].idxmax()]
        best_consistent_name = best_consistent['model_id'].replace(
            'long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')
        
        print(f"""
    BEST CONSISTENT MODEL (Positive IC in Both Halves) - RECOMMENDED:
    ------------------------------------------------------------------
    Model: {best_consistent_name}
    - IC:           {best_consistent['ic']:.4f}
    - Rank IC:      {best_consistent['rank_ic']:.4f}
    - IC 1st Half:  {best_consistent['ic_first_half']:.4f}  ✓ Positive
    - IC 2nd Half:  {best_consistent['ic_second_half']:.4f}  ✓ Positive
    - ICIR:         {best_consistent['icir']:.2f}
    - Direction Acc: {best_consistent['direction_accuracy']:.1%}
    - Sharpe (ref):  {best_consistent['sharpe_ratio']:.3f}
    
    ✓ This model shows consistent predictive power across time periods.
        """)
    else:
        print("""
    ⚠️  NO CONSISTENT MODEL FOUND
    -----------------------------
    No model has positive IC in both first and second half of the test period.
    This indicates that no model has reliable, consistent predictive power.
    
    Possible reasons:
    1. WTI returns may be largely unpredictable
    2. Models are overfitting to noise
    3. Features don't capture relevant information
        """)
    
    # Compare with Sharpe-based selection
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    best_sharpe_name = best_sharpe['model_id'].replace(
        'long_term_forecast_WTI-log_', '').replace('long_term_forecast_WTI_', '')
    
    print(f"""
    -------------------------------------------------------------------------
    COMPARISON: Best by Sharpe (NOT recommended for selection)
    -------------------------------------------------------------------------
    Model: {best_sharpe_name}
    - Sharpe:       {best_sharpe['sharpe_ratio']:.3f}
    - IC:           {best_sharpe['ic']:.4f}
    - Rank IC:      {best_sharpe['rank_ic']:.4f}
    - IC 1st Half:  {best_sharpe['ic_first_half']:.4f}
    - IC 2nd Half:  {best_sharpe['ic_second_half']:.4f}
    
    ⚠️  Sharpe ratio is a backtesting result and prone to overfitting!
        Use IC-based metrics for model selection.
    """)
    
    # Save enhanced rankings sorted by IC
    output_path = df_path.replace('.csv', '_by_IC.csv')
    df_sorted = df.sort_values('ic', ascending=False)
    df_sorted.to_csv(output_path, index=False)
    print(f"\n    Rankings sorted by IC saved to: {output_path}")
    
    print("\n" + "="*100)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Model Selection Analysis (Paper Methodology)')
    parser.add_argument('csv_path', type=str, nargs='?',
                        default='./WTI_trading_report_rankings.csv',
                        help='Path to the rankings CSV file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        print("\nUsage: python analyze_best_model.py <path_to_rankings.csv>")
        return
    
    df = load_rankings(args.csv_path)
    analyze_models(df, args.csv_path)


if __name__ == '__main__':
    main()