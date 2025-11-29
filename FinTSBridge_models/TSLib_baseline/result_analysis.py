import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 默认 horizon 映射：H+1/H+2/H+3
DEFAULT_HORIZONS = {
    1: {"true": "true_3", "pred": "pred_0"},
    2: {"true": "true_4", "pred": "pred_1"},
    3: {"true": "true_5", "pred": "pred_2"},
}


def _clip_series_for_plot(s: pd.Series, q: float = 0.01, enable_clip: bool = True) -> pd.Series:
    """
    仅用于画图：按分位数裁剪极端值，不改变原始数据。
    """
    if not enable_clip:
        return s
    lower = s.quantile(q)
    upper = s.quantile(1 - q)
    return s.clip(lower=lower, upper=upper)


def _cumulative_from_log_return(log_r: pd.Series) -> pd.Series:
    """
    log return 累计成普通收益：
        假设 log_r = log(1 + r)
        累计后：R_t = exp(sum_{i<=t} log_r_i) - 1
    """
    return np.exp(log_r.cumsum()) - 1.0


def load_result_table(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    """
    读取预测结果表，并按照日期排序。
    结果表需要包含:
        date,true_0..true_5,pred_0..pred_5
    """
    df = pd.read_csv(csv_path)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # 丢掉 true/pred 全 NaN 的行（保险）
    true_cols = [f"true_{i}" for i in range(6)]
    pred_cols = [f"pred_{i}" for i in range(6)]
    df = df.dropna(subset=true_cols + pred_cols, how="all")

    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    return df


def compute_per_horizon_metrics(
    df: pd.DataFrame,
    horizons: dict = None,
) -> pd.DataFrame:
    """
    按 horizon 计算 MSE/MAE/R^2/相关系数。
    horizons: 例如 {1: {"true": "true_3", "pred": "pred_0"}, ...}
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    metrics = []
    for h, cols in horizons.items():
        t_col = cols["true"]
        p_col = cols["pred"]

        mask = df[[t_col, p_col]].notna().all(axis=1)
        y_true = df.loc[mask, t_col].astype(float)
        y_pred = df.loc[mask, p_col].astype(float)

        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        var_y = np.var(y_true)
        r2 = 1 - np.mean((y_pred - y_true) ** 2) / (var_y + 1e-12)
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        metrics.append({
            "horizon": h,
            "true_col": t_col,
            "pred_col": p_col,
            "n_samples": len(y_true),
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Corr": corr,
        })

    metrics_df = pd.DataFrame(metrics)
    return metrics_df


def plot_returns_and_cumulative(
    df: pd.DataFrame,
    date_col: str = "date",
    horizons: dict = None,
    clip_for_plot: bool = True,
    clip_q: float = 0.01,
):
    """
    一次性画出：
      - 上排：每个 horizon 的 log-return 真实 vs 预测（裁剪极端值以便观察）
      - 下排：每个 horizon 的 cumulative return（事件时间对齐）
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=False)
    axes = axes.reshape(2, 3)

    for idx, (h, cols) in enumerate(horizons.items()):
        t_col = cols["true"]
        p_col = cols["pred"]

        y_true = df[t_col].astype(float)
        y_pred = df[p_col].astype(float)

        # ---------- 上：log-return 折线 ----------
        ax_ret = axes[0, idx]
        y_true_clip = _clip_series_for_plot(y_true, clip_q, clip_for_plot)
        y_pred_clip = _clip_series_for_plot(y_pred, clip_q, clip_for_plot)

        if date_col in df.columns:
            x = df[date_col]
            ax_ret.plot(x, y_true_clip, label=f"True H+{h}", linewidth=0.8)
            ax_ret.plot(x, y_pred_clip, label=f"Pred H+{h}", linewidth=0.8, linestyle="--")
            ax_ret.set_xlabel("Date")
        else:
            x = np.arange(len(df))
            ax_ret.plot(x, y_true_clip, label=f"True H+{h}", linewidth=0.8)
            ax_ret.plot(x, y_pred_clip, label=f"Pred H+{h}", linewidth=0.8, linestyle="--")
            ax_ret.set_xlabel("Index")

        ax_ret.set_title(f"H+{h} log return (clipped for plot)")
        ax_ret.legend(fontsize=8)
        ax_ret.grid(alpha=0.3)

        # ---------- 下：事件时间对齐的 cumulative ----------
        ax_cum = axes[1, idx]

        # shift(-h): H+1 对应下一天，H+2 对应两天后，以此类推
        shift_step = -h
        y_true_event = y_true.shift(shift_step)
        y_pred_event = y_pred.shift(shift_step)

        event_df = pd.DataFrame({"true": y_true_event, "pred": y_pred_event})
        if date_col in df.columns:
            event_df[date_col] = df[date_col]
            event_df = event_df.dropna(subset=["true", "pred"])
            x_cum = event_df[date_col]
        else:
            event_df = event_df.dropna(subset=["true", "pred"])
            x_cum = np.arange(len(event_df))

        cum_true = _cumulative_from_log_return(event_df["true"])
        cum_pred = _cumulative_from_log_return(event_df["pred"])

        ax_cum.plot(x_cum, cum_true, label=f"True H+{h}", linewidth=0.9)
        ax_cum.plot(x_cum, cum_pred, label=f"Pred H+{h}", linewidth=0.9, linestyle="--")
        ax_cum.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)

        ax_cum.set_title(f"H+{h} cumulative return (event-time aligned)")
        ax_cum.set_xlabel("Date" if date_col in df.columns else "Index")
        ax_cum.set_ylabel("Cumulative return")
        ax_cum.legend(fontsize=8)
        ax_cum.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_scatter_per_horizon(
    df: pd.DataFrame,
    horizons: dict = None,
    clip_q: float = 0.01,
):
    """
    每个 horizon 一张 scatter：真实 log-return vs 预测 log-return，
    同时画 y = x 基准线，并在标题上标注 MSE/MAE/相关系数。
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes = axes.ravel()

    for idx, (h, cols) in enumerate(horizons.items()):
        t_col = cols["true"]
        p_col = cols["pred"]

        mask = df[[t_col, p_col]].notna().all(axis=1)
        y_true = df.loc[mask, t_col].astype(float)
        y_pred = df.loc[mask, p_col].astype(float)

        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.3, s=8)

        # 基准线 y=x
        all_vals = np.concatenate([y_true.values, y_pred.values])
        lo, hi = np.quantile(all_vals, [clip_q, 1 - clip_q])
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, xs, color="red", linewidth=1, linestyle="--", label="y = x")

        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        ax.set_title(f"H+{h} scatter (log return)\nMSE={mse:.4g}, MAE={mae:.4g}, Corr={corr:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def analyze_log_return_multi_horizon(
    csv_path: str,
    date_col: str = "date",
    horizons: dict = None,
    clip_for_plot: bool = True,
    clip_q: float = 0.01,
) -> pd.DataFrame:
    """
    一键分析入口函数：
      1) 读取结果表
      2) 打印每个 horizon 的误差表
      3) 画 log-return 折线 + cumulative 曲线
      4) 画 scatter 图
    返回：per-horizon metrics DataFrame
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    df = load_result_table(csv_path, date_col=date_col)

    metrics_df = compute_per_horizon_metrics(df, horizons=horizons)
    print("\n=== Per-horizon metrics (log-return space) ===")
    print(metrics_df.to_string(index=False))

    plot_returns_and_cumulative(
        df,
        date_col=date_col,
        horizons=horizons,
        clip_for_plot=clip_for_plot,
        clip_q=clip_q,
    )

    plot_scatter_per_horizon(
        df,
        horizons=horizons,
        clip_q=clip_q,
    )

    return metrics_df


# 如果你想直接用 python 运行这个文件，也可以在这里指定默认路径
if __name__ == "__main__":
    # 示例：改成你自己的结果文件
    metrics = analyze_log_return_multi_horizon("WTI_log_PatchTST_results.csv")
