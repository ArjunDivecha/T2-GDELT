"""
Step Nine — apply country weights to Portfolio_Data returns (long–short).

Terminal reporting mirrors Step Nine Calculate Portfolio Returns.py (loads, period,
verification, stats table, net/turnover breakdown, correlation / TE / IR, tails).

VERSION: 1.1  LAST UPDATED: 2026-04-06
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gdelt_track_config import PostStep5TrackConfig

PORTFOLIO_DATA = "Portfolio_Data.xlsx"


def run_portfolio_returns(cfg: PostStep5TrackConfig) -> None:
    plt.rcParams["figure.figsize"] = [12, 6]

    print("Loading data...")
    weights_df = pd.read_excel(cfg.final_country_weights_xlsx, sheet_name="All Periods", index_col=0)
    returns_df = pd.read_excel(PORTFOLIO_DATA, sheet_name="Returns", index_col=0)
    benchmark_df = pd.read_excel(PORTFOLIO_DATA, sheet_name="Benchmarks", index_col=0)

    weights_df.index = pd.to_datetime(weights_df.index).to_period("M").to_timestamp()
    returns_df.index = pd.to_datetime(returns_df.index).to_period("M").to_timestamp()
    benchmark_df.index = pd.to_datetime(benchmark_df.index).to_period("M").to_timestamp()

    weights_dates = set(weights_df.index)
    returns_dates = set(returns_df.index)
    common_dates = sorted(list(weights_dates.intersection(returns_dates)))
    if len(common_dates) > 1:
        common_dates = common_dates[:-1]

    if not common_dates:
        raise ValueError("No overlapping dates between weights and returns.")

    print(f"\nAnalysis period: {common_dates[0]} to {common_dates[-1]}")
    print(f"Number of months: {len(common_dates)}")

    print("\nCalculating portfolio returns...")
    portfolio_returns = np.zeros(len(common_dates))
    for i, date in enumerate(common_dates):
        weights = weights_df.loc[date]
        next_returns = returns_df.loc[date]
        common_countries = set(weights.index).intersection(next_returns.index)
        if len(common_countries) == 0:
            portfolio_returns[i] = np.nan
            continue
        weighted_return = 0.0
        for country in common_countries:
            w = weights[country]
            r = next_returns[country]
            if not np.isnan(w) and not np.isnan(r):
                weighted_return += w * r
        portfolio_returns[i] = weighted_return

    print("Calculating portfolio turnover...")

    def calculate_turnover(weights_df_inner, dates):
        turnover_data = []
        for i, date in enumerate(dates):
            if i == 0:
                turnover_data.append(np.nan)
                continue
            current_weights = weights_df_inner.loc[date]
            previous_weights = weights_df_inner.loc[dates[i - 1]]
            all_c = set(current_weights.index).union(set(previous_weights.index))
            turnover = 0
            for country in all_c:
                cw = current_weights.get(country, 0)
                pw = previous_weights.get(country, 0)
                if pd.isna(cw):
                    cw = 0
                if pd.isna(pw):
                    pw = 0
                turnover += abs(cw - pw)
            turnover_data.append(turnover / 2)
        return pd.Series(turnover_data, index=dates)

    turnover_series = calculate_turnover(weights_df, common_dates)
    idx = pd.DatetimeIndex(common_dates)
    results = pd.DataFrame(
        {
            "Portfolio": portfolio_returns,
            "Equal Weight": benchmark_df.loc[common_dates, "equal_weight"].to_numpy(),
            "Turnover": turnover_series.to_numpy(),
        },
        index=idx,
    )
    results["Net Return"] = results["Portfolio"] - results["Equal Weight"]
    cumulative_returns = (1 + results[["Portfolio", "Equal Weight"]]).cumprod()
    cumulative_net = (1 + results["Net Return"]).cumprod()
    cumulative_turnover = results["Turnover"].cumsum()

    def calculate_stats(returns, turnover=None):
        stats = {}
        stats["Annual Return"] = returns.mean() * 12 * 100
        stats["Annual Vol"] = returns.std() * np.sqrt(12) * 100
        stats["Sharpe Ratio"] = (returns.mean() * 12) / (returns.std() * np.sqrt(12))
        stats["Max Drawdown"] = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min() * 100
        stats["Hit Rate"] = (returns > 0).mean() * 100
        stats["Skewness"] = returns.skew()
        stats["Kurtosis"] = returns.kurtosis()
        if turnover is not None:
            stats["Avg Monthly Turnover"] = turnover.mean() * 100
            stats["Annual Turnover"] = turnover.mean() * 12 * 100
            stats["Max Monthly Turnover"] = turnover.max() * 100
            stats["Min Monthly Turnover"] = turnover.min() * 100
            stats["Turnover Volatility"] = turnover.std() * 100
        return pd.Series(stats)

    portfolio_stats = calculate_stats(results["Portfolio"], results["Turnover"])
    equal_weight_stats = calculate_stats(results["Equal Weight"])
    net_return_stats = calculate_stats(results["Net Return"])
    turnover_fields = [
        "Avg Monthly Turnover",
        "Annual Turnover",
        "Max Monthly Turnover",
        "Min Monthly Turnover",
        "Turnover Volatility",
    ]
    for field in turnover_fields:
        if field in portfolio_stats:
            equal_weight_stats[field] = 0.0
            net_return_stats[field] = 0.0
    stats = pd.DataFrame(
        {"Portfolio": portfolio_stats, "Equal Weight": equal_weight_stats, "Net Return": net_return_stats}
    )

    portfolio_mean = results["Portfolio"].mean() * 12 * 100
    benchmark_mean = results["Equal Weight"].mean() * 12 * 100
    expected_net_mean = portfolio_mean - benchmark_mean
    print("\nVerification of Net Return calculation:")
    print(f"Portfolio Annual Return: {portfolio_mean:.6f}%")
    print(f"Equal Weight Annual Return: {benchmark_mean:.6f}%")
    print(f"Expected Net Annual Return: {expected_net_mean:.6f}%")
    print(f"Calculated Net Annual Return: {stats.loc['Annual Return', 'Net Return']:.6f}%")
    print(
        f"Difference: {abs(expected_net_mean - stats.loc['Annual Return', 'Net Return']):.6f}%"
    )

    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    cumulative_returns["Portfolio"].plot(label="Portfolio", color="blue")
    cumulative_returns["Equal Weight"].plot(label="Equal Weight", color="red", alpha=0.7)
    plt.title("Cumulative Total Returns")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 2)
    cumulative_net.plot(label="Cumulative Net Return", color="green")
    plt.axhline(y=1, color="r", linestyle="--", alpha=0.3)
    plt.title("Cumulative Net Returns (Portfolio - Equal Weight)")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 3)
    (results["Turnover"] * 100).plot(label="Monthly Turnover", color="orange", alpha=0.7)
    plt.title("Monthly Portfolio Turnover")
    plt.ylabel("Turnover (%)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.final_portfolio_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

    turnover_analysis = pd.DataFrame(
        {
            "Monthly Turnover": results["Turnover"],
            "Cumulative Turnover": cumulative_turnover,
            "Annualized Turnover": results["Turnover"] * 12,
        },
        index=results.index,
    )
    print("\nSaving results...")
    out_xlsx = cfg.final_portfolio_xlsx
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        results[["Portfolio", "Equal Weight", "Net Return"]].to_excel(writer, sheet_name="Monthly Returns")
        cumulative_returns.to_excel(writer, sheet_name="Cumulative Returns")
        stats.to_excel(writer, sheet_name="Statistics")
        pd.DataFrame({"Net Return": results["Net Return"], "Cumulative Net": cumulative_net}).to_excel(
            writer, sheet_name="Net Returns"
        )
        turnover_analysis.to_excel(writer, sheet_name="Turnover Analysis")
        workbook = writer.book
        date_format = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        for sheet_name in ["Monthly Returns", "Cumulative Returns", "Net Returns", "Turnover Analysis"]:
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 0, 15, date_format)
            ws.set_column(1, 10, 12)

    print(f"\nResults saved to {out_xlsx}")
    print(f"PDF saved to {cfg.final_portfolio_pdf}")

    print("\nPortfolio Statistics:")
    print("--------------------")
    print(stats)

    net_returns = results["Net Return"]
    positive_months = (net_returns > 0).sum()
    negative_months = (net_returns < 0).sum()
    avg_positive = net_returns[net_returns > 0].mean() * 100
    avg_negative = net_returns[net_returns < 0].mean() * 100
    n_nr = len(net_returns)
    print("\nNet Return Analysis:")
    print("-------------------")
    print(f"Positive Months: {positive_months} ({positive_months / n_nr * 100:.1f}%)")
    print(f"Negative Months: {negative_months} ({negative_months / n_nr * 100:.1f}%)")
    print(f"Average Positive Return: {avg_positive:.2f}%")
    print(f"Average Negative Return: {avg_negative:.2f}%")
    if avg_negative != 0 and not np.isnan(avg_negative):
        print(f"Win/Loss Ratio: {abs(avg_positive / avg_negative):.2f}")

    turnover_stats = results["Turnover"].dropna()
    print("\nTurnover Analysis:")
    print("-----------------")
    print(f"Average Monthly Turnover: {turnover_stats.mean() * 100:.2f}%")
    print(f"Annualized Turnover: {turnover_stats.mean() * 12 * 100:.2f}%")
    print(f"Maximum Monthly Turnover: {turnover_stats.max() * 100:.2f}%")
    print(f"Minimum Monthly Turnover: {turnover_stats.min() * 100:.2f}%")
    print(f"Turnover Volatility: {turnover_stats.std() * 100:.2f}%")
    print(f"Total Cumulative Turnover: {cumulative_turnover.iloc[-1] * 100:.2f}%")

    print("\nMost Recent Net Returns:")
    print(results["Net Return"].tail())

    print("\nMost Recent Turnover:")
    print((results["Turnover"] * 100).tail())

    correlation = results[["Portfolio", "Equal Weight"]].corr().iloc[0, 1]
    print(f"\nCorrelation with Equal Weight: {correlation:.2f}")

    tracking_error = (results["Portfolio"] - results["Equal Weight"]).std() * np.sqrt(12) * 100
    print(f"Tracking Error: {tracking_error:.2f}%")

    active_return = results["Portfolio"].mean() - results["Equal Weight"].mean()
    te_dec = tracking_error / 100
    info_ratio = (active_return * 12) / te_dec if te_dec != 0 else float("nan")
    print(f"Information Ratio: {info_ratio:.2f}")

    print("\nMost Recent Returns:")
    print(results[["Portfolio", "Equal Weight", "Net Return", "Turnover"]].tail())
