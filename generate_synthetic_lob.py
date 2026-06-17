"""
Synthetic LOB data generator for DIB-MASAC reproducibility.
Generates data with similar statistical properties to LOBench A-Share datasets.
"""

import numpy as np
import pandas as pd
import argparse

def generate_synthetic_lob(n_ticks=50000, n_levels=10, seed=42, output_path="synthetic_lob.csv"):
    np.random.seed(seed)

    # Mid-price: GBM with realistic A-Share parameters
    dt = 3.0          # 3-second intervals (matches LOBench)
    mu = 0.0
    sigma = 0.002
    S0 = 10.0

    prices = [S0]
    for _ in range(n_ticks - 1):
        dS = mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        prices.append(max(prices[-1] * (1 + dS), 0.01))
    prices = np.array(prices)

    # Build 10-level bid/ask from mid-price
    tick_size = 0.01
    spread_ticks = np.random.randint(1, 4, size=n_ticks)

    cols = {}
    for i in range(1, n_levels + 1):
        bid_offset = (spread_ticks + i - 1) * tick_size
        ask_offset = (spread_ticks + i - 1) * tick_size
        cols[f"BidPrice{i}"] = prices - bid_offset + np.random.randn(n_ticks) * 0.001
        cols[f"AskPrice{i}"] = prices + ask_offset + np.random.randn(n_ticks) * 0.001
        cols[f"BidVolume{i}"] = np.random.exponential(scale=500, size=n_ticks).astype(int) + 100
        cols[f"AskVolume{i}"] = np.random.exponential(scale=500, size=n_ticks).astype(int) + 100

    cols["MidPrice"] = prices

    df = pd.DataFrame(cols)

    # Normalize (same as LOBench preprocessing)
    for col in df.columns:
        if col != "MidPrice":
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    df.to_csv(output_path, index=False)
    print(f"Saved {n_ticks} ticks to {output_path}")
    print(f"Shape: {df.shape}")
    print(df.describe().iloc[:, :4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ticks", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="synthetic_lob.csv")
    args = parser.parse_args()

    generate_synthetic_lob(
        n_ticks=args.n_ticks,
        seed=args.seed,
        output_path=args.output
    )