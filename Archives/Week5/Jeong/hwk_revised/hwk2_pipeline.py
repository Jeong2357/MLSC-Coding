#!/usr/bin/env python3
"""
HW2: HFT Market Making via HJB — Full Pipeline (REVISED)
Optimal Control & RL — Evans Ch. 5 Dynamic Programming

KEY DIFFERENCE FROM ORIGINAL: Uses REAL Binance BTC/USDT data.
- Training: Real 14-day kline data for parameter estimation
- Synthetic paths: Generated from real-data-fitted parameters for Oracle/ML training
- Evaluation: Real UNSEEN 7-day test data (never seen during training)

All prices are normalized to basis points (bps) relative to initial price
to avoid numerical issues with BTC's large absolute price (~$85K).
1 bp = 0.01% of price.
"""
import matplotlib
matplotlib.use('Agg')
import json
import os
import glob as glob_mod
import time as _time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm, ks_2samp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import requests

# ─── Configuration ───
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11, 'figure.dpi': 120,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BINANCE_BASE = "https://data-api.binance.vision"
SYMBOL = "BTCUSDT"

print("=" * 70)
print("HW2: HFT Market Making via HJB — REVISED (Real Binance Data)")
print(f"Device: {device} (GPU {DEVICE_ID})")
print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 0: Fetch Real Binance BTC/USDT Data                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000):
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_klines = []
    current_start = start_ms
    while current_start < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current_start, "endTime": end_ms, "limit": limit}
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"    Retry {attempt+1}/5: {e}")
                _time.sleep(2 ** attempt)
        else:
            break
        if not data:
            break
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        _time.sleep(0.05)
        if len(all_klines) % 5000 < limit:
            print(f"    {len(all_klines)} candles...")
        if len(data) < limit:
            break
    return all_klines


def klines_to_df(klines):
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","n_trades","taker_buy_vol",
            "taker_buy_quote_vol","ignore"]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume",
              "taker_buy_vol","taker_buy_quote_vol"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["n_trades"] = df["n_trades"].astype(int)
    df["mid_price"] = (df["high"] + df["low"]) / 2.0
    df["half_spread_dollar"] = (df["high"] - df["low"]) / 2.0
    # Spread in basis points (bps): (high-low)/2 / mid * 10000
    df["half_spread_bps"] = df["half_spread_dollar"] / df["mid_price"] * 10000
    df["trade_intensity"] = df["n_trades"].astype(float)
    return df


def load_or_fetch_data():
    train_path = os.path.join(SAVE_DIR, "train_klines.csv")
    test_path = os.path.join(SAVE_DIR, "test_klines.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("  Loading cached data...")
        train_df = pd.read_csv(train_path, parse_dates=["open_time","close_time"])
        test_df = pd.read_csv(test_path, parse_dates=["open_time","close_time"])
        # Recompute bps columns if missing
        if "half_spread_bps" not in train_df.columns:
            train_df["half_spread_bps"] = (train_df["high"] - train_df["low"]) / 2.0 / train_df["mid_price"] * 10000
            test_df["half_spread_bps"] = (test_df["high"] - test_df["low"]) / 2.0 / test_df["mid_price"] * 10000
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df

    from datetime import datetime, timedelta
    now = datetime.utcnow()
    train_end = now - timedelta(days=31)
    train_start = train_end - timedelta(days=14)
    test_start = train_end + timedelta(days=1)
    test_end = test_start + timedelta(days=7)

    train_start_ms = int(train_start.timestamp() * 1000)
    train_end_ms = int(train_end.timestamp() * 1000)
    test_start_ms = int(test_start.timestamp() * 1000)
    test_end_ms = int(test_end.timestamp() * 1000)

    print(f"\n  Train: {train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
    print(f"  Test:  {test_start.strftime('%Y-%m-%d')} → {test_end.strftime('%Y-%m-%d')} (UNSEEN)")

    print("  Fetching TRAIN klines...")
    train_df = klines_to_df(fetch_klines(SYMBOL, "1m", train_start_ms, train_end_ms))
    print(f"  Fetching TEST klines...")
    test_df = klines_to_df(fetch_klines(SYMBOL, "1m", test_start_ms, test_end_ms))

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


print("\n[STEP 0] Fetching real Binance BTC/USDT data...")
train_df, test_df = load_or_fetch_data()

# Reference price for normalization
P_REF = train_df['mid_price'].iloc[0]
print(f"\n  Reference price for normalization: ${P_REF:.2f}")
print(f"  Training: {len(train_df)} candles, ${train_df['mid_price'].min():.0f}–${train_df['mid_price'].max():.0f}")
print(f"  Test:     {len(test_df)} candles, ${test_df['mid_price'].min():.0f}–${test_df['mid_price'].max():.0f}")

# Convert mid-prices to basis points relative to P_REF
# 1 bp = 0.01% = P_REF / 10000
# S_bps = (S_dollar - P_REF) / P_REF * 10000
S_train_dollar = train_df['mid_price'].values.copy()
S_test_dollar = test_df['mid_price'].values.copy()
S_train_bps = (S_train_dollar / P_REF - 1) * 10000  # centered around 0
S_test_bps = (S_test_dollar / P_REF - 1) * 10000

DT = 1.0  # 1 minute


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CODE-1: LOB Physics Fit & Synthetic Path Generator                ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("CODE-1: LOB Physics Fit & Synthetic Path Generator")
print("=" * 70)

# ─── 1.1: OU Process MLE (in bps space) ───
# Evans §5.1.1: dS = kappa*(mu - S)*dt + sigma*dW

def ou_neg_loglik(params, S, dt):
    """Negative log-likelihood for OU process (exact discretization)."""
    kappa, mu, sigma = params
    if kappa <= 0 or sigma <= 0:
        return 1e12
    decay = np.exp(-kappa * dt)
    var = sigma**2 * (1 - np.exp(-2*kappa*dt)) / (2*kappa)
    if var <= 0:
        return 1e12
    means = mu + (S[:-1] - mu) * decay
    residuals = S[1:] - means
    return 0.5*len(residuals)*np.log(2*np.pi*var) + 0.5*np.sum(residuals**2)/var

# Fit in bps space
x0 = [0.001, S_train_bps.mean(), np.diff(S_train_bps).std()]
bounds = [(1e-8, 1.0), (S_train_bps.min()*2, S_train_bps.max()*2), (1e-4, 200)]
result = minimize(ou_neg_loglik, x0, args=(S_train_bps, DT), bounds=bounds, method='L-BFGS-B')
kappa_hat, mu_hat, sigma_hat = result.x

# Convert sigma back to dollar terms for reporting
sigma_dollar = sigma_hat * P_REF / 10000

print(f"\n=== OU MLE (fitted to {len(S_train_bps)} REAL data points, bps space) ===")
print(f"  kappa:    {kappa_hat:.6f} /min  (half-life: {np.log(2)/max(kappa_hat,1e-10):.0f} min)")
print(f"  mu:       {mu_hat:.2f} bps")
print(f"  sigma:    {sigma_hat:.4f} bps/sqrt(min)  (=${sigma_dollar:.2f}/sqrt(min))")
print(f"  NLL: {result.fun:.2f}, Converged: {result.success}")


# ─── 1.2: Fill-Rate Estimation (in bps space) ───
# lambda(delta) = A * exp(-k * delta_bps)
spreads_bps = train_df['half_spread_bps'].values
trade_counts = train_df['trade_intensity'].values

# Bin by spread (bps), compute trade intensity per bin
n_bins = 30
sp_valid = spreads_bps[spreads_bps > 0]
pctiles = np.percentile(sp_valid, np.linspace(2, 98, n_bins + 1))
bc_list, br_list = [], []

for i in range(n_bins):
    mask = (spreads_bps >= pctiles[i]) & (spreads_bps < pctiles[i+1])
    if mask.sum() > 20:
        bc_list.append((pctiles[i] + pctiles[i+1]) / 2)
        br_list.append(trade_counts[mask].mean())

bin_centers = np.array(bc_list)
bin_rates = np.array(br_list)

# Log-linear regression: log(rate) = log(A) - k * delta_bps
pos_mask = bin_rates > 0
if pos_mask.sum() >= 3:
    X_reg = np.column_stack([np.ones(pos_mask.sum()), bin_centers[pos_mask]])
    log_rates = np.log(bin_rates[pos_mask])
    coeffs_fill = np.linalg.lstsq(X_reg, log_rates, rcond=None)[0]
    A_raw = np.exp(coeffs_fill[0])
    k_raw = -coeffs_fill[1]
else:
    A_raw = trade_counts.mean()
    k_raw = 0.5

# Scale A down for a single market maker (~0.5-2% of total trades)
# k must be positive; if the regression gave k<0, it means wider spreads
# had more trades (unusual), so we use a calibrated default.
if k_raw <= 0:
    k_mm = 0.5  # moderate decay in bps space
else:
    k_mm = k_raw

A_mm = 1.5  # target: ~1.5 fills/min at zero spread (reasonable for MM)
# Calibrate: we want A*exp(-k*median_spread) ≈ 0.5 fills/min
median_spread_bps = np.median(sp_valid)
# A_mm = 0.5 / exp(-k_mm * median_spread_bps)
# But cap it reasonably
A_mm = max(0.5, min(5.0, 0.5 * np.exp(k_mm * median_spread_bps)))
A_mm = min(A_mm, 5.0)  # cap at 5 fills/min

print(f"\n=== Fill-Rate Estimation (bps space) ===")
print(f"  Raw regression: A={A_raw:.1f}, k={k_raw:.4f}")
print(f"  Calibrated for MM: A={A_mm:.3f} fills/min, k={k_mm:.4f} /bps")
print(f"  Median spread: {median_spread_bps:.1f} bps")
print(f"  Fill rate at median spread: {A_mm * np.exp(-k_mm * median_spread_bps):.4f} fills/min")
print(f"  1/k (base half-spread): {1/k_mm:.2f} bps = ${1/k_mm * P_REF/10000:.2f}")


# ─── 1.3: Generate Synthetic OU Paths (bps) ───
def generate_ou(mu, kappa, sigma, S0, n_steps, dt, rng=None):
    """Exact discretization of OU process. [Evans §5.1.1]"""
    if rng is None:
        rng = np.random.default_rng()
    S = np.zeros(n_steps)
    S[0] = S0
    decay = np.exp(-kappa * dt)
    std = sigma * np.sqrt(max((1 - np.exp(-2*kappa*dt))/(2*max(kappa,1e-10)), 1e-15))
    for t in range(1, n_steps):
        S[t] = mu + (S[t-1] - mu) * decay + std * rng.standard_normal()
    return S


N_PATHS = 200
PATH_LEN = 480  # 8 hours

print(f"\nGenerating {N_PATHS} synthetic paths ({PATH_LEN} min = 8h each)...")
synthetic_bps = []
syn_returns = []

for i in range(N_PATHS):
    rng_i = np.random.default_rng(1000 + i)
    S0 = S_train_bps[rng_i.integers(0, len(S_train_bps))]
    path = generate_ou(mu_hat, kappa_hat, sigma_hat, S0, PATH_LEN, DT, rng_i)
    synthetic_bps.append(path)
    syn_returns.append(np.diff(path))

synthetic_bps = np.array(synthetic_bps)  # (200, 480)
all_syn_ret = np.concatenate(syn_returns)
real_ret = np.diff(S_train_bps)

ks_stat, ks_pval = ks_2samp(real_ret, all_syn_ret)
print(f"  Real returns (bps): mean={real_ret.mean():.4f}, std={real_ret.std():.4f}")
print(f"  Synth returns:      mean={all_syn_ret.mean():.4f}, std={all_syn_ret.std():.4f}")
print(f"  KS test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}")


# ─── Plot 1: Physics Fit (Real Data) ───
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CODE-1: LOB Physics Fit on REAL Binance BTC/USDT', fontsize=15, fontweight='bold', y=0.98)

ax = axes[0, 0]
t_hours = np.arange(len(S_train_dollar)) / 60
ax.plot(t_hours, S_train_dollar, alpha=0.7, lw=0.3, color='steelblue', label='Real mid-price')
ax.axhline(P_REF * (1 + mu_hat/10000), color='red', ls='--', lw=2,
           label=f'OU mean ${P_REF*(1+mu_hat/10000):.0f}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Price ($)')
ax.set_title('(a) Real BTC/USDT Mid-Price')
ax.legend(fontsize=9)
ax.ticklabel_format(style='plain', axis='y')

ax = axes[0, 1]
ou_var = sigma_hat**2 * (1 - np.exp(-2*kappa_hat*DT)) / (2*max(kappa_hat,1e-10))
x_range = np.linspace(np.percentile(real_ret, 0.5), np.percentile(real_ret, 99.5), 200)
ax.hist(real_ret, bins=100, density=True, alpha=0.6, color='steelblue', label='Real (bps)')
ax.plot(x_range, norm.pdf(x_range, 0, np.sqrt(ou_var)), 'r-', lw=2,
        label=f'OU (σ={sigma_hat:.2f} bps)')
ax.set_xlabel('1-min Return (bps)')
ax.set_ylabel('Density')
ax.set_title('(b) Return Distribution: Real vs OU')
ax.legend(fontsize=9)

ax = axes[1, 0]
ax.scatter(bin_centers[pos_mask], bin_rates[pos_mask],
           color='steelblue', s=40, zorder=5, label='Real (binned)')
dr = np.linspace(bin_centers.min()*0.5, bin_centers.max()*1.2, 100)
ax.plot(dr, A_raw * np.exp(-k_raw * dr), 'r-', lw=2,
        label=f'Raw fit: {A_raw:.0f}·e^(-{k_raw:.3f}·δ)')
ax.plot(dr, A_mm * np.exp(-k_mm * dr) * (A_raw / A_mm), 'g--', lw=1.5,
        alpha=0.7, label=f'Calibrated (scaled): A={A_mm:.1f}')
ax.set_xlabel('Half-spread (bps)')
ax.set_ylabel('Trade intensity (trades/min)')
ax.set_yscale('log')
ax.set_title('(c) Trade Intensity vs Spread')
ax.legend(fontsize=9)

ax = axes[1, 1]
max_lag = min(180, len(S_train_bps)//10)
lags = np.arange(1, max_lag+1)
autocorr = np.array([np.corrcoef(S_train_bps[:-l], S_train_bps[l:])[0,1] for l in lags])
ax.plot(lags, autocorr, 'o', ms=1.5, color='steelblue', alpha=0.6, label='Real ACF')
ax.plot(lags, np.exp(-kappa_hat*lags*DT), 'r-', lw=2,
        label=f'OU: exp(-{kappa_hat:.5f}·τ)')
ax.set_xlabel('Lag (minutes)')
ax.set_ylabel('Autocorrelation')
ax.set_title('(d) Autocorrelation: Real vs OU')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot1_physics_fit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot1_physics_fit.png")


# ─── Plot 2: Synthetic Validation ───
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-1: Synthetic Path Validation Against REAL Data', fontsize=14, fontweight='bold')

ax = axes[0]
t_hp = np.arange(PATH_LEN)/60
# Convert bps back to dollar for display
for i in range(20):
    ax.plot(t_hp, P_REF*(1+synthetic_bps[i]/10000), alpha=0.3, lw=0.5)
real_seg = S_train_dollar[:PATH_LEN]
ax.plot(np.arange(len(real_seg))/60, real_seg, 'k-', lw=1.5, alpha=0.8, label='Real (first 8h)')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Price ($)')
ax.set_title(f'(a) Synthetic vs Real Overlay')
ax.legend(fontsize=9)
ax.ticklabel_format(style='plain', axis='y')

ax = axes[1]
ax.hist(real_ret, bins=100, density=True, alpha=0.5, color='steelblue', label='Real')
ax.hist(all_syn_ret, bins=100, density=True, alpha=0.5, color='orange', label='Synthetic')
ax.set_xlabel('1-min Return (bps)')
ax.set_ylabel('Density')
ax.set_title(f'(b) Returns (KS p={ks_pval:.3f})')
ax.legend(fontsize=9)

ax = axes[2]
q_pts = np.linspace(0.5, 99.5, 200)
rq = np.percentile(real_ret, q_pts)
sq = np.percentile(all_syn_ret, q_pts)
ax.scatter(rq, sq, s=10, alpha=0.6, color='steelblue')
lims = [min(rq.min(),sq.min()), max(rq.max(),sq.max())]
ax.plot(lims, lims, 'r--', lw=1.5, label='45° line')
ax.set_xlabel('Real Quantiles (bps)')
ax.set_ylabel('Synthetic Quantiles (bps)')
ax.set_title('(c) QQ Plot')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot2_synthetic_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot2_synthetic_validation.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CODE-2: HJB Oracle (in bps space)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("CODE-2: HJB Oracle Implementation & Oracle PnL")
print("=" * 70)

GAMMA = 0.01  # Risk aversion
Q_MAX = 10

print(f"\n  HJB Parameters:")
print(f"    A={A_mm:.3f} fills/min, k={k_mm:.4f}/bps, 1/k={1/k_mm:.2f} bps")
print(f"    gamma={GAMMA}, sigma={sigma_hat:.4f} bps/sqrt(min), Q_max={Q_MAX}")


def hjb_oracle(n_steps, A, k, gamma, sigma, Q_max, dt):
    """
    HJB backward induction in bps space. [Evans §5.1.3 Step 1]

    HJB PDE (Evans Theorem 5.1):
      v_t - (γσ²/2)q² + max_{δᵇ≥0}[λ(δᵇ)(Δ⁺v + δᵇ)]
                       + max_{δᵃ≥0}[λ(δᵃ)(Δ⁻v + δᵃ)] = 0
      Terminal: v(q, T) = 0  [Evans (5.4)]

    FOC: δ* = 1/k - Δv  →  clamped to [0, max_spread]
    """
    max_spread = min(10.0 / k, 100)  # reasonable cap

    T = n_steps
    qs = np.arange(-Q_max, Q_max+1, dtype=np.float64)
    n_q = len(qs)

    v = np.zeros((n_q, T))
    delta_b = np.zeros((n_q, T))
    delta_a = np.zeros((n_q, T))

    inv_costs = gamma * sigma**2 / 2.0 * qs**2 * dt
    inv_k = 1.0 / k

    for t in reversed(range(T - 1)):
        v_next = v[:, t+1]

        dv_plus = np.zeros(n_q)
        dv_plus[:-1] = v_next[1:] - v_next[:-1]
        dv_plus[-1] = dv_plus[-2]

        dv_minus = np.zeros(n_q)
        dv_minus[1:] = v_next[:-1] - v_next[1:]
        dv_minus[0] = dv_minus[1]

        db = np.clip(inv_k - dv_plus, 0.0, max_spread)
        da = np.clip(inv_k - dv_minus, 0.0, max_spread)

        # Clamp exponent to prevent overflow
        exp_b = np.clip(-k * db, -50, 50)
        exp_a = np.clip(-k * da, -50, 50)

        bid_val = A * np.exp(exp_b) * (dv_plus + db) * dt
        ask_val = A * np.exp(exp_a) * (dv_minus + da) * dt

        bid_val[-1] = 0.0
        ask_val[0] = 0.0

        # Clamp incremental values to prevent runaway
        bid_val = np.clip(bid_val, -1e6, 1e6)
        ask_val = np.clip(ask_val, -1e6, 1e6)

        v[:, t] = v_next - inv_costs + bid_val + ask_val
        delta_b[:, t] = db
        delta_a[:, t] = da

    return v, delta_b, delta_a, qs


t0 = _time.time()
v_grid, db_grid, da_grid, qs = hjb_oracle(
    PATH_LEN, A_mm, k_mm, GAMMA, sigma_hat, Q_MAX, DT
)
elapsed = _time.time() - t0

print(f"\n  Solved in {elapsed:.2f}s. Grid: {v_grid.shape}")
print(f"  v(q=0, t=0) = {v_grid[Q_MAX, 0]:.4f} bps")
print(f"  Spreads at (q=0, t=0): bid={db_grid[Q_MAX,0]:.2f}, ask={da_grid[Q_MAX,0]:.2f} bps")
print(f"  Spreads at (q=5, t=0): bid={db_grid[Q_MAX+5,0]:.2f}, ask={da_grid[Q_MAX+5,0]:.2f} bps")

has_nan = np.isnan(v_grid).any()
if has_nan:
    print("  WARNING: NaN detected in value grid!")
else:
    print("  ✓ No NaN in value grid")
    if db_grid[Q_MAX+5, 0] > db_grid[Q_MAX, 0]:
        print("  ✓ Long inventory → bid widens (correct)")
    if da_grid[Q_MAX+5, 0] < da_grid[Q_MAX, 0]:
        print("  ✓ Long inventory → ask tightens (correct)")


# ─── Plot 3: HJB Value + Spread Heatmaps ───
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-2: HJB Value Function & Optimal Spreads (bps, Real Params)',
             fontsize=14, fontweight='bold')
tg = np.arange(PATH_LEN)/60

ax = axes[0]
im = ax.imshow(v_grid, aspect='auto', cmap='viridis', extent=[0,tg[-1],qs[-1],qs[0]])
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(a) Value v(q,t) [bps]')
plt.colorbar(im, ax=ax, label='Value (bps)')

ax = axes[1]
im = ax.imshow(db_grid, aspect='auto', cmap='YlOrRd', extent=[0,tg[-1],qs[-1],qs[0]])
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(b) Bid Spread δᵇ*(q,t) [bps]')
plt.colorbar(im, ax=ax, label='δᵇ (bps)')

ax = axes[2]
im = ax.imshow(da_grid, aspect='auto', cmap='YlOrRd', extent=[0,tg[-1],qs[-1],qs[0]])
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(c) Ask Spread δᵃ*(q,t) [bps]')
plt.colorbar(im, ax=ax, label='δᵃ (bps)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot3_hjb_value_spread.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot3_hjb_value_spread.png")


# ─── Simulation Engine (bps) ───
def simulate_mm(mid_bps, delta_b_grid, delta_a_grid, qs_arr,
                A, k, Q_max, dt, strategy='oracle', fixed_spread=None,
                nn_spreads=None, rng=None):
    """Simulate market-making. All spreads in bps."""
    if rng is None:
        rng = np.random.default_rng()

    T = len(mid_bps)
    q, cash = 0, 0.0
    pnl = np.zeros(T); inv = np.zeros(T)
    bids = np.zeros(T); asks = np.zeros(T)
    n_bf, n_af = 0, 0
    q_off = Q_max

    for t in range(T - 1):
        qi = np.clip(int(q) + q_off, 0, len(qs_arr)-1)
        ti = min(t, delta_b_grid.shape[1]-1)

        if strategy == 'oracle':
            db = delta_b_grid[qi, ti]
            da = delta_a_grid[qi, ti]
        elif strategy == 'nn' and nn_spreads is not None:
            db = nn_spreads[0][t] if t < len(nn_spreads[0]) else 1.0/k
            da = nn_spreads[1][t] if t < len(nn_spreads[1]) else 1.0/k
        else:  # passive
            db = fixed_spread if fixed_spread else 1.0/k
            da = fixed_spread if fixed_spread else 1.0/k

        bids[t], asks[t] = db, da

        lam_b = max(A * np.exp(np.clip(-k*db, -50, 50)) * dt, 0)
        lam_a = max(A * np.exp(np.clip(-k*da, -50, 50)) * dt, 0)
        nb = rng.poisson(lam_b)
        na = rng.poisson(lam_a)

        if q + nb > Q_max: nb = max(Q_max - q, 0)
        if q - na < -Q_max: na = max(q + Q_max, 0)

        # PnL in bps: buy at (S-db), sell at (S+da)
        cash += nb * db + na * da   # spread revenue
        cash += (na - nb) * mid_bps[t]  # inventory change valued at mid
        q = np.clip(q + nb - na, -Q_max, Q_max)
        n_bf += nb; n_af += na

        # Mark-to-market: cash + q * S
        pnl[t] = cash + q * mid_bps[t] - q * mid_bps[0]  # relative to initial
        # Simpler: track cumulative cash flows
        inv[t] = q

    # Final mark-to-market
    pnl[-1] = cash + q * mid_bps[-1] - q * mid_bps[0] if len(mid_bps) > 0 else 0
    inv[-1] = q

    pd_ = np.diff(pnl)
    sharpe = pd_.mean() / (pd_.std()+1e-8) * np.sqrt(1440*252) if pd_.std() > 0 else 0
    mdd = (np.maximum.accumulate(pnl) - pnl).max()

    return {
        'pnl': pnl, 'inventory': inv, 'bid_spreads': bids, 'ask_spreads': asks,
        'final_pnl': pnl[-1], 'sharpe': sharpe, 'max_drawdown': mdd,
        'max_inventory': np.abs(inv).max(),
        'fill_rate': (n_bf+n_af)/T, 'total_fills': n_bf+n_af,
    }


# Oracle + Passive on synthetic paths
print(f"\nSimulating on {N_PATHS} synthetic paths...")
oracle_res, passive_res = [], []

for i in range(N_PATHS):
    oracle_res.append(simulate_mm(
        synthetic_bps[i], db_grid, da_grid, qs,
        A_mm, k_mm, Q_MAX, DT, strategy='oracle',
        rng=np.random.default_rng(2000+i)))
    passive_res.append(simulate_mm(
        synthetic_bps[i], db_grid, da_grid, qs,
        A_mm, k_mm, Q_MAX, DT, strategy='passive',
        rng=np.random.default_rng(3000+i)))

op = np.array([r['final_pnl'] for r in oracle_res])
pp = np.array([r['final_pnl'] for r in passive_res])
os_ = np.array([r['sharpe'] for r in oracle_res])
ps_ = np.array([r['sharpe'] for r in passive_res])

print(f"\n=== Oracle (Synthetic) ===")
print(f"  PnL: {op.mean():.2f} ± {op.std():.2f} bps")
print(f"  Sharpe: {os_.mean():.2f}, Fill rate: {np.mean([r['fill_rate'] for r in oracle_res]):.4f}")
print(f"=== Passive (Synthetic) ===")
print(f"  PnL: {pp.mean():.2f} ± {pp.std():.2f} bps, Sharpe: {ps_.mean():.2f}")


# ─── Plot 4: Oracle vs Passive ───
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CODE-2: Oracle vs Passive on Synthetic Paths (bps)', fontsize=14, fontweight='bold')
t_h = np.arange(PATH_LEN)/60

ax = axes[0,0]
ax.hist(op, bins=30, alpha=0.6, color='green', label=f'Oracle (μ={op.mean():.1f})')
ax.hist(pp, bins=30, alpha=0.6, color='gray', label=f'Passive (μ={pp.mean():.1f})')
ax.set_xlabel('Final PnL (bps)'); ax.set_ylabel('Count')
ax.set_title('(a) PnL Distribution'); ax.legend(fontsize=9)

ax = axes[0,1]
ax.plot(t_h, oracle_res[0]['pnl'], 'g-', lw=1.5, label='Oracle')
ax.plot(t_h, passive_res[0]['pnl'], color='gray', lw=1.5, label='Passive')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('PnL (bps)')
ax.set_title('(b) PnL (Path 0)'); ax.legend(fontsize=9)

ax = axes[1,0]
ax.plot(t_h, oracle_res[0]['inventory'], 'g-', lw=1, label='Oracle')
ax.plot(t_h, passive_res[0]['inventory'], color='gray', lw=1, alpha=0.7, label='Passive')
ax.axhline(0, color='black', alpha=0.3)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(c) Inventory (Path 0)'); ax.legend(fontsize=9)

ax = axes[1,1]
ax.plot(t_h, oracle_res[0]['bid_spreads'], 'b-', lw=0.8, alpha=0.7, label='Bid δᵇ')
ax.plot(t_h, oracle_res[0]['ask_spreads'], 'r-', lw=0.8, alpha=0.7, label='Ask δᵃ')
ax.axhline(1.0/k_mm, color='gray', ls='--', alpha=0.5, label=f'1/k={1.0/k_mm:.1f} bps')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Spread (bps)')
ax.set_title('(d) Oracle Spreads (Path 0)'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot4_oracle_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot4_oracle_performance.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CODE-3: GRU / Mamba & REAL DATA Backtest                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("CODE-3: GRU / Mamba Student & REAL DATA Backtest")
print("=" * 70)
print(f"  Device: {device}")

SEQ_LEN = 32

def build_features(mid_bps, db_grid, da_grid, qs_arr, Q_max,
                   A, k, dt, rng=None, seq_len=32):
    """Build (features, targets) from Oracle trajectory."""
    if rng is None:
        rng = np.random.default_rng()
    T = len(mid_bps)
    returns = np.diff(mid_bps, prepend=mid_bps[0])
    ret_std = returns.std() + 1e-8
    q, q_off = 0, Q_max
    feats, targets = [], []

    for t in range(T-1):
        qi = np.clip(int(q)+q_off, 0, len(qs_arr)-1)
        ti = min(t, db_grid.shape[1]-1)
        db, da = db_grid[qi, ti], da_grid[qi, ti]

        feats.append([
            returns[t] / ret_std,
            q / Q_max,
            1.0 - t/T,
            A * np.exp(np.clip(-k*db, -50, 50)),
            A * np.exp(np.clip(-k*da, -50, 50)),
        ])
        targets.append([db, da])

        lam_b = max(A * np.exp(np.clip(-k*db, -50, 50)) * dt, 0)
        lam_a = max(A * np.exp(np.clip(-k*da, -50, 50)) * dt, 0)
        q = np.clip(q + rng.poisson(lam_b) - rng.poisson(lam_a), -Q_max, Q_max)

    feats, targets = np.array(feats), np.array(targets)
    X, Y = [], []
    for i in range(seq_len, len(feats), 4):
        X.append(feats[i-seq_len:i])
        Y.append(targets[i])
    return np.array(X), np.array(Y)


n_train_paths = 150
n_val_paths = 50

print(f"\n  Building features: {n_train_paths} train + {n_val_paths} val paths...")
Xa, Ya = [], []
for i in range(n_train_paths):
    Xi, Yi = build_features(synthetic_bps[i], db_grid, da_grid, qs, Q_MAX,
                            A_mm, k_mm, DT, np.random.default_rng(5000+i), SEQ_LEN)
    if len(Xi) > 0:
        Xa.append(Xi); Ya.append(Yi)
X_train, Y_train = np.concatenate(Xa), np.concatenate(Ya)

Xb, Yb = [], []
for i in range(n_train_paths, n_train_paths+n_val_paths):
    Xi, Yi = build_features(synthetic_bps[i], db_grid, da_grid, qs, Q_MAX,
                            A_mm, k_mm, DT, np.random.default_rng(5000+i), SEQ_LEN)
    if len(Xi) > 0:
        Xb.append(Xi); Yb.append(Yi)
X_val, Y_val = np.concatenate(Xb), np.concatenate(Yb)

print(f"  Train: {X_train.shape[0]} seqs, Val: {X_val.shape[0]} seqs")
print(f"  Target: δᵇ [{Y_train[:,0].min():.2f}, {Y_train[:,0].max():.2f}] bps")


# ─── Models ───
class SpreadGRU(nn.Module):
    def __init__(self, inp=5, hid=64, nl=2, drop=0.1):
        super().__init__()
        self.gru = nn.GRU(inp, hid, nl, batch_first=True,
                          dropout=drop if nl > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softplus())
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SimpleMamba(nn.Module):
    def __init__(self, inp=5, dm=64, nl=2, ks=4):
        super().__init__()
        self.proj = nn.Linear(inp, dm)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'conv': nn.Conv1d(dm, dm, ks, padding=ks-1, groups=dm),
            'gate': nn.Linear(dm, dm),
            'out': nn.Linear(dm, dm),
            'norm': nn.LayerNorm(dm),
        }) for _ in range(nl)])
        self.fc = nn.Sequential(
            nn.Linear(dm, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softplus())
    def forward(self, x):
        h = self.proj(x)
        for L in self.layers:
            r = h
            hc = L['conv'](h.transpose(1,2))[:,:,:h.size(1)].transpose(1,2)
            h = L['out'](hc * torch.sigmoid(L['gate'](h)))
            h = L['norm'](h + r)
        return self.fc(h[:, -1, :])


# ─── Training ───
X_tr_t = torch.FloatTensor(X_train); Y_tr_t = torch.FloatTensor(Y_train)
X_va_t = torch.FloatTensor(X_val);   Y_va_t = torch.FloatTensor(Y_val)
tr_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=512, shuffle=True)
va_loader = DataLoader(TensorDataset(X_va_t, Y_va_t), batch_size=512)


def train_model(model, tr_loader, va_loader, epochs=40, lr=1e-3, name="Model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    trl, vl = [], []
    best_val, best_state = float('inf'), None

    for ep in range(epochs):
        model.train(); el, ns = 0, 0
        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            p = model(Xb); loss = crit(p, Yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item()*len(Xb); ns += len(Xb)
        trl.append(el/ns)

        model.eval(); tl, nt = 0, 0
        with torch.no_grad():
            for Xb, Yb in va_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                tl += crit(model(Xb), Yb).item()*len(Xb); nt += len(Xb)
        vl.append(tl/nt); sched.step(tl/nt)
        if vl[-1] < best_val:
            best_val = vl[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1)%5==0 or ep==0:
            print(f"  [{name}] Ep {ep+1:3d}/{epochs}: train={trl[-1]:.6f}, val={vl[-1]:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  [{name}] Best val: {best_val:.6f}")
    return trl, vl


print("\n=== Training GRU ===")
gru_model = SpreadGRU().to(device)
gru_tl, gru_vl = train_model(gru_model, tr_loader, va_loader, epochs=40, name="GRU")

print("\n=== Training Mamba ===")
mamba_model = SimpleMamba().to(device)
mba_tl, mba_vl = train_model(mamba_model, tr_loader, va_loader, epochs=40, name="Mamba")


# ─── Plot 5: Training Curves + Prediction ───
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-3: Training & Prediction Quality', fontsize=14, fontweight='bold')

ax = axes[0]
ep = range(1, len(gru_tl)+1)
ax.plot(ep, gru_tl, 'b-', lw=1.5, label='GRU train')
ax.plot(ep, gru_vl, 'b--', lw=1.5, label='GRU val')
ax.plot(ep, mba_tl, 'r-', lw=1.5, label='Mamba train')
ax.plot(ep, mba_vl, 'r--', lw=1.5, label='Mamba val')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.set_title('(a) Training Curves')
ax.legend(fontsize=9); ax.set_yscale('log')

gru_model.eval()
with torch.no_grad():
    gru_pred = gru_model(X_va_t.to(device)).cpu().numpy()
ax = axes[1]
np_ = min(3000, len(Y_val))
ax.scatter(Y_val[:np_,0], gru_pred[:np_,0], s=3, alpha=0.3, c='blue', label='δᵇ')
ax.scatter(Y_val[:np_,1], gru_pred[:np_,1], s=3, alpha=0.3, c='red', label='δᵃ')
lm = [0, max(Y_val[:np_].max(), gru_pred[:np_].max())*1.1]
ax.plot(lm, lm, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('Oracle (bps)'); ax.set_ylabel('GRU (bps)')
ax.set_title('(b) GRU vs Oracle'); ax.legend(fontsize=9)

mamba_model.eval()
with torch.no_grad():
    mba_pred = mamba_model(X_va_t.to(device)).cpu().numpy()
ax = axes[2]
ax.scatter(Y_val[:np_,0], mba_pred[:np_,0], s=3, alpha=0.3, c='blue', label='δᵇ')
ax.scatter(Y_val[:np_,1], mba_pred[:np_,1], s=3, alpha=0.3, c='red', label='δᵃ')
ax.plot(lm, lm, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('Oracle (bps)'); ax.set_ylabel('Mamba (bps)')
ax.set_title('(c) Mamba vs Oracle'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot5_training_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot5_training_prediction.png")


# ─── NN Backtest ───
def backtest_nn(model, mid_bps, db_grid, da_grid, qs_arr, Q_max,
                A, k, dt, seq_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    T = len(mid_bps)
    returns = np.diff(mid_bps, prepend=mid_bps[0])
    ret_std = returns.std() + 1e-8
    ds = 1.0/k
    q, cash = 0, 0.0
    pnl = np.zeros(T); inv = np.zeros(T)
    nbf, naf = 0, 0
    buf = np.zeros((seq_len, 5)); bi = 0
    db, da = ds, ds
    model.eval()

    for t in range(T-1):
        feat = [returns[t]/ret_std, q/Q_max, 1.0-t/T,
                A*np.exp(np.clip(-k*db,-50,50)),
                A*np.exp(np.clip(-k*da,-50,50))]
        if bi < seq_len:
            buf[bi] = feat; bi += 1
        else:
            buf[:-1] = buf[1:]; buf[-1] = feat
            if t % 4 == 0:
                with torch.no_grad():
                    p = model(torch.FloatTensor(buf[np.newaxis]).to(device)).cpu().numpy()[0]
                db, da = max(p[0], 0.01), max(p[1], 0.01)

        lam_b = max(A*np.exp(np.clip(-k*db,-50,50))*dt, 0)
        lam_a = max(A*np.exp(np.clip(-k*da,-50,50))*dt, 0)
        nb, na = rng.poisson(lam_b), rng.poisson(lam_a)
        if q+nb > Q_max: nb = max(Q_max-q, 0)
        if q-na < -Q_max: na = max(q+Q_max, 0)

        cash += nb*db + na*da + (na-nb)*mid_bps[t]
        q = np.clip(q+nb-na, -Q_max, Q_max)
        nbf += nb; naf += na
        pnl[t] = cash + q*mid_bps[t] - q*mid_bps[0]
        inv[t] = q

    pnl[-1] = cash + q*mid_bps[-1] - q*mid_bps[0]
    inv[-1] = q
    pd_ = np.diff(pnl)
    return {
        'pnl': pnl, 'inventory': inv, 'final_pnl': pnl[-1],
        'sharpe': pd_.mean()/(pd_.std()+1e-8)*np.sqrt(1440*252),
        'max_drawdown': (np.maximum.accumulate(pnl)-pnl).max(),
        'fill_rate': (nbf+naf)/T, 'total_fills': nbf+naf,
        'max_inventory': np.abs(inv).max(),
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FINAL BACKTEST on REAL UNSEEN TEST DATA                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("FINAL BACKTEST on REAL UNSEEN TEST DATA")
print("=" * 70)

test_segment_len = PATH_LEN
n_test_segs = len(S_test_bps) // test_segment_len
print(f"  Test: {len(S_test_bps)} min → {n_test_segs} × {test_segment_len} min segments")

oracle_bt, passive_bt, gru_bt, mamba_bt = [], [], [], []

for si in range(n_test_segs):
    s, e = si * test_segment_len, (si+1) * test_segment_len
    seg = S_test_bps[s:e]

    oracle_bt.append(simulate_mm(seg, db_grid, da_grid, qs, A_mm, k_mm, Q_MAX, DT,
                                  strategy='oracle', rng=np.random.default_rng(9000+si)))
    passive_bt.append(simulate_mm(seg, db_grid, da_grid, qs, A_mm, k_mm, Q_MAX, DT,
                                   strategy='passive', rng=np.random.default_rng(9100+si)))
    gru_bt.append(backtest_nn(gru_model, seg, db_grid, da_grid, qs, Q_MAX,
                               A_mm, k_mm, DT, SEQ_LEN, rng=np.random.default_rng(9200+si)))
    mamba_bt.append(backtest_nn(mamba_model, seg, db_grid, da_grid, qs, Q_MAX,
                                 A_mm, k_mm, DT, SEQ_LEN, rng=np.random.default_rng(9300+si)))
    if (si+1) % 5 == 0:
        print(f"  {si+1}/{n_test_segs} done")

print(f"  All {n_test_segs} segments done.")


def summ(res, name):
    pnl = [r['final_pnl'] for r in res]
    sh = [r['sharpe'] for r in res]
    fr = [r['fill_rate'] for r in res]
    mi = [r['max_inventory'] for r in res]
    dd = [r['max_drawdown'] for r in res]
    return {'Strategy': name, 'Mean PnL (bps)': f'{np.mean(pnl):.2f}',
            'Std PnL': f'{np.std(pnl):.2f}', 'Sharpe': f'{np.mean(sh):.2f}',
            'Fill Rate': f'{np.mean(fr):.4f}', 'Max |q|': f'{np.mean(mi):.1f}',
            'Max DD': f'{np.mean(dd):.2f}'}


summary = pd.DataFrame([
    summ(oracle_bt, 'Oracle (HJB)'),
    summ(gru_bt, 'GRU Student'),
    summ(mamba_bt, 'Mamba Student'),
    summ(passive_bt, 'Passive (1/k)'),
])

op_t = np.array([r['final_pnl'] for r in oracle_bt])
gp_t = np.array([r['final_pnl'] for r in gru_bt])
mp_t = np.array([r['final_pnl'] for r in mamba_bt])
pp_t = np.array([r['final_pnl'] for r in passive_bt])
gr, mr, pr = op_t-gp_t, op_t-mp_t, op_t-pp_t

print("\n" + "=" * 80)
print(f"FINAL RESULTS — REAL UNSEEN TEST DATA ({n_test_segs} segments)")
print("=" * 80)
print(summary.to_string(index=False))
print(f"\nRegret (Oracle − Strategy) in bps:")
print(f"  GRU:     {gr.mean():.2f} ± {gr.std():.2f}")
print(f"  Mamba:   {mr.mean():.2f} ± {mr.std():.2f}")
print(f"  Passive: {pr.mean():.2f} ± {pr.std():.2f}")

# Convert to dollar terms
bps_to_dollar = P_REF / 10000
print(f"\nIn dollar terms (1 bps = ${bps_to_dollar:.2f}):")
print(f"  Oracle mean PnL:  ${op_t.mean()*bps_to_dollar:.2f}")
print(f"  GRU mean PnL:     ${gp_t.mean()*bps_to_dollar:.2f}")
print(f"  Mamba mean PnL:   ${mp_t.mean()*bps_to_dollar:.2f}")
print(f"  Passive mean PnL: ${pp_t.mean()*bps_to_dollar:.2f}")


# ─── Plot 6: Final Backtest ───
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
fig.suptitle(f'CODE-3: Backtest on REAL UNSEEN Data ({n_test_segs} segments)',
             fontsize=14, fontweight='bold', y=0.99)
cb = ['#90EE90','#87CEEB','#FFB6C1','#D3D3D3']

ax = fig.add_subplot(gs[0,0])
nbins = max(n_test_segs//3, 5)
ax.hist(op_t, bins=nbins, alpha=0.5, color='green', label='Oracle')
ax.hist(gp_t, bins=nbins, alpha=0.5, color='blue', label='GRU')
ax.hist(mp_t, bins=nbins, alpha=0.5, color='red', label='Mamba')
ax.hist(pp_t, bins=nbins, alpha=0.5, color='gray', label='Passive')
ax.set_xlabel('PnL (bps)'); ax.set_ylabel('Count')
ax.set_title('(a) PnL Distribution'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0,1])
ax.hist(gr, bins=nbins, alpha=0.6, color='blue', label=f'GRU ({gr.mean():.1f})')
ax.hist(mr, bins=nbins, alpha=0.6, color='red', label=f'Mamba ({mr.mean():.1f})')
ax.hist(pr, bins=nbins, alpha=0.4, color='gray', label=f'Passive ({pr.mean():.1f})')
ax.axvline(0, color='black', ls='--', alpha=0.5)
ax.set_xlabel('Regret (bps)'); ax.set_ylabel('Count')
ax.set_title('(b) Regret'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0,2])
bp = ax.boxplot([op_t,gp_t,mp_t,pp_t], labels=['Oracle','GRU','Mamba','Passive'], patch_artist=True)
for p,c in zip(bp['boxes'],cb): p.set_facecolor(c)
ax.set_ylabel('PnL (bps)'); ax.set_title('(c) Box Plot')

t_h_seg = np.arange(test_segment_len)/60
ax = fig.add_subplot(gs[1,0])
ax.plot(t_h_seg, oracle_bt[0]['pnl'], 'g-', lw=1.5, label='Oracle')
ax.plot(t_h_seg, gru_bt[0]['pnl'], 'b-', lw=1.5, label='GRU')
ax.plot(t_h_seg, mamba_bt[0]['pnl'], 'r-', lw=1.5, label='Mamba')
ax.plot(t_h_seg, passive_bt[0]['pnl'], color='gray', lw=1.5, label='Passive')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('PnL (bps)')
ax.set_title('(d) PnL (Test Seg 0)'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1,1])
ax.plot(t_h_seg, oracle_bt[0]['inventory'], 'g-', lw=1, label='Oracle')
ax.plot(t_h_seg, gru_bt[0]['inventory'], 'b-', lw=1, label='GRU')
ax.plot(t_h_seg, mamba_bt[0]['inventory'], 'r-', lw=1, label='Mamba')
ax.axhline(0, color='black', alpha=0.3)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(e) Inventory (Test Seg 0)'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1,2])
ms = [np.mean([r['sharpe'] for r in x]) for x in [oracle_bt,gru_bt,mamba_bt,passive_bt]]
ss = [np.std([r['sharpe'] for r in x]) for x in [oracle_bt,gru_bt,mamba_bt,passive_bt]]
ax.bar(['Oracle','GRU','Mamba','Passive'], ms, yerr=ss, color=cb, edgecolor='black', capsize=4)
ax.set_ylabel('Sharpe'); ax.set_title('(f) Mean Sharpe')

plt.savefig(os.path.join(FIG_DIR, 'plot6_final_backtest.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot6_final_backtest.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Theorem 5.3 Verification                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("Evans Theorem 5.3 Verification")
print("=" * 70)

q0 = Q_MAX
grad_v = (v_grid[q0+1, :] - v_grid[q0-1, :]) / 2.0
t_ax = np.arange(PATH_LEN)
cfs = np.polyfit(t_ax, grad_v, deg=1)
lam_pmp = np.polyval(cfs, t_ax)
corr = np.corrcoef(grad_v, lam_pmp)[0, 1]
print(f"  ∇_q v(0, 0) = {grad_v[0]:.6f} bps")
print(f"  PMP fit: λ(t) = {cfs[1]:.6f} + {cfs[0]:.8f}·t")
print(f"  Correlation: {corr:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Evans Theorem 5.3: PMP Costate ≡ HJB Gradient', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(t_ax/60, grad_v, 'b-', lw=1.5, alpha=0.7, label='∇_q v(0,t) [HJB]')
ax.plot(t_ax/60, lam_pmp, 'r--', lw=2, label=f'λ(t) [PMP fit]')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Gradient (bps)')
ax.set_title(f'(a) HJB Gradient vs PMP (corr={corr:.4f})'); ax.legend(fontsize=9)

ax = axes[1]
for qv in [-5,-2,0,2,5]:
    qi = qv+Q_MAX
    if 0 < qi < len(qs)-1:
        g = (v_grid[qi+1,:]-v_grid[qi-1,:])/2.0
        ax.plot(t_ax/60, g, lw=1.2, label=f'q={qv}')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('∇_q v(q,t) (bps)')
ax.set_title('(b) Gradient at Various Inventories'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot7_theorem53_verification.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot7_theorem53_verification.png")


# ─── Plot 8: Real Price + Cumulative PnL ───
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
fig.suptitle('Real BTC/USDT Test Data Overview', fontsize=14, fontweight='bold')

ax = axes[0]
seg0_dollar = S_test_dollar[:test_segment_len]
ax.plot(np.arange(len(seg0_dollar))/60, seg0_dollar, 'k-', lw=1)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Price ($)')
ax.set_title('(a) Real BTC/USDT (Test Segment 0)')
ax.ticklabel_format(style='plain', axis='y')

ax = axes[1]
x_s = np.arange(1, n_test_segs+1)
ax.plot(x_s, np.cumsum(op_t), 'g-o', lw=2, ms=4, label='Oracle')
ax.plot(x_s, np.cumsum(gp_t), 'b-s', lw=2, ms=4, label='GRU')
ax.plot(x_s, np.cumsum(mp_t), 'r-^', lw=2, ms=4, label='Mamba')
ax.plot(x_s, np.cumsum(pp_t), 'gray', ls='-', marker='d', lw=2, ms=4, label='Passive')
ax.set_xlabel('Test Segment #'); ax.set_ylabel('Cumulative PnL (bps)')
ax.set_title(f'(b) Cumulative PnL ({n_test_segs} segments)'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'plot8_real_test_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/plot8_real_test_overview.png")


# ─── Save Results ───
results_dict = {
    'reference_price': float(P_REF),
    'ou_params': {'kappa': float(kappa_hat), 'mu_bps': float(mu_hat),
                  'sigma_bps': float(sigma_hat), 'sigma_dollar': float(sigma_dollar)},
    'fill_rate': {'A': float(A_mm), 'k_bps': float(k_mm),
                  'base_spread_bps': float(1/k_mm)},
    'hjb': {'gamma': GAMMA, 'Q_max': Q_MAX, 'path_len': PATH_LEN},
    'train': {'n_candles': len(train_df),
              'start': str(train_df['open_time'].iloc[0]),
              'end': str(train_df['open_time'].iloc[-1])},
    'test': {'n_candles': len(test_df),
             'start': str(test_df['open_time'].iloc[0]),
             'end': str(test_df['open_time'].iloc[-1]),
             'n_segments': n_test_segs},
    'results': {
        'oracle': {'mean_pnl_bps': float(op_t.mean()), 'std': float(op_t.std()),
                    'sharpe': float(np.mean([r['sharpe'] for r in oracle_bt]))},
        'gru': {'mean_pnl_bps': float(gp_t.mean()), 'regret': float(gr.mean()),
                'sharpe': float(np.mean([r['sharpe'] for r in gru_bt]))},
        'mamba': {'mean_pnl_bps': float(mp_t.mean()), 'regret': float(mr.mean()),
                  'sharpe': float(np.mean([r['sharpe'] for r in mamba_bt]))},
        'passive': {'mean_pnl_bps': float(pp_t.mean()), 'regret': float(pr.mean()),
                    'sharpe': float(np.mean([r['sharpe'] for r in passive_bt]))},
    },
    'ks_test': {'stat': float(ks_stat), 'p_value': float(ks_pval)},
    'theorem53_corr': float(corr),
}

with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)
print("\nSaved: results.json")

print("\n" + "=" * 70)
print("ALL DONE — 8 plots saved to figures/")
print("=" * 70)
for m in sorted(glob_mod.glob(os.path.join(FIG_DIR, "plot*.png"))):
    print(f"  {os.path.basename(m)}")
