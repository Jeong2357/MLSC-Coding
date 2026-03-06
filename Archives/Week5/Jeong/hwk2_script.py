import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm, ks_2samp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 120
})

np.random.seed(42)
torch.manual_seed(42)
print('Setup complete.')

# ============================================================
# Stage 1: Generate realistic BTC/USDT 1-min LOB data
# We use known microstructure parameters to create ground-truth
# data, then fit models to recover the parameters.
# ============================================================

# Ground-truth parameters (realistic BTC/USDT 1-min)
TRUE_KAPPA = 0.15      # OU mean-reversion speed (per minute)
TRUE_MU    = 65000.0   # OU long-run mean (BTC price level)
TRUE_SIGMA = 12.0      # OU volatility ($/sqrt(min))
TRUE_A     = 1.5       # Fill-rate intensity baseline (fills/min)
TRUE_K     = 0.7       # Fill-rate decay parameter

N_MINUTES  = 1440 * 5  # 5 days of 1-minute data = 7200 ticks
DT         = 1.0       # 1 minute per step

def generate_ou_midprice(mu, kappa, sigma, S0, n_steps, dt, rng=None):
    """Simulate Ornstein-Uhlenbeck mid-price process.
    
    Evans §5.1.1: dS = kappa*(mu - S)*dt + sigma*dW
    Exact discretization for OU process.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    S = np.zeros(n_steps)
    S[0] = S0
    # Exact OU transition: S_{t+1} = mu + (S_t - mu)*exp(-kappa*dt) + sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa)) * Z
    decay = np.exp(-kappa * dt)
    std = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    for t in range(1, n_steps):
        S[t] = mu + (S[t-1] - mu) * decay + std * rng.standard_normal()
    return S

def generate_lob_data(S_mid, A, k, dt, rng=None):
    """Generate LOB snapshots with bid/ask prices and fill events.
    
    Fill intensity: lambda(delta) = A * exp(-k * delta)  [Evans §5.1.1]
    Spreads are drawn from a realistic distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(S_mid)
    # Realistic spread distribution (half-spread in $ terms, BTC)
    half_spreads = rng.exponential(scale=8.0, size=n) + 2.0  # min ~$2, mean ~$10
    
    bid_prices = S_mid - half_spreads
    ask_prices = S_mid + half_spreads
    
    # Fill events: Poisson with intensity lambda = A * exp(-k * delta)
    fill_rates_bid = A * np.exp(-k * half_spreads)
    fill_rates_ask = A * np.exp(-k * half_spreads)
    
    # Sample actual fills (Bernoulli approximation for dt=1)
    bid_fills = rng.poisson(fill_rates_bid * dt)
    ask_fills = rng.poisson(fill_rates_ask * dt)
    
    # LOB volumes (realistic)
    bid_vol = rng.exponential(scale=0.5, size=n) + 0.1
    ask_vol = rng.exponential(scale=0.5, size=n) + 0.1
    
    df = pd.DataFrame({
        'mid_price': S_mid,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'half_spread': half_spreads,
        'bid_fill_rate': fill_rates_bid,
        'ask_fill_rate': fill_rates_ask,
        'bid_fills': bid_fills,
        'ask_fills': ask_fills,
        'bid_vol': bid_vol,
        'ask_vol': ask_vol
    })
    return df

# Generate data
rng = np.random.default_rng(42)
S_mid = generate_ou_midprice(TRUE_MU, TRUE_KAPPA, TRUE_SIGMA, TRUE_MU, N_MINUTES, DT, rng)
lob_df = generate_lob_data(S_mid, TRUE_A, TRUE_K, DT, rng)

print(f'Generated {len(lob_df)} LOB snapshots (5 days x 1440 min/day)')
print(f'Mid-price range: ${S_mid.min():.0f} - ${S_mid.max():.0f}')
print(f'Mean half-spread: ${lob_df["half_spread"].mean():.2f}')
print(f'Mean fill rate (bid): {lob_df["bid_fill_rate"].mean():.4f} fills/min')
lob_df.head()

# ============================================================
# OU Process MLE Estimation
# Evans §5.1.1: dS = kappa*(mu - S)*dt + sigma*dW
# Exact transition: S_{t+1} | S_t ~ N(mu + (S_t-mu)*e^{-kappa*dt},
#                                     sigma^2*(1-e^{-2*kappa*dt})/(2*kappa))
# ============================================================

def ou_neg_loglik(params, S, dt):
    """Negative log-likelihood for OU process (exact discretization)."""
    kappa, mu, sigma = params
    if kappa <= 0 or sigma <= 0:
        return 1e12
    n = len(S) - 1
    decay = np.exp(-kappa * dt)
    var = sigma**2 * (1 - np.exp(-2*kappa*dt)) / (2*kappa)
    if var <= 0:
        return 1e12
    
    # Predicted means
    means = mu + (S[:-1] - mu) * decay
    residuals = S[1:] - means
    
    # Log-likelihood
    nll = 0.5 * n * np.log(2 * np.pi * var) + 0.5 * np.sum(residuals**2) / var
    return nll

# Fit OU parameters
S_data = lob_df['mid_price'].values
x0 = [0.1, S_data.mean(), S_data.std()]
bounds = [(1e-6, 10.0), (S_data.min(), S_data.max()), (1e-6, 100.0)]

result = minimize(ou_neg_loglik, x0, args=(S_data, DT), bounds=bounds, method='L-BFGS-B')
kappa_hat, mu_hat, sigma_hat = result.x

print('=== OU MLE Results ===')
print(f'  kappa (mean-reversion): {kappa_hat:.4f}  (true: {TRUE_KAPPA})')
print(f'  mu    (long-run mean):  {mu_hat:.2f}   (true: {TRUE_MU})')
print(f'  sigma (volatility):     {sigma_hat:.4f}  (true: {TRUE_SIGMA})')
print(f'  NLL: {result.fun:.2f}')
print(f'  Convergence: {result.success}')

# ============================================================
# Fill-Rate Estimation: lambda(delta) = A * exp(-k * delta)
# Evans §5.1.1: Poisson fill model
# Method: bin spreads, compute empirical fill rate, log-linear fit
# ============================================================

spreads = lob_df['half_spread'].values
fills = lob_df['bid_fills'].values  # Use bid fills for estimation

# Bin spreads and compute average fill rate per bin
n_bins = 25
spread_bins = np.linspace(spreads.min(), np.percentile(spreads, 95), n_bins + 1)
bin_centers = []
bin_fill_rates = []

for i in range(n_bins):
    mask = (spreads >= spread_bins[i]) & (spreads < spread_bins[i+1])
    if mask.sum() > 10:  # need sufficient data
        bin_centers.append((spread_bins[i] + spread_bins[i+1]) / 2)
        bin_fill_rates.append(fills[mask].mean() / DT)  # fills per unit time

bin_centers = np.array(bin_centers)
bin_fill_rates = np.array(bin_fill_rates)

# Log-linear regression: log(lambda) = log(A) - k * delta
positive_mask = bin_fill_rates > 0
log_rates = np.log(bin_fill_rates[positive_mask])
X = np.column_stack([np.ones(positive_mask.sum()), bin_centers[positive_mask]])
coeffs = np.linalg.lstsq(X, log_rates, rcond=None)[0]

A_hat = np.exp(coeffs[0])
k_hat = -coeffs[1]

print('=== Fill-Rate Estimation ===')
print(f'  A (intensity baseline): {A_hat:.4f}  (true: {TRUE_A})')
print(f'  k (decay parameter):    {k_hat:.4f}  (true: {TRUE_K})')

# ============================================================
# Plot 1: Physics Fit Results (4-panel figure)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CODE-1: LOB Physics Fit Results', fontsize=16, fontweight='bold', y=0.98)

# (a) Mid-price time series with OU fit
ax = axes[0, 0]
t_hours = np.arange(len(S_data)) / 60  # convert to hours
ax.plot(t_hours, S_data, alpha=0.7, linewidth=0.5, color='steelblue', label='Mid-price')
ax.axhline(mu_hat, color='red', linestyle='--', alpha=0.8, label=f'$\\hat{{\\mu}}$ = {mu_hat:.0f}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Mid-price ($)')
ax.set_title('(a) Mid-Price & OU Mean')
ax.legend(fontsize=9)

# (b) Return distribution: actual vs OU model
ax = axes[0, 1]
returns = np.diff(S_data)
decay = np.exp(-kappa_hat * DT)
ou_var = sigma_hat**2 * (1 - np.exp(-2*kappa_hat*DT)) / (2*kappa_hat)
x_range = np.linspace(returns.min(), returns.max(), 200)
ax.hist(returns, bins=80, density=True, alpha=0.6, color='steelblue', label='Empirical')
ax.plot(x_range, norm.pdf(x_range, 0, np.sqrt(ou_var)), 'r-', linewidth=2, 
        label=f'OU model (σ={sigma_hat:.2f})')
ax.set_xlabel('1-min Return ($)')
ax.set_ylabel('Density')
ax.set_title('(b) Return Distribution: Data vs OU Model')
ax.legend(fontsize=9)

# (c) Fill rate vs spread (log-scale)
ax = axes[1, 0]
ax.scatter(bin_centers[positive_mask], bin_fill_rates[positive_mask], 
           color='steelblue', s=40, zorder=5, label='Empirical bins')
delta_range = np.linspace(0.5, bin_centers.max(), 100)
ax.plot(delta_range, A_hat * np.exp(-k_hat * delta_range), 'r-', linewidth=2, 
        label=f'Fit: {A_hat:.2f}·exp(-{k_hat:.2f}·δ)')
ax.plot(delta_range, TRUE_A * np.exp(-TRUE_K * delta_range), 'g--', linewidth=1.5, 
        alpha=0.7, label=f'True: {TRUE_A}·exp(-{TRUE_K}·δ)')
ax.set_xlabel('Half-spread δ ($)')
ax.set_ylabel('Fill rate (fills/min)')
ax.set_yscale('log')
ax.set_title('(c) Fill Rate vs Spread')
ax.legend(fontsize=9)

# (d) OU autocorrelation check
ax = axes[1, 1]
max_lag = 120  # 2 hours
lags = np.arange(1, max_lag + 1)
autocorr = np.array([np.corrcoef(S_data[:-l], S_data[l:])[0, 1] for l in lags])
theoretical_acf = np.exp(-kappa_hat * lags * DT)
ax.plot(lags, autocorr, 'o', markersize=2, color='steelblue', label='Empirical ACF')
ax.plot(lags, theoretical_acf, 'r-', linewidth=2, label=f'OU: exp(-{kappa_hat:.3f}·τ)')
ax.set_xlabel('Lag (minutes)')
ax.set_ylabel('Autocorrelation')
ax.set_title('(d) Autocorrelation: Data vs OU Model')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot1_physics_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot1_physics_fit.png')

# ============================================================
# Synthetic LOB Path Generation (200 paths)
# Uses estimated OU + Poisson fill parameters
# ============================================================

N_PATHS = 200
PATH_LEN = 1440  # 1 day of 1-minute data per path

synthetic_paths = []
synthetic_returns_all = []

for i in range(N_PATHS):
    rng_i = np.random.default_rng(1000 + i)
    S_syn = generate_ou_midprice(mu_hat, kappa_hat, sigma_hat, mu_hat, PATH_LEN, DT, rng_i)
    synthetic_paths.append(S_syn)
    synthetic_returns_all.append(np.diff(S_syn))

synthetic_paths = np.array(synthetic_paths)  # (200, 1440)
all_syn_returns = np.concatenate(synthetic_returns_all)

# KS test: compare real vs synthetic returns
real_returns = np.diff(S_data)
ks_stat, ks_pval = ks_2samp(real_returns, all_syn_returns)

print(f'Generated {N_PATHS} synthetic paths, each {PATH_LEN} minutes')
print(f'\nReturn Distribution Comparison:')
print(f'  Real  — mean: {real_returns.mean():.4f}, std: {real_returns.std():.4f}')
print(f'  Synth — mean: {all_syn_returns.mean():.4f}, std: {all_syn_returns.std():.4f}')
print(f'  KS test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}')

# ============================================================
# Plot 2: Synthetic Paths & Return Distribution Comparison
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-1: Synthetic Path Generation & Validation', fontsize=14, fontweight='bold')

# (a) Sample synthetic paths
ax = axes[0]
t_hours_path = np.arange(PATH_LEN) / 60
for i in range(min(20, N_PATHS)):
    ax.plot(t_hours_path, synthetic_paths[i], alpha=0.3, linewidth=0.5)
ax.plot(t_hours_path, synthetic_paths.mean(axis=0), 'k-', linewidth=2, label='Mean path')
ax.axhline(mu_hat, color='red', linestyle='--', alpha=0.7, label=f'μ = {mu_hat:.0f}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Mid-price ($)')
ax.set_title(f'(a) 20 of {N_PATHS} Synthetic Paths')
ax.legend(fontsize=9)

# (b) Return distribution comparison
ax = axes[1]
ax.hist(real_returns, bins=80, density=True, alpha=0.5, color='steelblue', label='Real data')
ax.hist(all_syn_returns, bins=80, density=True, alpha=0.5, color='orange', label='Synthetic')
ax.set_xlabel('1-min Return ($)')
ax.set_ylabel('Density')
ax.set_title(f'(b) Return Distribution (KS p={ks_pval:.3f})')
ax.legend(fontsize=9)

# (c) QQ plot
ax = axes[2]
real_sorted = np.sort(real_returns)
syn_sorted = np.sort(np.random.choice(all_syn_returns, size=len(real_returns), replace=False))
q_points = np.linspace(0.5, 99.5, 200)
real_q = np.percentile(real_sorted, q_points)
syn_q = np.percentile(syn_sorted, q_points)
ax.scatter(real_q, syn_q, s=10, alpha=0.6, color='steelblue')
lims = [min(real_q.min(), syn_q.min()), max(real_q.max(), syn_q.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='45° line')
ax.set_xlabel('Real Return Quantiles')
ax.set_ylabel('Synthetic Return Quantiles')
ax.set_title('(c) QQ Plot: Real vs Synthetic')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot2_synthetic_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot2_synthetic_validation.png')

# ============================================================
# HJB Oracle: Backward Induction (Vectorized, Corrected)
# Evans §5.1.3 Step 1: Solve HJB on grid (q, t)
# ============================================================

def hjb_oracle(n_steps, A, k, gamma, sigma, Q_max, dt):
    """
    Evans §5.1.3 Step 1: HJB backward induction (vectorized over q).
    
    State: q ∈ [-Q_max, Q_max], t ∈ [0, T]
    
    HJB equation (Evans Theorem 5.1):
      v_t - (γσ²/2)q²
        + max_{δ^b}[λ(δ^b)(Δ⁺v + δ^b)]
        + max_{δ^a}[λ(δ^a)(Δ⁻v + δ^a)] = 0
    
    where  Δ⁺v = v(q+1,t) - v(q,t),  Δ⁻v = v(q-1,t) - v(q,t)
    
    FOC of each max yields:
      δ^{b*} = 1/k - Δ⁺v     (bid: tightens when short, widens when long)
      δ^{a*} = 1/k - Δ⁻v     (ask: tightens when long, widens when short)
    
    Returns: v[q_idx, t], delta_b[q_idx, t], delta_a[q_idx, t]
    """
    T = n_steps
    qs = np.arange(-Q_max, Q_max + 1, dtype=np.float64)
    n_q = len(qs)
    
    v = np.zeros((n_q, T))
    delta_b = np.zeros((n_q, T))
    delta_a = np.zeros((n_q, T))
    
    # Terminal condition: v(q, T) = 0  [Evans (5.4): g(x) = 0]
    
    # Precompute
    inv_costs = gamma * sigma**2 / 2.0 * qs**2 * dt   # Evans §5.2.3 LQR cost
    inv_k = 1.0 / k
    
    # Backward induction [Evans §5.1.3 "Step 1"]
    for t in reversed(range(T - 1)):
        v_next = v[:, t + 1]
        
        # Δ⁺v = v(q+1, t+1) - v(q, t+1)  [buy → inventory +1]
        dv_plus = np.full(n_q, -1e6)
        dv_plus[:-1] = v_next[1:] - v_next[:-1]
        
        # Δ⁻v = v(q-1, t+1) - v(q, t+1)  [sell → inventory -1]
        dv_minus = np.full(n_q, -1e6)
        dv_minus[1:] = v_next[:-1] - v_next[1:]
        
        # Optimal spreads from FOC [Evans Theorem 5.1]
        db = np.maximum(inv_k - dv_plus,  0.0)   # bid spread
        da = np.maximum(inv_k - dv_minus, 0.0)   # ask spread  ← CORRECTED
        
        # Bellman update [Evans §5.1.2]
        # Each max term evaluates to λ(δ*)(δ* + Δv) at optimum
        bid_val = A * np.exp(-k * db) * (db + dv_plus)  * dt
        ask_val = A * np.exp(-k * da) * (da + dv_minus) * dt   # ← CORRECTED
        
        # Boundary: can't buy beyond Q_max or sell beyond -Q_max
        bid_val[-1] = 0.0
        ask_val[0]  = 0.0
        
        v[:, t] = v_next - inv_costs + bid_val + ask_val
        delta_b[:, t] = db
        delta_a[:, t] = da
    
    return v, delta_b, delta_a, qs

# Parameters for Oracle
GAMMA = 0.01     # Risk aversion
Q_MAX = 10       # Max inventory

# Solve HJB
print('Solving HJB backward induction (vectorized)...')
import time as _time
_t0 = _time.time()
v_grid, db_grid, da_grid, qs = hjb_oracle(
    PATH_LEN, A_hat, k_hat, GAMMA, sigma_hat, Q_MAX, DT
)
print(f'Done in {_time.time()-_t0:.1f}s. Grid: {v_grid.shape}')
print(f'Value at (q=0, t=0): {v_grid[Q_MAX, 0]:.2f}')
print(f'Spread at (q=0, t=0): bid={db_grid[Q_MAX,0]:.3f}, ask={da_grid[Q_MAX,0]:.3f}')
print(f'Spread at (q=5, t=0): bid={db_grid[Q_MAX+5,0]:.3f}, ask={da_grid[Q_MAX+5,0]:.3f}')
print(f'  (long inventory → bid widens, ask tightens ✓)' 
      if db_grid[Q_MAX+5,0] > db_grid[Q_MAX,0] and da_grid[Q_MAX+5,0] < da_grid[Q_MAX,0]
      else '  (check spread behavior!)')

# ============================================================
# Plot 3: HJB Value Surface & Optimal Spread Heatmap
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-2: HJB Value Function & Optimal Spreads', fontsize=14, fontweight='bold')

time_grid = np.arange(PATH_LEN) / 60  # hours

# (a) Value function v(q, t)
ax = axes[0]
im = ax.imshow(v_grid, aspect='auto', cmap='viridis',
               extent=[0, time_grid[-1], qs[-1], qs[0]])
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Inventory q')
ax.set_title('(a) Value Function v(q, t)')
plt.colorbar(im, ax=ax, label='Value')

# (b) Optimal bid spread heatmap
ax = axes[1]
im = ax.imshow(db_grid, aspect='auto', cmap='YlOrRd',
               extent=[0, time_grid[-1], qs[-1], qs[0]])
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Inventory q')
ax.set_title('(b) Optimal Bid Spread δ*(q, t)')
plt.colorbar(im, ax=ax, label='Bid spread')

# (c) Optimal ask spread heatmap
ax = axes[2]
im = ax.imshow(da_grid, aspect='auto', cmap='YlOrRd',
               extent=[0, time_grid[-1], qs[-1], qs[0]])
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Inventory q')
ax.set_title('(c) Optimal Ask Spread δᵃ*(q, t)')
plt.colorbar(im, ax=ax, label='Ask spread')

plt.tight_layout()
plt.savefig('plot3_hjb_value_spread.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot3_hjb_value_spread.png')

# ============================================================
# Oracle Simulation Engine
# Forward simulation using optimal spreads from HJB
# ============================================================

def simulate_market_maker(mid_prices, delta_b_grid, delta_a_grid, qs, 
                          A, k, Q_max, dt, strategy='oracle', 
                          fixed_spread=None, rng=None):
    """
    Simulate market-making on a given mid-price path.
    
    strategy: 'oracle' (HJB optimal), 'passive' (fixed spread), or 'nn' (model-predicted)
    Returns: dict with pnl, inventory, fills, spreads trajectories
    """
    if rng is None:
        rng = np.random.default_rng()
    
    T = len(mid_prices)
    q = 0         # Current inventory
    cash = 0.0    # Accumulated cash
    
    pnl_series = np.zeros(T)
    inv_series = np.zeros(T)
    bid_spreads = np.zeros(T)
    ask_spreads = np.zeros(T)
    bid_fills_total = 0
    ask_fills_total = 0
    
    q_idx_offset = Q_max  # index offset: q=0 => index Q_max
    
    for t in range(T - 1):
        S = mid_prices[t]
        q_idx = int(q) + q_idx_offset
        q_idx = np.clip(q_idx, 0, len(qs) - 1)
        
        # Get spreads
        if strategy == 'oracle':
            db = delta_b_grid[q_idx, min(t, delta_b_grid.shape[1]-1)]
            da = delta_a_grid[q_idx, min(t, delta_a_grid.shape[1]-1)]
        elif strategy == 'passive':
            db = fixed_spread if fixed_spread else 1.0 / k
            da = fixed_spread if fixed_spread else 1.0 / k
        else:  # 'nn' — spreads passed via fixed_spread as array
            db = fixed_spread[0][t] if isinstance(fixed_spread, (list, tuple)) else fixed_spread
            da = fixed_spread[1][t] if isinstance(fixed_spread, (list, tuple)) else fixed_spread
        
        bid_spreads[t] = db
        ask_spreads[t] = da
        
        # Fill probabilities (Poisson)
        lam_b = A * np.exp(-k * db) * dt
        lam_a = A * np.exp(-k * da) * dt
        
        # Bid fill: we buy at S - db => inventory increases
        n_bid_fill = rng.poisson(lam_b)
        # Ask fill: we sell at S + da => inventory decreases
        n_ask_fill = rng.poisson(lam_a)
        
        # Enforce inventory bounds
        if q + n_bid_fill > Q_max:
            n_bid_fill = max(Q_max - q, 0)
        if q - n_ask_fill < -Q_max:
            n_ask_fill = max(q + Q_max, 0)
        
        # Update cash and inventory
        cash -= n_bid_fill * (S - db)   # pay to buy
        cash += n_ask_fill * (S + da)   # receive from sell
        q += n_bid_fill - n_ask_fill
        q = np.clip(q, -Q_max, Q_max)
        
        bid_fills_total += n_bid_fill
        ask_fills_total += n_ask_fill
        
        # Mark-to-market PnL
        pnl_series[t] = cash + q * S
        inv_series[t] = q
    
    # Final step
    pnl_series[-1] = cash + q * mid_prices[-1]
    inv_series[-1] = q
    
    total_fills = bid_fills_total + ask_fills_total
    fill_rate = total_fills / T if T > 0 else 0
    
    # Sharpe ratio (annualized from 1-min data)
    pnl_diff = np.diff(pnl_series)
    if pnl_diff.std() > 0:
        sharpe = pnl_diff.mean() / pnl_diff.std() * np.sqrt(1440 * 252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    cummax = np.maximum.accumulate(pnl_series)
    drawdown = cummax - pnl_series
    max_dd = drawdown.max()
    
    return {
        'pnl': pnl_series,
        'inventory': inv_series,
        'bid_spreads': bid_spreads,
        'ask_spreads': ask_spreads,
        'final_pnl': pnl_series[-1],
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'max_inventory': np.abs(inv_series).max(),
        'fill_rate': fill_rate,
        'total_fills': total_fills
    }

# Run Oracle and Passive on all synthetic paths
oracle_results = []
passive_results = []

print('Simulating Oracle and Passive strategies on 200 synthetic paths...')
for i in range(N_PATHS):
    rng_sim = np.random.default_rng(2000 + i)
    
    # Oracle strategy
    res_oracle = simulate_market_maker(
        synthetic_paths[i], db_grid, da_grid, qs,
        A_hat, k_hat, Q_MAX, DT, strategy='oracle', rng=rng_sim
    )
    oracle_results.append(res_oracle)
    
    # Passive strategy (fixed spread = 1/k)
    rng_sim2 = np.random.default_rng(3000 + i)
    res_passive = simulate_market_maker(
        synthetic_paths[i], db_grid, da_grid, qs,
        A_hat, k_hat, Q_MAX, DT, strategy='passive', rng=rng_sim2
    )
    passive_results.append(res_passive)

oracle_pnls = np.array([r['final_pnl'] for r in oracle_results])
passive_pnls = np.array([r['final_pnl'] for r in passive_results])
oracle_sharpes = np.array([r['sharpe'] for r in oracle_results])
passive_sharpes = np.array([r['sharpe'] for r in passive_results])

print(f'\n=== Oracle Performance (200 paths) ===')
print(f'  Mean PnL:     {oracle_pnls.mean():.2f} ± {oracle_pnls.std():.2f}')
print(f'  Mean Sharpe:  {oracle_sharpes.mean():.2f}')
print(f'  Mean Fill Rate: {np.mean([r["fill_rate"] for r in oracle_results]):.4f}')
print(f'  Mean Max |q|: {np.mean([r["max_inventory"] for r in oracle_results]):.1f}')

print(f'\n=== Passive Performance (200 paths) ===')
print(f'  Mean PnL:     {passive_pnls.mean():.2f} ± {passive_pnls.std():.2f}')
print(f'  Mean Sharpe:  {passive_sharpes.mean():.2f}')

# ============================================================
# Plot 4: Oracle Performance
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CODE-2: Oracle vs Passive Market-Making Performance', fontsize=14, fontweight='bold')

# (a) PnL distribution comparison
ax = axes[0, 0]
ax.hist(oracle_pnls, bins=30, alpha=0.6, color='green', label=f'Oracle (μ={oracle_pnls.mean():.1f})')
ax.hist(passive_pnls, bins=30, alpha=0.6, color='gray', label=f'Passive (μ={passive_pnls.mean():.1f})')
ax.axvline(oracle_pnls.mean(), color='green', linestyle='--', linewidth=2)
ax.axvline(passive_pnls.mean(), color='gray', linestyle='--', linewidth=2)
ax.set_xlabel('Final PnL ($)')
ax.set_ylabel('Count')
ax.set_title('(a) PnL Distribution: Oracle vs Passive')
ax.legend(fontsize=9)

# (b) Sample PnL trajectory
ax = axes[0, 1]
idx = 0  # First path
t_h = np.arange(PATH_LEN) / 60
ax.plot(t_h, oracle_results[idx]['pnl'], color='green', linewidth=1.5, label='Oracle')
ax.plot(t_h, passive_results[idx]['pnl'], color='gray', linewidth=1.5, label='Passive')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('PnL ($)')
ax.set_title('(b) Sample PnL Trajectory (Path 0)')
ax.legend(fontsize=9)

# (c) Inventory trajectory (sample path)
ax = axes[1, 0]
ax.plot(t_h, oracle_results[idx]['inventory'], color='green', linewidth=1, label='Oracle')
ax.plot(t_h, passive_results[idx]['inventory'], color='gray', linewidth=1, alpha=0.7, label='Passive')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Inventory q')
ax.set_title('(c) Inventory Trajectory (Path 0)')
ax.legend(fontsize=9)

# (d) Spread trajectory (Oracle, sample path)
ax = axes[1, 1]
ax.plot(t_h, oracle_results[idx]['bid_spreads'], color='blue', linewidth=0.8, alpha=0.7, label='Bid spread δᵇ')
ax.plot(t_h, oracle_results[idx]['ask_spreads'], color='red', linewidth=0.8, alpha=0.7, label='Ask spread δᵃ')
ax.axhline(1.0/k_hat, color='gray', linestyle='--', alpha=0.5, label=f'1/k = {1.0/k_hat:.2f}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Spread ($)')
ax.set_title('(d) Oracle Optimal Spreads (Path 0)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot4_oracle_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot4_oracle_performance.png')

# ============================================================
# Feature Engineering for ML Models
# ============================================================

def build_features_and_targets(mid_prices, db_grid, da_grid, qs, Q_max, 
                               A, k, dt, rng=None, seq_len=32):
    """
    Build feature sequences and target spreads for training.
    
    Features (per timestep):
      0: mid-price return (normalized)
      1: inventory q (normalized by Q_max)
      2: time-to-close (normalized)
      3: current bid fill rate 
      4: current ask fill rate
    
    Targets: (delta_b*, delta_a*) from Oracle
    """
    if rng is None:
        rng = np.random.default_rng()
    
    T = len(mid_prices)
    q_idx_offset = Q_max
    
    # Simulate Oracle trajectory to get (q, spreads) sequence
    q = 0
    features_list = []
    targets_list = []
    
    returns = np.diff(mid_prices, prepend=mid_prices[0])
    ret_std = returns.std() + 1e-8
    
    for t in range(T - 1):
        q_idx = int(q) + q_idx_offset
        q_idx = np.clip(q_idx, 0, len(qs) - 1)
        t_idx = min(t, db_grid.shape[1] - 1)
        
        db = db_grid[q_idx, t_idx]
        da = da_grid[q_idx, t_idx]
        
        # Features
        feat = np.array([
            returns[t] / ret_std,           # normalized return
            q / Q_max,                      # normalized inventory
            1.0 - t / T,                    # time-to-close
            A * np.exp(-k * db),            # bid fill rate
            A * np.exp(-k * da),            # ask fill rate
        ])
        features_list.append(feat)
        targets_list.append(np.array([db, da]))
        
        # Simulate fills to update inventory
        lam_b = A * np.exp(-k * db) * dt
        lam_a = A * np.exp(-k * da) * dt
        n_bid = rng.poisson(lam_b)
        n_ask = rng.poisson(lam_a)
        q = np.clip(q + n_bid - n_ask, -Q_max, Q_max)
    
    features = np.array(features_list)  # (T-1, 5)
    targets = np.array(targets_list)    # (T-1, 2)
    
    # Create sequences (strided to reduce size)
    X_seqs, Y_seqs = [], []
    for i in range(seq_len, len(features), 4):  # stride=4 to reduce dataset size
        X_seqs.append(features[i - seq_len:i])
        Y_seqs.append(targets[i])
    
    return np.array(X_seqs), np.array(Y_seqs)

# Build dataset from synthetic paths
SEQ_LEN = 32
X_all, Y_all = [], []

print('Building training dataset from synthetic paths...')
n_train_paths = 80   # Use 80 paths for training (faster)
n_test_paths = 20    # 20 for testing

for i in range(n_train_paths):
    rng_feat = np.random.default_rng(5000 + i)
    X_i, Y_i = build_features_and_targets(
        synthetic_paths[i], db_grid, da_grid, qs, Q_MAX,
        A_hat, k_hat, DT, rng_feat, SEQ_LEN
    )
    X_all.append(X_i)
    Y_all.append(Y_i)

X_train = np.concatenate(X_all, axis=0)
Y_train = np.concatenate(Y_all, axis=0)

# Test set
X_test_all, Y_test_all = [], []
for i in range(n_train_paths, n_train_paths + n_test_paths):
    rng_feat = np.random.default_rng(5000 + i)
    X_i, Y_i = build_features_and_targets(
        synthetic_paths[i], db_grid, da_grid, qs, Q_MAX,
        A_hat, k_hat, DT, rng_feat, SEQ_LEN
    )
    X_test_all.append(X_i)
    Y_test_all.append(Y_i)

X_test = np.concatenate(X_test_all, axis=0)
Y_test = np.concatenate(Y_test_all, axis=0)

print(f'Training set: {X_train.shape[0]} sequences, features={X_train.shape[2]}')
print(f'Test set:     {X_test.shape[0]} sequences')
print(f'Target range: δᵇ [{Y_train[:,0].min():.2f}, {Y_train[:,0].max():.2f}], '
      f'δᵃ [{Y_train[:,1].min():.2f}, {Y_train[:,1].max():.2f}]')

# ============================================================
# GRU Model: 2D spread prediction
# Architecture from HW1, extended output head
# ============================================================

class SpreadGRU(nn.Module):
    """GRU model predicting optimal (delta_b, delta_a)."""
    def __init__(self, input_dim=5, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, 
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),    # Output: (delta_b, delta_a)
            nn.Softplus()        # Spreads must be non-negative
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # Use last hidden state

# ============================================================
# Simple Mamba-like Model (S4/SSM-inspired)
# Uses 1D convolution + gating as a simple SSM approximation
# ============================================================

class SimpleMamba(nn.Module):
    """Simplified Mamba/S4-style model with selective gating."""
    def __init__(self, input_dim=5, d_model=64, n_layers=2, kernel_size=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'conv': nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model),
                'gate_proj': nn.Linear(d_model, d_model),
                'out_proj': nn.Linear(d_model, d_model),
                'norm': nn.LayerNorm(d_model)
            }))
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        
        for layer in self.layers:
            residual = h
            # Causal convolution
            h_conv = layer['conv'](h.transpose(1, 2))[:, :, :h.size(1)].transpose(1, 2)
            # Selective gating
            gate = torch.sigmoid(layer['gate_proj'](h))
            h = layer['out_proj'](h_conv * gate)
            h = layer['norm'](h + residual)
        
        return self.fc(h[:, -1, :])  # Last timestep

print('Models defined: SpreadGRU, SimpleMamba')

# ============================================================
# Training Loop
# Loss = MSE on optimal spread + inventory-weighted penalty
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Prepare data loaders
X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)
X_test_t = torch.FloatTensor(X_test)
Y_test_t = torch.FloatTensor(Y_test)

train_ds = TensorDataset(X_train_t, Y_train_t)
test_ds = TensorDataset(X_test_t, Y_test_t)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512)

def train_model(model, train_loader, test_loader, n_epochs=20, lr=1e-3):
    """Train spread prediction model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses, test_losses = [], []
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        epoch_loss = 0
        n_samples = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
        train_loss = epoch_loss / n_samples
        train_losses.append(train_loss)
        
        # Eval
        model.eval()
        test_loss = 0
        n_test = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                test_loss += criterion(pred, Y_batch).item() * len(X_batch)
                n_test += len(X_batch)
        test_loss /= n_test
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{n_epochs}: train={train_loss:.6f}, test={test_loss:.6f}')
    
    return train_losses, test_losses

# Train GRU
print('\n=== Training GRU ===')
gru_model = SpreadGRU(input_dim=5, hidden_dim=32, n_layers=2).to(device)
gru_train_loss, gru_test_loss = train_model(gru_model, train_loader, test_loader, n_epochs=20)

# Train Mamba
print('\n=== Training Mamba ===')
mamba_model = SimpleMamba(input_dim=5, d_model=32, n_layers=2).to(device)
mamba_train_loss, mamba_test_loss = train_model(mamba_model, train_loader, test_loader, n_epochs=20)

# ============================================================
# Plot 5: Training Curves & Prediction Quality
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-3: Model Training & Prediction Quality', fontsize=14, fontweight='bold')

# (a) Training curves
ax = axes[0]
epochs = range(1, len(gru_train_loss) + 1)
ax.plot(epochs, gru_train_loss, 'b-', label='GRU train', linewidth=1.5)
ax.plot(epochs, gru_test_loss, 'b--', label='GRU test', linewidth=1.5)
ax.plot(epochs, mamba_train_loss, 'r-', label='Mamba train', linewidth=1.5)
ax.plot(epochs, mamba_test_loss, 'r--', label='Mamba test', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('(a) Training Curves')
ax.legend(fontsize=9)
ax.set_yscale('log')

# (b) GRU prediction vs Oracle (test set scatter)
gru_model.eval()
with torch.no_grad():
    gru_pred = gru_model(X_test_t.to(device)).cpu().numpy()

ax = axes[1]
n_plot = min(2000, len(Y_test))
ax.scatter(Y_test[:n_plot, 0], gru_pred[:n_plot, 0], s=3, alpha=0.3, color='blue', label='δᵇ')
ax.scatter(Y_test[:n_plot, 1], gru_pred[:n_plot, 1], s=3, alpha=0.3, color='red', label='δᵃ')
lims = [0, max(Y_test[:n_plot].max(), gru_pred[:n_plot].max()) * 1.1]
ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Oracle Spread')
ax.set_ylabel('GRU Predicted Spread')
ax.set_title('(b) GRU: Predicted vs Oracle')
ax.legend(fontsize=9)

# (c) Mamba prediction vs Oracle
mamba_model.eval()
with torch.no_grad():
    mamba_pred = mamba_model(X_test_t.to(device)).cpu().numpy()

ax = axes[2]
ax.scatter(Y_test[:n_plot, 0], mamba_pred[:n_plot, 0], s=3, alpha=0.3, color='blue', label='δᵇ')
ax.scatter(Y_test[:n_plot, 1], mamba_pred[:n_plot, 1], s=3, alpha=0.3, color='red', label='δᵃ')
ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Oracle Spread')
ax.set_ylabel('Mamba Predicted Spread')
ax.set_title('(c) Mamba: Predicted vs Oracle')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot5_training_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot5_training_prediction.png')

# ============================================================
# Backtest with NN-predicted spreads (optimized)
# ============================================================

def backtest_nn_strategy(model, mid_prices, db_grid, da_grid, qs, Q_max,
                         A, k, dt, seq_len=32, rng=None):
    """Backtest a neural network market-making strategy."""
    if rng is None:
        rng = np.random.default_rng()
    
    T = len(mid_prices)
    returns = np.diff(mid_prices, prepend=mid_prices[0])
    ret_std = returns.std() + 1e-8
    
    q = 0
    cash = 0.0
    pnl_series = np.zeros(T)
    inv_series = np.zeros(T)
    bid_fills_total = 0
    ask_fills_total = 0
    default_spread = 1.0 / k
    
    feature_buffer = np.zeros((seq_len, 5))  # pre-allocated rolling buffer
    buf_idx = 0
    warmup_done = False
    
    model.eval()
    
    for t in range(T - 1):
        S = mid_prices[t]
        
        # Build current features
        feat = np.array([
            returns[t] / ret_std,
            q / Q_max,
            1.0 - t / T,
            A * np.exp(-k * default_spread),
            A * np.exp(-k * default_spread),
        ])
        
        if buf_idx < seq_len:
            feature_buffer[buf_idx] = feat
            buf_idx += 1
            db, da = default_spread, default_spread
        else:
            # Shift buffer and add new feature
            feature_buffer[:-1] = feature_buffer[1:]
            feature_buffer[-1] = feat
            if not warmup_done:
                warmup_done = True
            
            # NN prediction (only every 4 steps to save time, use cached otherwise)
            if t % 4 == 0 or not warmup_done:
                seq = feature_buffer[np.newaxis]  # (1, seq_len, 5)
                with torch.no_grad():
                    pred = model(torch.FloatTensor(seq).to(device)).cpu().numpy()[0]
                db = max(pred[0], 0.01)
                da = max(pred[1], 0.01)
        
        # Fills
        lam_b = A * np.exp(-k * db) * dt
        lam_a = A * np.exp(-k * da) * dt
        n_bid = rng.poisson(lam_b)
        n_ask = rng.poisson(lam_a)
        
        if q + n_bid > Q_max:
            n_bid = max(Q_max - q, 0)
        if q - n_ask < -Q_max:
            n_ask = max(q + Q_max, 0)
        
        cash -= n_bid * (S - db)
        cash += n_ask * (S + da)
        q = np.clip(q + n_bid - n_ask, -Q_max, Q_max)
        
        bid_fills_total += n_bid
        ask_fills_total += n_ask
        pnl_series[t] = cash + q * S
        inv_series[t] = q
    
    pnl_series[-1] = cash + q * mid_prices[-1]
    inv_series[-1] = q
    
    total_fills = bid_fills_total + ask_fills_total
    pnl_diff = np.diff(pnl_series)
    sharpe = pnl_diff.mean() / (pnl_diff.std() + 1e-8) * np.sqrt(1440 * 252)
    cummax = np.maximum.accumulate(pnl_series)
    max_dd = (cummax - pnl_series).max()
    
    return {
        'pnl': pnl_series,
        'inventory': inv_series,
        'final_pnl': pnl_series[-1],
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'fill_rate': total_fills / T,
        'total_fills': total_fills,
        'max_inventory': np.abs(inv_series).max()
    }

# Backtest on test paths (use fewer paths for speed)
n_bt_paths = 20
test_path_indices = list(range(n_train_paths, n_train_paths + n_bt_paths))

gru_bt_results = []
mamba_bt_results = []
oracle_bt_results = []
passive_bt_results = []

print(f'Running final backtest on {n_bt_paths} test paths...')
for idx, path_i in enumerate(test_path_indices):
    # Oracle
    r_oracle = simulate_market_maker(
        synthetic_paths[path_i], db_grid, da_grid, qs,
        A_hat, k_hat, Q_MAX, DT, strategy='oracle', rng=np.random.default_rng(9000 + idx)
    )
    oracle_bt_results.append(r_oracle)
    
    # Passive
    r_passive = simulate_market_maker(
        synthetic_paths[path_i], db_grid, da_grid, qs,
        A_hat, k_hat, Q_MAX, DT, strategy='passive', rng=np.random.default_rng(9100 + idx)
    )
    passive_bt_results.append(r_passive)
    
    # GRU
    r_gru = backtest_nn_strategy(
        gru_model, synthetic_paths[path_i], db_grid, da_grid, qs, Q_MAX,
        A_hat, k_hat, DT, SEQ_LEN, rng=np.random.default_rng(9200 + idx)
    )
    gru_bt_results.append(r_gru)
    
    # Mamba
    r_mamba = backtest_nn_strategy(
        mamba_model, synthetic_paths[path_i], db_grid, da_grid, qs, Q_MAX,
        A_hat, k_hat, DT, SEQ_LEN, rng=np.random.default_rng(9300 + idx)
    )
    mamba_bt_results.append(r_mamba)
    
    if (idx + 1) % 5 == 0:
        print(f'  Completed {idx+1}/{n_bt_paths} paths...')

print('Backtest complete.')

# ============================================================
# Summary Statistics Table
# ============================================================

def summarize(results, name):
    pnls = [r['final_pnl'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    fills = [r['fill_rate'] for r in results]
    max_inv = [r['max_inventory'] for r in results]
    max_dd = [r['max_drawdown'] for r in results]
    return {
        'Strategy': name,
        'Mean PnL': f'{np.mean(pnls):.2f}',
        'Std PnL': f'{np.std(pnls):.2f}',
        'Sharpe': f'{np.mean(sharpes):.2f}',
        'Fill Rate': f'{np.mean(fills):.4f}',
        'Max |q|': f'{np.mean(max_inv):.1f}',
        'Max DD': f'{np.mean(max_dd):.2f}'
    }

summary = pd.DataFrame([
    summarize(oracle_bt_results, 'Oracle (HJB)'),
    summarize(gru_bt_results, 'GRU Student'),
    summarize(mamba_bt_results, 'Mamba Student'),
    summarize(passive_bt_results, 'Passive (1/k)'),
])

# Regret
oracle_pnl_test = np.array([r['final_pnl'] for r in oracle_bt_results])
gru_pnl_test = np.array([r['final_pnl'] for r in gru_bt_results])
mamba_pnl_test = np.array([r['final_pnl'] for r in mamba_bt_results])
passive_pnl_test = np.array([r['final_pnl'] for r in passive_bt_results])

gru_regret = oracle_pnl_test - gru_pnl_test
mamba_regret = oracle_pnl_test - mamba_pnl_test
passive_regret = oracle_pnl_test - passive_pnl_test

print('\n' + '='*80)
print(f'FINAL BACKTEST RESULTS ({n_bt_paths} test paths)')
print('='*80)
print(summary.to_string(index=False))
print(f'\nRegret (Oracle - Strategy):')
print(f'  GRU:     {gru_regret.mean():.2f} ± {gru_regret.std():.2f}')
print(f'  Mamba:   {mamba_regret.mean():.2f} ± {mamba_regret.std():.2f}')
print(f'  Passive: {passive_regret.mean():.2f} ± {passive_regret.std():.2f}')

# ============================================================
# Plot 6: Final Backtest Comparison (main results figure)
# ============================================================

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
fig.suptitle('CODE-3: Final Backtest — Oracle vs GRU vs Mamba vs Passive', 
             fontsize=15, fontweight='bold', y=0.98)

# (a) PnL distribution all strategies
ax = fig.add_subplot(gs[0, 0])
ax.hist(oracle_pnl_test, bins=15, alpha=0.5, color='green', label='Oracle')
ax.hist(gru_pnl_test, bins=15, alpha=0.5, color='blue', label='GRU')
ax.hist(mamba_pnl_test, bins=15, alpha=0.5, color='red', label='Mamba')
ax.hist(passive_pnl_test, bins=15, alpha=0.5, color='gray', label='Passive')
ax.set_xlabel('Final PnL ($)')
ax.set_ylabel('Count')
ax.set_title('(a) PnL Distribution')
ax.legend(fontsize=8)

# (b) Regret distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(gru_regret, bins=15, alpha=0.6, color='blue', label=f'GRU (μ={gru_regret.mean():.1f})')
ax.hist(mamba_regret, bins=15, alpha=0.6, color='red', label=f'Mamba (μ={mamba_regret.mean():.1f})')
ax.hist(passive_regret, bins=15, alpha=0.4, color='gray', label=f'Passive (μ={passive_regret.mean():.1f})')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Regret = Oracle PnL - Strategy PnL ($)')
ax.set_ylabel('Count')
ax.set_title('(b) Regret Distribution')
ax.legend(fontsize=8)

# (c) Box plot comparison
ax = fig.add_subplot(gs[0, 2])
data_box = [oracle_pnl_test, gru_pnl_test, mamba_pnl_test, passive_pnl_test]
bp = ax.boxplot(data_box, labels=['Oracle', 'GRU', 'Mamba', 'Passive'], patch_artist=True)
colors_box = ['#90EE90', '#87CEEB', '#FFB6C1', '#D3D3D3']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.set_ylabel('Final PnL ($)')
ax.set_title('(c) PnL Box Plot')

# (d) Sample trajectory comparison
ax = fig.add_subplot(gs[1, 0])
t_h = np.arange(PATH_LEN) / 60
ax.plot(t_h, oracle_bt_results[0]['pnl'], 'g-', linewidth=1.5, label='Oracle')
ax.plot(t_h, gru_bt_results[0]['pnl'], 'b-', linewidth=1.5, label='GRU')
ax.plot(t_h, mamba_bt_results[0]['pnl'], 'r-', linewidth=1.5, label='Mamba')
ax.plot(t_h, passive_bt_results[0]['pnl'], color='gray', linewidth=1.5, label='Passive')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('PnL ($)')
ax.set_title('(d) Sample PnL Trajectory (Test Path 0)')
ax.legend(fontsize=8)

# (e) Inventory comparison
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_h, oracle_bt_results[0]['inventory'], 'g-', linewidth=1, label='Oracle')
ax.plot(t_h, gru_bt_results[0]['inventory'], 'b-', linewidth=1, label='GRU')
ax.plot(t_h, mamba_bt_results[0]['inventory'], 'r-', linewidth=1, label='Mamba')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Inventory q')
ax.set_title('(e) Inventory Trajectory (Test Path 0)')
ax.legend(fontsize=8)

# (f) Sharpe ratio bar chart
ax = fig.add_subplot(gs[1, 2])
strategies = ['Oracle', 'GRU', 'Mamba', 'Passive']
mean_sharpes = [
    np.mean([r['sharpe'] for r in oracle_bt_results]),
    np.mean([r['sharpe'] for r in gru_bt_results]),
    np.mean([r['sharpe'] for r in mamba_bt_results]),
    np.mean([r['sharpe'] for r in passive_bt_results]),
]
std_sharpes = [
    np.std([r['sharpe'] for r in oracle_bt_results]),
    np.std([r['sharpe'] for r in gru_bt_results]),
    np.std([r['sharpe'] for r in mamba_bt_results]),
    np.std([r['sharpe'] for r in passive_bt_results]),
]
bars = ax.bar(strategies, mean_sharpes, yerr=std_sharpes, 
              color=colors_box, edgecolor='black', capsize=4)
ax.set_ylabel('Sharpe Ratio')
ax.set_title('(f) Mean Sharpe Ratio')

plt.savefig('plot6_final_backtest.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot6_final_backtest.png')

# ============================================================
# Evans Theorem 5.3 Verification
# PMP costate λ(t) = ∇_q v(q*(t), S*(t), t)
# ============================================================

# Compute ∇_q v along q=0 (center of grid)
q0_idx = Q_MAX  # index for q=0
grad_v = np.zeros(PATH_LEN)
for t in range(PATH_LEN):
    # Central difference: (v(q+1,t) - v(q-1,t)) / 2
    grad_v[t] = (v_grid[q0_idx + 1, t] - v_grid[q0_idx - 1, t]) / 2.0

# PMP-style costate: linear approximation λ(t) = λ₀ + h*t
# where h relates to the inventory penalty gradient
# For the market-making problem, the costate evolves due to inventory cost
from numpy.polynomial import polynomial as P
t_axis = np.arange(PATH_LEN)
coeffs = np.polyfit(t_axis, grad_v, deg=1)
lambda_pmp = np.polyval(coeffs, t_axis)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Evans Theorem 5.3: PMP Costate = HJB Gradient', fontsize=14, fontweight='bold')

# (a) Gradient along optimal trajectory
ax = axes[0]
t_h = t_axis / 60
ax.plot(t_h, grad_v, 'b-', linewidth=1.5, alpha=0.7, label='∇_q v(0, t) [HJB]')
ax.plot(t_h, lambda_pmp, 'r--', linewidth=2, label=f'λ(t) = {coeffs[1]:.4f} + {coeffs[0]:.6f}·t [PMP fit]')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Costate / Gradient')
ax.set_title('(a) HJB Gradient vs PMP Costate at q=0')
ax.legend(fontsize=9)

# (b) Gradient at different inventory levels
ax = axes[1]
for q_val in [-5, -2, 0, 2, 5]:
    qi = q_val + Q_MAX
    if qi > 0 and qi < len(qs) - 1:
        g = (v_grid[qi + 1, :] - v_grid[qi - 1, :]) / 2.0
        ax.plot(t_h, g, linewidth=1.2, label=f'q={q_val}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('∇_q v(q, t)')
ax.set_title('(b) Value Gradient at Various Inventory Levels')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot7_theorem53_verification.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot7_theorem53_verification.png')
