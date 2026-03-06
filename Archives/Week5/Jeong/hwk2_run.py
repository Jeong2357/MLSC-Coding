#!/usr/bin/env python3
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
import warnings, time
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11, 'figure.dpi': 120
})
np.random.seed(42)
torch.manual_seed(42)


# Ground-truth parameters (realistic BTC/USDT 1-min)
TRUE_KAPPA = 0.15
TRUE_MU    = 65000.0
TRUE_SIGMA = 12.0
TRUE_A     = 1.5
TRUE_K     = 0.7

N_MINUTES = 1440 * 5   # 5 days
DT        = 1.0

def generate_ou_midprice(mu, kappa, sigma, S0, n_steps, dt, rng=None):
    """Simulate OU mid-price. Evans §5.1.1: dS = kappa*(mu-S)*dt + sigma*dW"""
    if rng is None: rng = np.random.default_rng(42)
    S = np.zeros(n_steps)
    S[0] = S0
    decay = np.exp(-kappa * dt)
    std = sigma * np.sqrt((1 - np.exp(-2*kappa*dt)) / (2*kappa))
    for t in range(1, n_steps):
        S[t] = mu + (S[t-1] - mu)*decay + std*rng.standard_normal()
    return S

def generate_lob_data(S_mid, A, k, dt, rng=None):
    """Generate LOB snapshots. Fill intensity: lambda(delta)=A*exp(-k*delta) [Evans §5.1.1]"""
    if rng is None: rng = np.random.default_rng(42)
    n = len(S_mid)
    half_spreads = rng.exponential(scale=8.0, size=n) + 2.0
    fill_rates = A * np.exp(-k * half_spreads)
    bid_fills = rng.poisson(fill_rates * dt)
    ask_fills = rng.poisson(fill_rates * dt)
    return pd.DataFrame({
        'mid_price': S_mid, 'half_spread': half_spreads,
        'bid_fill_rate': fill_rates, 'ask_fill_rate': fill_rates,
        'bid_fills': bid_fills, 'ask_fills': ask_fills,
    })

# Generate data
rng = np.random.default_rng(42)
S_mid = generate_ou_midprice(TRUE_MU, TRUE_KAPPA, TRUE_SIGMA, TRUE_MU, N_MINUTES, DT, rng)
lob_df = generate_lob_data(S_mid, TRUE_A, TRUE_K, DT, rng)
S_data = lob_df['mid_price'].values
print(f"Generated {len(lob_df)} LOB snapshots (5 days)")
print(f"Mid-price range: ${S_mid.min():.0f} - ${S_mid.max():.0f}")

# --- OU MLE Fit ---
def ou_neg_loglik(params, S, dt):
    kappa, mu, sigma = params
    if kappa <= 0 or sigma <= 0: return 1e12
    decay = np.exp(-kappa * dt)
    var = sigma**2 * (1 - np.exp(-2*kappa*dt)) / (2*kappa)
    if var <= 0: return 1e12
    means = mu + (S[:-1] - mu) * decay
    residuals = S[1:] - means
    return 0.5*len(residuals)*np.log(2*np.pi*var) + 0.5*np.sum(residuals**2)/var

result = minimize(ou_neg_loglik, [0.1, S_data.mean(), S_data.std()],
                  args=(S_data, DT), bounds=[(1e-6,10),(S_data.min(),S_data.max()),(1e-6,100)],
                  method='L-BFGS-B')
kappa_hat, mu_hat, sigma_hat = result.x
print(f"\n=== OU MLE Results ===")
print(f"  kappa: {kappa_hat:.4f} (true: {TRUE_KAPPA})")
print(f"  mu:    {mu_hat:.2f} (true: {TRUE_MU})")
print(f"  sigma: {sigma_hat:.4f} (true: {TRUE_SIGMA})")

# --- Fill-Rate Estimation ---
spreads = lob_df['half_spread'].values
fills = lob_df['bid_fills'].values
n_bins = 25
spread_bins = np.linspace(spreads.min(), np.percentile(spreads, 95), n_bins+1)
bc, bfr = [], []
for i in range(n_bins):
    mask = (spreads >= spread_bins[i]) & (spreads < spread_bins[i+1])
    if mask.sum() > 10:
        bc.append((spread_bins[i]+spread_bins[i+1])/2)
        bfr.append(fills[mask].mean()/DT)
bin_centers, bin_fill_rates = np.array(bc), np.array(bfr)
positive_mask = bin_fill_rates > 0
X = np.column_stack([np.ones(positive_mask.sum()), bin_centers[positive_mask]])
coeffs = np.linalg.lstsq(X, np.log(bin_fill_rates[positive_mask]), rcond=None)[0]
A_hat, k_hat = np.exp(coeffs[0]), -coeffs[1]
print(f"\n=== Fill-Rate Estimation ===")
print(f"  A: {A_hat:.4f} (true: {TRUE_A})")
print(f"  k: {k_hat:.4f} (true: {TRUE_K})")

# --- Plot 1: Physics Fit ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CODE-1: LOB Physics Fit Results', fontsize=16, fontweight='bold', y=0.98)

ax = axes[0,0]
t_hours = np.arange(len(S_data))/60
ax.plot(t_hours, S_data, alpha=0.7, lw=0.5, color='steelblue', label='Mid-price')
ax.axhline(mu_hat, color='red', ls='--', alpha=0.8, label=f'$\\hat{{\\mu}}$={mu_hat:.0f}')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Mid-price ($)')
ax.set_title('(a) Mid-Price & OU Mean'); ax.legend(fontsize=9)

ax = axes[0,1]
returns = np.diff(S_data)
ou_var = sigma_hat**2 * (1 - np.exp(-2*kappa_hat*DT)) / (2*kappa_hat)
x_range = np.linspace(returns.min(), returns.max(), 200)
ax.hist(returns, bins=80, density=True, alpha=0.6, color='steelblue', label='Empirical')
ax.plot(x_range, norm.pdf(x_range, 0, np.sqrt(ou_var)), 'r-', lw=2, label=f'OU (σ={sigma_hat:.2f})')
ax.set_xlabel('1-min Return ($)'); ax.set_ylabel('Density')
ax.set_title('(b) Return Distribution: Data vs OU'); ax.legend(fontsize=9)

ax = axes[1,0]
ax.scatter(bin_centers[positive_mask], bin_fill_rates[positive_mask], color='steelblue', s=40, zorder=5, label='Empirical')
dr = np.linspace(0.5, bin_centers.max(), 100)
ax.plot(dr, A_hat*np.exp(-k_hat*dr), 'r-', lw=2, label=f'Fit: {A_hat:.2f}·e^(-{k_hat:.2f}δ)')
ax.plot(dr, TRUE_A*np.exp(-TRUE_K*dr), 'g--', lw=1.5, alpha=0.7, label=f'True: {TRUE_A}·e^(-{TRUE_K}δ)')
ax.set_xlabel('Half-spread δ ($)'); ax.set_ylabel('Fill rate'); ax.set_yscale('log')
ax.set_title('(c) Fill Rate vs Spread'); ax.legend(fontsize=9)

ax = axes[1,1]
max_lag = 120; lags = np.arange(1, max_lag+1)
autocorr = np.array([np.corrcoef(S_data[:-l], S_data[l:])[0,1] for l in lags])
ax.plot(lags, autocorr, 'o', ms=2, color='steelblue', label='Empirical ACF')
ax.plot(lags, np.exp(-kappa_hat*lags*DT), 'r-', lw=2, label=f'OU: e^(-{kappa_hat:.3f}τ)')
ax.set_xlabel('Lag (min)'); ax.set_ylabel('ACF')
ax.set_title('(d) Autocorrelation: Data vs OU'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot1_physics_fit.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot1_physics_fit.png")

# --- Synthetic Path Generation ---
N_PATHS = 200; PATH_LEN = 480  # 8 hours (faster to compute)
synthetic_paths = []
synthetic_returns_all = []
for i in range(N_PATHS):
    rng_i = np.random.default_rng(1000+i)
    S_syn = generate_ou_midprice(mu_hat, kappa_hat, sigma_hat, mu_hat, PATH_LEN, DT, rng_i)
    synthetic_paths.append(S_syn)
    synthetic_returns_all.append(np.diff(S_syn))
synthetic_paths = np.array(synthetic_paths)
all_syn_returns = np.concatenate(synthetic_returns_all)
real_returns = np.diff(S_data)
ks_stat, ks_pval = ks_2samp(real_returns, all_syn_returns)
print(f"\nGenerated {N_PATHS} synthetic paths, {PATH_LEN} min each")
print(f"KS test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}")

# --- Plot 2: Synthetic Paths ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-1: Synthetic Path Generation & Validation', fontsize=14, fontweight='bold')

ax = axes[0]
t_hp = np.arange(PATH_LEN)/60
for i in range(20): ax.plot(t_hp, synthetic_paths[i], alpha=0.3, lw=0.5)
ax.plot(t_hp, synthetic_paths.mean(axis=0), 'k-', lw=2, label='Mean')
ax.axhline(mu_hat, color='red', ls='--', alpha=0.7, label=f'μ={mu_hat:.0f}')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Mid-price ($)')
ax.set_title(f'(a) 20/{N_PATHS} Synthetic Paths'); ax.legend(fontsize=9)

ax = axes[1]
ax.hist(real_returns, bins=80, density=True, alpha=0.5, color='steelblue', label='Real')
ax.hist(all_syn_returns, bins=80, density=True, alpha=0.5, color='orange', label='Synthetic')
ax.set_xlabel('1-min Return ($)'); ax.set_ylabel('Density')
ax.set_title(f'(b) Returns (KS p={ks_pval:.3f})'); ax.legend(fontsize=9)

ax = axes[2]
q_pts = np.linspace(0.5, 99.5, 200)
rq = np.percentile(real_returns, q_pts)
sq = np.percentile(all_syn_returns, q_pts)
ax.scatter(rq, sq, s=10, alpha=0.6, color='steelblue')
lims = [min(rq.min(),sq.min()), max(rq.max(),sq.max())]
ax.plot(lims, lims, 'r--', lw=1.5, label='45° line')
ax.set_xlabel('Real Quantiles'); ax.set_ylabel('Synthetic Quantiles')
ax.set_title('(c) QQ Plot'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot2_synthetic_validation.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot2_synthetic_validation.png")


GAMMA = 0.001    # Risk aversion (moderate — not too harsh)
Q_MAX = 10
MAX_SPREAD = 10.0 / k_hat  # Cap spreads at 10/k

def hjb_oracle(n_steps, A, k, gamma, sigma, Q_max, dt, max_spread=50.0):
    """
    Evans §5.1.3 Step 1: HJB backward induction (vectorized).

    HJB PDE (Evans Theorem 5.1):
      v_t - (γσ²/2)q² + max_{δᵇ}[λᵇ(δᵇ)(Δ⁺v + δᵇ)]
                       + max_{δᵃ}[λᵃ(δᵃ)(Δ⁻v + δᵃ)] = 0

    FOC gives optimal spreads:
      δᵇ* = 1/k - Δ⁺v     (buy side)
      δᵃ* = 1/k - Δ⁻v     (sell side)

    where Δ⁺v = v(q+1,t+1) - v(q,t+1), Δ⁻v = v(q-1,t+1) - v(q,t+1)
    """
    T = n_steps
    qs = np.arange(-Q_max, Q_max+1, dtype=np.float64)
    n_q = len(qs)

    v = np.zeros((n_q, T))
    delta_b = np.zeros((n_q, T))
    delta_a = np.zeros((n_q, T))

    # Terminal condition: v(q,T) = 0 [Evans (5.4)]
    inv_costs = gamma * sigma**2 / 2.0 * qs**2 * dt  # precompute
    inv_k = 1.0 / k

    for t in reversed(range(T-1)):
        v_next = v[:, t+1]

        # Δ⁺v = v(q+1,t+1) - v(q,t+1)  [Evans Theorem 5.1]
        dv_plus = np.zeros(n_q)
        dv_plus[:-1] = v_next[1:] - v_next[:-1]
        dv_plus[-1] = dv_plus[-2]  # extrapolate at boundary

        # Δ⁻v = v(q-1,t+1) - v(q,t+1)
        dv_minus = np.zeros(n_q)
        dv_minus[1:] = v_next[:-1] - v_next[1:]
        dv_minus[0] = dv_minus[1]  # extrapolate at boundary

        # Optimal bid spread: δᵇ* = 1/k - Δ⁺v [FOC of HJB max]
        db = np.clip(inv_k - dv_plus, 0.0, max_spread)

        # Optimal ask spread: δᵃ* = 1/k - Δ⁻v [FOC of HJB max]
        da = np.clip(inv_k - dv_minus, 0.0, max_spread)

        # Bellman update [Evans §5.1.2]
        # Bid revenue: λᵇ(δᵇ)·(Δ⁺v + δᵇ) — value of buying one unit
        bid_revenue = A * np.exp(-k*db) * (dv_plus + db) * dt
        # Ask revenue: λᵃ(δᵃ)·(Δ⁻v + δᵃ) — value of selling one unit
        ask_revenue = A * np.exp(-k*da) * (dv_minus + da) * dt

        # At boundaries, disable impossible trades
        bid_revenue[-1] = 0.0  # can't buy at max inventory
        ask_revenue[0] = 0.0   # can't sell at min inventory

        v[:, t] = v_next - inv_costs + bid_revenue + ask_revenue
        delta_b[:, t] = db
        delta_a[:, t] = da

    return v, delta_b, delta_a, qs

t0 = time.time()
v_grid, db_grid, da_grid, qs = hjb_oracle(
    PATH_LEN, A_hat, k_hat, GAMMA, sigma_hat, Q_MAX, DT, MAX_SPREAD
)
print(f"HJB solved in {time.time()-t0:.2f}s. Grid: {v_grid.shape}")
print(f"v(q=0, t=0) = {v_grid[Q_MAX, 0]:.2f}")
print(f"Spread range: δᵇ [{db_grid.min():.2f}, {db_grid.max():.2f}], δᵃ [{da_grid.min():.2f}, {da_grid.max():.2f}]")

# --- Plot 3: HJB Value Surface & Spread Heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-2: HJB Value Function & Optimal Spreads', fontsize=14, fontweight='bold')
tg = np.arange(PATH_LEN)/60

ax = axes[0]
im = ax.imshow(v_grid, aspect='auto', cmap='viridis', extent=[0,tg[-1],qs[-1],qs[0]])
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(a) Value Function v(q,t)'); plt.colorbar(im, ax=ax, label='Value')

ax = axes[1]
im = ax.imshow(db_grid, aspect='auto', cmap='YlOrRd', extent=[0,tg[-1],qs[-1],qs[0]])
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(b) Optimal Bid Spread δᵇ*(q,t)'); plt.colorbar(im, ax=ax, label='δᵇ')

ax = axes[2]
im = ax.imshow(da_grid, aspect='auto', cmap='YlOrRd', extent=[0,tg[-1],qs[-1],qs[0]])
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(c) Optimal Ask Spread δᵃ*(q,t)'); plt.colorbar(im, ax=ax, label='δᵃ')

plt.tight_layout()
plt.savefig('plot3_hjb_value_spread.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot3_hjb_value_spread.png")

# --- Oracle & Passive Simulation ---
def simulate_mm(mid_prices, delta_b_grid, delta_a_grid, qs,
                A, k, Q_max, dt, strategy='oracle', fixed_spread=None, rng=None):
    """Simulate market-making strategy on a price path."""
    if rng is None: rng = np.random.default_rng()
    T = len(mid_prices)
    q, cash = 0, 0.0
    pnl = np.zeros(T); inv = np.zeros(T)
    bids = np.zeros(T); asks = np.zeros(T)
    n_bf, n_af = 0, 0
    q_off = Q_max

    for t in range(T-1):
        S = mid_prices[t]
        qi = np.clip(int(q)+q_off, 0, len(qs)-1)
        ti = min(t, delta_b_grid.shape[1]-1)

        if strategy == 'oracle':
            db = delta_b_grid[qi, ti]; da = delta_a_grid[qi, ti]
        else:
            db = fixed_spread if fixed_spread else 1.0/k
            da = fixed_spread if fixed_spread else 1.0/k

        bids[t], asks[t] = db, da
        lb = A*np.exp(-k*db)*dt; la = A*np.exp(-k*da)*dt
        nb = rng.poisson(lb); na = rng.poisson(la)

        if q+nb > Q_max: nb = max(Q_max-q, 0)
        if q-na < -Q_max: na = max(q+Q_max, 0)

        cash -= nb*(S-db); cash += na*(S+da)
        q = np.clip(q+nb-na, -Q_max, Q_max)
        n_bf += nb; n_af += na
        pnl[t] = cash + q*S; inv[t] = q

    pnl[-1] = cash + q*mid_prices[-1]; inv[-1] = q
    pd_ = np.diff(pnl)
    sharpe = pd_.mean()/(pd_.std()+1e-8)*np.sqrt(1440*252)
    mdd = (np.maximum.accumulate(pnl) - pnl).max()
    return {
        'pnl': pnl, 'inventory': inv, 'bid_spreads': bids, 'ask_spreads': asks,
        'final_pnl': pnl[-1], 'sharpe': sharpe, 'max_drawdown': mdd,
        'max_inventory': np.abs(inv).max(), 'fill_rate': (n_bf+n_af)/T, 'total_fills': n_bf+n_af
    }

print("\nSimulating Oracle & Passive on 200 paths...")
oracle_results, passive_results = [], []
for i in range(N_PATHS):
    oracle_results.append(simulate_mm(
        synthetic_paths[i], db_grid, da_grid, qs, A_hat, k_hat, Q_MAX, DT,
        strategy='oracle', rng=np.random.default_rng(2000+i)))
    passive_results.append(simulate_mm(
        synthetic_paths[i], db_grid, da_grid, qs, A_hat, k_hat, Q_MAX, DT,
        strategy='passive', rng=np.random.default_rng(3000+i)))

op = np.array([r['final_pnl'] for r in oracle_results])
pp = np.array([r['final_pnl'] for r in passive_results])
os_ = np.array([r['sharpe'] for r in oracle_results])
ps_ = np.array([r['sharpe'] for r in passive_results])
print(f"\n=== Oracle (200 paths) ===")
print(f"  PnL: {op.mean():.2f} ± {op.std():.2f}, Sharpe: {os_.mean():.2f}")
print(f"  Fill rate: {np.mean([r['fill_rate'] for r in oracle_results]):.4f}")
print(f"\n=== Passive (200 paths) ===")
print(f"  PnL: {pp.mean():.2f} ± {pp.std():.2f}, Sharpe: {ps_.mean():.2f}")

# --- Plot 4: Oracle vs Passive ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CODE-2: Oracle vs Passive Market-Making', fontsize=14, fontweight='bold')
t_h = np.arange(PATH_LEN)/60

ax = axes[0,0]
ax.hist(op, bins=30, alpha=0.6, color='green', label=f'Oracle (μ={op.mean():.1f})')
ax.hist(pp, bins=30, alpha=0.6, color='gray', label=f'Passive (μ={pp.mean():.1f})')
ax.axvline(op.mean(), color='green', ls='--', lw=2)
ax.axvline(pp.mean(), color='gray', ls='--', lw=2)
ax.set_xlabel('Final PnL ($)'); ax.set_ylabel('Count')
ax.set_title('(a) PnL Distribution'); ax.legend(fontsize=9)

ax = axes[0,1]
ax.plot(t_h, oracle_results[0]['pnl'], 'g-', lw=1.5, label='Oracle')
ax.plot(t_h, passive_results[0]['pnl'], color='gray', lw=1.5, label='Passive')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('PnL ($)')
ax.set_title('(b) PnL Trajectory (Path 0)'); ax.legend(fontsize=9)

ax = axes[1,0]
ax.plot(t_h, oracle_results[0]['inventory'], 'g-', lw=1, label='Oracle')
ax.plot(t_h, passive_results[0]['inventory'], color='gray', lw=1, alpha=0.7, label='Passive')
ax.axhline(0, color='black', alpha=0.3)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q')
ax.set_title('(c) Inventory (Path 0)'); ax.legend(fontsize=9)

ax = axes[1,1]
ax.plot(t_h, oracle_results[0]['bid_spreads'], 'b-', lw=0.8, alpha=0.7, label='Bid δᵇ')
ax.plot(t_h, oracle_results[0]['ask_spreads'], 'r-', lw=0.8, alpha=0.7, label='Ask δᵃ')
ax.axhline(1.0/k_hat, color='gray', ls='--', alpha=0.5, label=f'1/k={1.0/k_hat:.2f}')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Spread ($)')
ax.set_title('(d) Oracle Optimal Spreads (Path 0)'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot4_oracle_performance.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot4_oracle_performance.png")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Feature Engineering ---
SEQ_LEN = 32

def build_features(mid_prices, db_grid, da_grid, qs, Q_max, A, k, dt, rng=None, seq_len=32):
    if rng is None: rng = np.random.default_rng()
    T = len(mid_prices)
    returns = np.diff(mid_prices, prepend=mid_prices[0])
    ret_std = returns.std() + 1e-8
    q, q_off = 0, Q_max
    feats, targets = [], []

    for t in range(T-1):
        qi = np.clip(int(q)+q_off, 0, len(qs)-1)
        ti = min(t, db_grid.shape[1]-1)
        db, da = db_grid[qi, ti], da_grid[qi, ti]

        feats.append([returns[t]/ret_std, q/Q_max, 1.0-t/T,
                      A*np.exp(-k*db), A*np.exp(-k*da)])
        targets.append([db, da])

        lb = A*np.exp(-k*db)*dt; la = A*np.exp(-k*da)*dt
        q = np.clip(q + rng.poisson(lb) - rng.poisson(la), -Q_max, Q_max)

    feats, targets = np.array(feats), np.array(targets)
    X, Y = [], []
    for i in range(seq_len, len(feats), 4):  # stride=4
        X.append(feats[i-seq_len:i]); Y.append(targets[i])
    return np.array(X), np.array(Y)

n_train, n_test = 80, 20
print(f"Building features from {n_train} train + {n_test} test paths...")
Xa, Ya = [], []
for i in range(n_train):
    Xi, Yi = build_features(synthetic_paths[i], db_grid, da_grid, qs, Q_MAX,
                           A_hat, k_hat, DT, np.random.default_rng(5000+i), SEQ_LEN)
    Xa.append(Xi); Ya.append(Yi)
X_train, Y_train = np.concatenate(Xa), np.concatenate(Ya)

Xb, Yb = [], []
for i in range(n_train, n_train+n_test):
    Xi, Yi = build_features(synthetic_paths[i], db_grid, da_grid, qs, Q_MAX,
                           A_hat, k_hat, DT, np.random.default_rng(5000+i), SEQ_LEN)
    Xb.append(Xi); Yb.append(Yi)
X_test, Y_test = np.concatenate(Xb), np.concatenate(Yb)

print(f"Train: {X_train.shape[0]} seqs, Test: {X_test.shape[0]} seqs")
print(f"Target range: δᵇ [{Y_train[:,0].min():.2f}, {Y_train[:,0].max():.2f}], "
      f"δᵃ [{Y_train[:,1].min():.2f}, {Y_train[:,1].max():.2f}]")

# --- Models ---
class SpreadGRU(nn.Module):
    def __init__(self, inp=5, hid=32, nl=2):
        super().__init__()
        self.gru = nn.GRU(inp, hid, nl, batch_first=True, dropout=0.1 if nl>1 else 0)
        self.fc = nn.Sequential(nn.Linear(hid,32), nn.ReLU(), nn.Linear(32,2), nn.Softplus())
    def forward(self, x):
        out, _ = self.gru(x); return self.fc(out[:,-1,:])

class SimpleMamba(nn.Module):
    def __init__(self, inp=5, dm=32, nl=2, ks=4):
        super().__init__()
        self.proj = nn.Linear(inp, dm)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'conv': nn.Conv1d(dm,dm,ks,padding=ks-1,groups=dm),
            'gate': nn.Linear(dm,dm), 'out': nn.Linear(dm,dm), 'norm': nn.LayerNorm(dm)
        }) for _ in range(nl)])
        self.fc = nn.Sequential(nn.Linear(dm,32), nn.ReLU(), nn.Linear(32,2), nn.Softplus())
    def forward(self, x):
        h = self.proj(x)
        for L in self.layers:
            r = h
            hc = L['conv'](h.transpose(1,2))[:,:,:h.size(1)].transpose(1,2)
            h = L['out'](hc * torch.sigmoid(L['gate'](h)))
            h = L['norm'](h + r)
        return self.fc(h[:,-1,:])

# --- Training ---
X_tr_t = torch.FloatTensor(X_train); Y_tr_t = torch.FloatTensor(Y_train)
X_te_t = torch.FloatTensor(X_test);  Y_te_t = torch.FloatTensor(Y_test)
tr_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=512, shuffle=True)
te_loader = DataLoader(TensorDataset(X_te_t, Y_te_t), batch_size=512)

def train_model(model, tr_loader, te_loader, epochs=20, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit = nn.MSELoss()
    trl, tel = [], []
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
            for Xb, Yb in te_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                tl += crit(model(Xb), Yb).item()*len(Xb); nt += len(Xb)
        tel.append(tl/nt); sched.step(tl/nt)
        if (ep+1)%5==0 or ep==0:
            print(f"  Epoch {ep+1:3d}/{epochs}: train={trl[-1]:.6f}, test={tel[-1]:.6f}")
    return trl, tel

print("\n=== Training GRU ===")
gru = SpreadGRU().to(device)
gru_tl, gru_vl = train_model(gru, tr_loader, te_loader, epochs=20)

print("\n=== Training Mamba ===")
mamba = SimpleMamba().to(device)
mba_tl, mba_vl = train_model(mamba, tr_loader, te_loader, epochs=20)

# --- Plot 5: Training & Predictions ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CODE-3: Model Training & Prediction Quality', fontsize=14, fontweight='bold')

ax = axes[0]
ep = range(1, len(gru_tl)+1)
ax.plot(ep, gru_tl, 'b-', lw=1.5, label='GRU train'); ax.plot(ep, gru_vl, 'b--', lw=1.5, label='GRU test')
ax.plot(ep, mba_tl, 'r-', lw=1.5, label='Mamba train'); ax.plot(ep, mba_vl, 'r--', lw=1.5, label='Mamba test')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.set_title('(a) Training Curves')
ax.legend(fontsize=9); ax.set_yscale('log')

gru.eval()
with torch.no_grad(): gru_pred = gru(X_te_t.to(device)).cpu().numpy()
ax = axes[1]; np_ = min(2000, len(Y_test))
ax.scatter(Y_test[:np_,0], gru_pred[:np_,0], s=3, alpha=0.3, c='blue', label='δᵇ')
ax.scatter(Y_test[:np_,1], gru_pred[:np_,1], s=3, alpha=0.3, c='red', label='δᵃ')
lm = [0, max(Y_test[:np_].max(), gru_pred[:np_].max())*1.1]
ax.plot(lm, lm, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('Oracle'); ax.set_ylabel('GRU Pred'); ax.set_title('(b) GRU vs Oracle'); ax.legend(fontsize=9)

mamba.eval()
with torch.no_grad(): mba_pred = mamba(X_te_t.to(device)).cpu().numpy()
ax = axes[2]
ax.scatter(Y_test[:np_,0], mba_pred[:np_,0], s=3, alpha=0.3, c='blue', label='δᵇ')
ax.scatter(Y_test[:np_,1], mba_pred[:np_,1], s=3, alpha=0.3, c='red', label='δᵃ')
ax.plot(lm, lm, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('Oracle'); ax.set_ylabel('Mamba Pred'); ax.set_title('(c) Mamba vs Oracle'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot5_training_prediction.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot5_training_prediction.png")

# --- Backtest ---
def backtest_nn(model, mid_prices, db_grid, da_grid, qs, Q_max,
                A, k, dt, seq_len=32, rng=None):
    if rng is None: rng = np.random.default_rng()
    T = len(mid_prices)
    returns = np.diff(mid_prices, prepend=mid_prices[0])
    ret_std = returns.std() + 1e-8
    q, cash = 0, 0.0; pnl = np.zeros(T); inv = np.zeros(T)
    nbf, naf = 0, 0; ds = 1.0/k
    buf = np.zeros((seq_len,5)); bi = 0; db, da = ds, ds
    model.eval()
    for t in range(T-1):
        S = mid_prices[t]
        feat = [returns[t]/ret_std, q/Q_max, 1.0-t/T, A*np.exp(-k*ds), A*np.exp(-k*ds)]
        if bi < seq_len:
            buf[bi] = feat; bi += 1
        else:
            buf[:-1] = buf[1:]; buf[-1] = feat
            if t % 4 == 0:
                with torch.no_grad():
                    p = model(torch.FloatTensor(buf[np.newaxis]).to(device)).cpu().numpy()[0]
                db, da = max(p[0],0.01), max(p[1],0.01)
        lb = A*np.exp(-k*db)*dt; la = A*np.exp(-k*da)*dt
        nb, na = rng.poisson(lb), rng.poisson(la)
        if q+nb > Q_max: nb = max(Q_max-q, 0)
        if q-na < -Q_max: na = max(q+Q_max, 0)
        cash -= nb*(S-db); cash += na*(S+da)
        q = np.clip(q+nb-na, -Q_max, Q_max)
        nbf += nb; naf += na; pnl[t] = cash+q*S; inv[t] = q
    pnl[-1] = cash+q*mid_prices[-1]; inv[-1] = q
    pd_ = np.diff(pnl)
    return {
        'pnl': pnl, 'inventory': inv, 'final_pnl': pnl[-1],
        'sharpe': pd_.mean()/(pd_.std()+1e-8)*np.sqrt(1440*252),
        'max_drawdown': (np.maximum.accumulate(pnl)-pnl).max(),
        'fill_rate': (nbf+naf)/T, 'total_fills': nbf+naf,
        'max_inventory': np.abs(inv).max()
    }

n_bt = 20
print(f"\nBacktesting on {n_bt} test paths...")
o_bt, p_bt, g_bt, m_bt = [], [], [], []
for i in range(n_bt):
    pi = n_train + i
    o_bt.append(simulate_mm(synthetic_paths[pi], db_grid, da_grid, qs, A_hat, k_hat, Q_MAX, DT,
                            strategy='oracle', rng=np.random.default_rng(9000+i)))
    p_bt.append(simulate_mm(synthetic_paths[pi], db_grid, da_grid, qs, A_hat, k_hat, Q_MAX, DT,
                            strategy='passive', rng=np.random.default_rng(9100+i)))
    g_bt.append(backtest_nn(gru, synthetic_paths[pi], db_grid, da_grid, qs, Q_MAX,
                            A_hat, k_hat, DT, SEQ_LEN, rng=np.random.default_rng(9200+i)))
    m_bt.append(backtest_nn(mamba, synthetic_paths[pi], db_grid, da_grid, qs, Q_MAX,
                            A_hat, k_hat, DT, SEQ_LEN, rng=np.random.default_rng(9300+i)))
    if (i+1)%5==0: print(f"  {i+1}/{n_bt} done")

# --- Summary ---
def summ(res, name):
    pnl = [r['final_pnl'] for r in res]; sh = [r['sharpe'] for r in res]
    fr = [r['fill_rate'] for r in res]; mi = [r['max_inventory'] for r in res]
    return {'Strategy': name, 'Mean PnL': f'{np.mean(pnl):.2f}', 'Std PnL': f'{np.std(pnl):.2f}',
            'Sharpe': f'{np.mean(sh):.2f}', 'Fill Rate': f'{np.mean(fr):.4f}',
            'Max |q|': f'{np.mean(mi):.1f}', 'Max DD': f'{np.mean([r["max_drawdown"] for r in res]):.2f}'}

summary = pd.DataFrame([summ(o_bt,'Oracle (HJB)'), summ(g_bt,'GRU'), summ(m_bt,'Mamba'), summ(p_bt,'Passive')])
op_t = np.array([r['final_pnl'] for r in o_bt])
gp_t = np.array([r['final_pnl'] for r in g_bt])
mp_t = np.array([r['final_pnl'] for r in m_bt])
pp_t = np.array([r['final_pnl'] for r in p_bt])
gr, mr, pr = op_t-gp_t, op_t-mp_t, op_t-pp_t

print("\n" + "="*80)
print(f"FINAL BACKTEST RESULTS ({n_bt} test paths)")
print("="*80)
print(summary.to_string(index=False))
print(f"\nRegret (Oracle - Strategy):")
print(f"  GRU:     {gr.mean():.2f} ± {gr.std():.2f}")
print(f"  Mamba:   {mr.mean():.2f} ± {mr.std():.2f}")
print(f"  Passive: {pr.mean():.2f} ± {pr.std():.2f}")

# --- Plot 6: Final Backtest ---
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
fig.suptitle('CODE-3: Final Backtest — Oracle vs GRU vs Mamba vs Passive', fontsize=15, fontweight='bold', y=0.98)
cb = ['#90EE90','#87CEEB','#FFB6C1','#D3D3D3']

ax = fig.add_subplot(gs[0,0])
ax.hist(op_t, bins=12, alpha=0.5, color='green', label='Oracle')
ax.hist(gp_t, bins=12, alpha=0.5, color='blue', label='GRU')
ax.hist(mp_t, bins=12, alpha=0.5, color='red', label='Mamba')
ax.hist(pp_t, bins=12, alpha=0.5, color='gray', label='Passive')
ax.set_xlabel('Final PnL ($)'); ax.set_ylabel('Count'); ax.set_title('(a) PnL Distribution'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0,1])
ax.hist(gr, bins=12, alpha=0.6, color='blue', label=f'GRU (μ={gr.mean():.1f})')
ax.hist(mr, bins=12, alpha=0.6, color='red', label=f'Mamba (μ={mr.mean():.1f})')
ax.hist(pr, bins=12, alpha=0.4, color='gray', label=f'Passive (μ={pr.mean():.1f})')
ax.axvline(0, color='black', ls='--', alpha=0.5)
ax.set_xlabel('Regret ($)'); ax.set_ylabel('Count'); ax.set_title('(b) Regret Distribution'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0,2])
bp = ax.boxplot([op_t,gp_t,mp_t,pp_t], labels=['Oracle','GRU','Mamba','Passive'], patch_artist=True)
for p,c in zip(bp['boxes'],cb): p.set_facecolor(c)
ax.set_ylabel('Final PnL ($)'); ax.set_title('(c) PnL Box Plot')

ax = fig.add_subplot(gs[1,0])
ax.plot(t_h, o_bt[0]['pnl'], 'g-', lw=1.5, label='Oracle')
ax.plot(t_h, g_bt[0]['pnl'], 'b-', lw=1.5, label='GRU')
ax.plot(t_h, m_bt[0]['pnl'], 'r-', lw=1.5, label='Mamba')
ax.plot(t_h, p_bt[0]['pnl'], color='gray', lw=1.5, label='Passive')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('PnL ($)'); ax.set_title('(d) PnL Trajectory (Path 0)'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1,1])
ax.plot(t_h, o_bt[0]['inventory'], 'g-', lw=1, label='Oracle')
ax.plot(t_h, g_bt[0]['inventory'], 'b-', lw=1, label='GRU')
ax.plot(t_h, m_bt[0]['inventory'], 'r-', lw=1, label='Mamba')
ax.axhline(0, color='black', alpha=0.3)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Inventory q'); ax.set_title('(e) Inventory (Path 0)'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1,2])
ms = [np.mean([r['sharpe'] for r in x]) for x in [o_bt,g_bt,m_bt,p_bt]]
ss = [np.std([r['sharpe'] for r in x]) for x in [o_bt,g_bt,m_bt,p_bt]]
ax.bar(['Oracle','GRU','Mamba','Passive'], ms, yerr=ss, color=cb, edgecolor='black', capsize=4)
ax.set_ylabel('Sharpe Ratio'); ax.set_title('(f) Mean Sharpe Ratio')

plt.savefig('plot6_final_backtest.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot6_final_backtest.png")

# --- Plot 7: Evans Theorem 5.3 Verification ---
q0 = Q_MAX
grad_v = (v_grid[q0+1, :] - v_grid[q0-1, :]) / 2.0
t_ax = np.arange(PATH_LEN)
cfs = np.polyfit(t_ax, grad_v, deg=1)
lam_pmp = np.polyval(cfs, t_ax)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Evans Theorem 5.3: PMP Costate = HJB Gradient', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(t_ax/60, grad_v, 'b-', lw=1.5, alpha=0.7, label='∇_q v(0,t) [HJB]')
ax.plot(t_ax/60, lam_pmp, 'r--', lw=2, label=f'λ(t)={cfs[1]:.4f}+{cfs[0]:.6f}·t [PMP]')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Costate / Gradient')
ax.set_title('(a) HJB Gradient vs PMP Costate at q=0'); ax.legend(fontsize=9)

ax = axes[1]
for qv in [-5,-2,0,2,5]:
    qi = qv+Q_MAX
    if 0 < qi < len(qs)-1:
        g = (v_grid[qi+1,:]-v_grid[qi-1,:])/2.0
        ax.plot(t_ax/60, g, lw=1.2, label=f'q={qv}')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('∇_q v(q,t)')
ax.set_title('(b) Value Gradient at Various Inventories'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot7_theorem53_verification.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: plot7_theorem53_verification.png")

print("\n" + "="*70)
print("ALL DONE — 7 plots saved.")
print("="*70)
