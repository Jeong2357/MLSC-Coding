# XLE Commodity Trading with Regret Minimization - Final Results

## Overview
This project implements a complete 5-step pipeline inspired by **Evans' Example 4.6.2
(Commodity Trading)** to train a neural network ensemble for energy sector trading.

The key metric is **Regret**: the gap between our strategy and the theoretically perfect
PMP Oracle that has perfect foresight.

$$\text{Regret} = \text{Oracle P&L} - \text{NN P&L} = \$33.56$$

## Step-by-Step Results

### Step 1: Extract Physics
- **Asset**: XLE (Energy Sector ETF), 1999-2023 training, 2024-2025 evaluation
- **Cyclical model**: log(p) = drift*t + intercept + A*sin(omega*t + phase) + noise
- Annual drift: 6.50%, Cycle amplitude: 21.4% (log scale)
- Cycle period: 8.8 years, Noise volatility: 24.3%

### Step 2: The Multiverse (Training Data)
- 200 overlapping windows from real XLE data (stride=30 days), each weighted 2x
- 200 synthetic paths with randomized physics parameters
- Total: 600 training paths of 300 days each
- **Key improvement**: Heavy emphasis on real data ensures the model learns actual XLE dynamics

### Step 3: The Oracle (Pontryagin Maximum Principle)
- Evans' PMP: costate lambda(t) = lambda_0 + h*t, bang-bang u(t) = M*sign(lambda - p)
- **Key insight**: We use the oracle's **switching function** sigma(t) = lambda_0 + h*t - p(t)
  as the training target. This captures the oracle's trading INTENT before inventory clipping,
  providing a clean, continuous signal for the NN to learn.
- Oracle mean PnL on training paths: $1655

### Step 4: The Student (GRU Ensemble)
- **Architecture**: 2-layer GRU (96 hidden) + MLP head with dropout
- **Features** (window=30 days): normalized price, returns, volatility,
  mean-reversion signal, momentum
- **Training target**: tanh(sigma / price_scale) — smoothed oracle intent
- **Ensemble**: 3 models trained with different seeds, predictions averaged
- 80 epochs with cosine annealing warm restarts
- Ensemble directional accuracy: 85.8%

### Step 5: The Final Exam

| Strategy | Total P&L | vs Buy-Hold |
|----------|-----------|-------------|
| **PMP Oracle** (Theoretical Best) | **$1099.73** | +$1084.63 |
| **NN Ensemble** (Our Model) | **$1066.16** | +$1051.07 |
| **Buy & Hold** (Benchmark) | **$15.10** | — |

### Performance Metrics
- **Regret** (Oracle - NN): $33.56
- **NN captures 96.9% of Oracle performance**
- **NN outperforms Buy-Hold by**: $1051.07 (6963x)
- **In-sample** (last 2yr training): NN=$2134.72 = 97.3% of Oracle=$2194.05
- Trading activity: 146 buys, 174 sells

## Key Insights

1. **Oracle = ceiling**: The PMP Oracle ($1099.73) uses perfect future knowledge
   via Evans' Pontryagin Maximum Principle. No causal model can exceed it.

2. **Switching function > clipped actions**: Training on sigma(t) = lambda_0 + h*t - p(t)
   (the oracle's intent) instead of inventory-clipped actions dramatically improved
   directional accuracy from ~25% to ~86%.

3. **Real data emphasis**: Doubling the weight of real XLE sliding windows ensures the
   model learns the actual energy sector dynamics, not just synthetic patterns.

4. **Ensemble averaging**: Combining 3 independently trained GRUs smooths
   predictions and reduces variance on unseen data.

5. **The cost of uncertainty**: Regret = $33.56 quantifies the fundamental price
   of not knowing the future — the irreducible gap between any causal trader and an oracle.

## Figures
1. **fig1_physics_fit.png** — XLE with fitted geometric cycle (1999-2023)
2. **fig2_synthetic_paths.png** — Real chunks + synthetic paths, return distributions
3. **fig3_oracle_example.png** — Oracle: price, switching function, actions, inventory
4. **fig4_training_loss.png** — Training convergence
5. **fig5_evaluation_backtest.png** — Full eval: price, cumulative P&L, actions, inventory
6. **fig6_actions_comparison.png** — Oracle vs NN: actions and inventory side-by-side
7. **fig7_performance_summary.png** — Performance comparison bar chart

## Configuration
- Holding cost: 0.0005/day | Max rate: 5.0/day | Max inventory: 80.0
- GRU: 96 hidden, 2 layers | Window: 30 days | Ensemble: 3 models
- Training: 80 epochs, 600 paths, 162000 samples total
