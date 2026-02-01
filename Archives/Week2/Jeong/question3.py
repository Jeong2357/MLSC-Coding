import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------
# Model: Evans Example 1
# x' = k * a * x,   a in [0,1]
# payoff J = ∫_0^T (1-a(t)) x(t) dt
# -------------------------

class PolicyNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t_norm, x_norm):
        # inputs: (N,), (N,)
        inp = torch.stack([t_norm, x_norm], dim=-1)  # (N,2)
        a = torch.sigmoid(self.net(inp)).squeeze(-1)  # (N,), in (0,1)
        return a


def rk4_step(x, a, dt, k):
    # x: scalar tensor, a: scalar tensor
    def f(x_, a_):
        return k * a_ * x_

    k1 = f(x, a)
    k2 = f(x + 0.5 * dt * k1, a)
    k3 = f(x + 0.5 * dt * k2, a)
    k4 = f(x + dt * k3, a)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rollout(policy, x0, T, dt, k, device):
    n_steps = int(T / dt)
    ts = torch.linspace(0.0, T, n_steps + 1, device=device)

    # store for plotting
    xs = torch.zeros(n_steps + 1, device=device)
    as_ = torch.zeros(n_steps + 1, device=device)

    x = torch.tensor(float(x0), device=device)
    xs[0] = x

    # normalized features
    # time in [0,1], x feature log-scale for stability
    x0_t = torch.tensor(float(x0), device=device)
    eps = 1e-8

    payoff = torch.tensor(0.0, device=device)

    for i in range(n_steps):
        t = ts[i]
        t_norm = t / T
        x_norm = torch.log((x + eps) / (x0_t + eps))  # scalar

        a = policy(t_norm.unsqueeze(0), x_norm.unsqueeze(0)).squeeze(0)  # scalar in (0,1)
        as_[i] = a

        # running profit (consumption): (1-a)*x
        r = (1.0 - a) * x
        payoff = payoff + r * dt

        x = rk4_step(x, a, dt, k)
        xs[i+1] = x

    # last action for completeness
    as_[-1] = as_[-2]
    return ts, xs, as_, payoff


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = PolicyNet(hidden=args.hidden).to(device)
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    best_payoff = -1e18
    best_state = None

    for it in range(args.iters):
        ts, xs, as_, payoff = rollout(policy, args.x0, args.T, args.dt, args.k, device)

        loss = -payoff  # maximize payoff

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()

        p = float(payoff.detach().cpu().item())
        if p > best_payoff:
            best_payoff = p
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

        if it % args.print_every == 0:
            print(f"iter {it:5d} | payoff={p:12.6f}")

    if best_state is not None:
        policy.load_state_dict(best_state)

    # final eval + plot
    with torch.no_grad():
        ts, xs, as_, payoff = rollout(policy, args.x0, args.T, args.dt, args.k, device)

    ts_np = ts.cpu().numpy()
    xs_np = xs.cpu().numpy()
    as_np = as_.cpu().numpy()
    payoff_val = float(payoff.cpu().item())

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(ts_np, xs_np, lw=2)
    axes[0].set_ylabel("Wealth x(t)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Learned investment strategy | payoff={payoff_val:.4f}")

    axes[1].plot(ts_np, as_np, lw=2)
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("a(t) (reinvest fraction)")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot to: {args.out}")
    print(f"Final payoff (total profit): {payoff_val:.6f}")
    print(f"Note: Although we did not assume bang-bang, Evans notes mention the optimal control for this model is bang-bang (with a switching time) [file:1].")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=1.0, help="growth rate k")
    ap.add_argument("--T", type=float, default=3.0, help="horizon T")
    ap.add_argument("--dt", type=float, default=0.01, help="time step")
    ap.add_argument("--x0", type=float, default=1.0, help="initial wealth")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--out", type=str, default="investment_strategy.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
