#!/usr/bin/env python3
"""
Pendulum Optimal Control with IQN (Implicit Quantile Networks) - PyTorch Implementation
Proper implementation with gradient-based learning and experience replay.

System: θ̈ + λθ̇ + ω²θ = α(t), where |α| ≤ 1
Goal: Minimize time to bring pendulum to rest
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================
LAMBDA = 0.5        # Damping coefficient
OMEGA_SQ = 4.0      # Natural frequency squared (ω²)
ALPHA_MAX = 1.0     # Control constraint |α| ≤ 1
DT = 0.05           # Time step for RK4 integration
MAX_STEPS = 200     # Maximum steps per episode

# ============================================================================
# PENDULUM ENVIRONMENT WITH RK4 INTEGRATION
# ============================================================================
class PendulumEnv:
    """Pendulum environment with Runge-Kutta 4th order integration"""

    def __init__(self, lambda_param=LAMBDA, omega_sq=OMEGA_SQ, dt=DT, max_steps=MAX_STEPS):
        self.lambda_param = lambda_param
        self.omega_sq = omega_sq
        self.dt = dt
        self.max_steps = max_steps
        self.state = None
        self.steps = 0

    def dynamics(self, state, alpha):
        """Pendulum dynamics: θ̈ + λθ̇ + ω²θ = α"""
        theta, theta_dot = state
        theta_ddot = alpha - self.lambda_param * theta_dot - self.omega_sq * theta
        return np.array([theta_dot, theta_ddot])

    def rk4_step(self, state, alpha):
        """Runge-Kutta 4th order integration step"""
        k1 = self.dynamics(state, alpha)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, alpha)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, alpha)
        k4 = self.dynamics(state + self.dt * k3, alpha)

        new_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

    def reset(self, theta_init=None, theta_dot_init=None):
        """Reset environment with initial conditions"""
        if theta_init is None:
            theta_init = np.random.uniform(-1.5, 1.5)
        if theta_dot_init is None:
            theta_dot_init = np.random.uniform(-1.0, 1.0)

        self.state = np.array([theta_init, theta_dot_init], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        """Execute one step: action ∈ {0, 1} → α ∈ {-1, +1}"""
        alpha = ALPHA_MAX if action == 1 else -ALPHA_MAX

        # RK4 integration
        self.state = self.rk4_step(self.state, alpha)
        self.steps += 1

        # Check terminal condition
        theta, theta_dot = self.state
        goal_reached = (abs(theta) < 0.05 and abs(theta_dot) < 0.05)
        done = goal_reached or self.steps >= self.max_steps

        # Improved reward shaping
        distance = np.sqrt(theta**2 + (theta_dot/3.0)**2)

        if goal_reached:
            reward = 200.0  # Large goal bonus
        else:
            # Dense reward: encourage moving toward goal, penalize time
            reward = -1.0 - 2.0 * distance

        return self.state.copy(), reward, done

# ============================================================================
# IQN NETWORK
# ============================================================================
class IQNNetwork(nn.Module):
    """
    Implicit Quantile Network
    Maps (state, quantile τ) → Q-value distribution
    """

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256, n_cos=64):
        super(IQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_cos = n_cos

        # State embedding network
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Quantile embedding network (cosine basis)
        self.quantile_net = nn.Sequential(
            nn.Linear(n_cos, hidden_dim),
            nn.ReLU()
        )

        # Final output layer
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, tau):
        """
        Forward pass
        Args:
            state: (batch_size, state_dim)
            tau: (batch_size, n_quantiles) or (batch_size * n_quantiles,)
        Returns:
            q_values: (batch_size, n_quantiles, action_dim)
        """
        batch_size = state.shape[0]

        # Handle tau shape
        if tau.dim() == 1:
            n_tau = tau.shape[0] // batch_size
            tau = tau.view(batch_size, n_tau)
        else:
            n_tau = tau.shape[1]

        # State embedding
        state_embed = self.state_net(state)  # (batch, hidden)
        state_embed = state_embed.unsqueeze(1).expand(-1, n_tau, -1)  # (batch, n_tau, hidden)

        # Quantile embedding using cosine basis
        # cos(i * π * τ) for i = 0, 1, ..., n_cos-1
        i_pi = torch.arange(0, self.n_cos, device=state.device).float() * np.pi
        cos_tau = torch.cos(tau.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0))  # (batch, n_tau, n_cos)

        quantile_embed = self.quantile_net(cos_tau)  # (batch, n_tau, hidden)

        # Element-wise product (Hadamard product)
        combined = state_embed * quantile_embed  # (batch, n_tau, hidden)

        # Output Q-values
        q_values = self.output_net(combined)  # (batch, n_tau, action_dim)

        return q_values

# ============================================================================
# REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=np.float32)

        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================================
# IQN AGENT
# ============================================================================
class IQNAgent:
    """Implicit Quantile Network Agent"""

    def __init__(self, env, hidden_dim=256, n_quantiles=32, kappa=1.0):
        self.env = env
        self.n_quantiles = n_quantiles
        self.kappa = kappa

        # Networks
        self.policy_net = IQNNetwork(state_dim=2, action_dim=2, hidden_dim=hidden_dim).to(device)
        self.target_net = IQNNetwork(state_dim=2, action_dim=2, hidden_dim=hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=50000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0

    def select_action(self, state, epsilon=None):
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return np.random.randint(2)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            tau = torch.rand(1, self.n_quantiles).to(device)
            q_values = self.policy_net(state_tensor, tau)  # (1, n_quantiles, 2)
            q_mean = q_values.mean(dim=1)  # (1, 2)
            action = q_mean.argmax(dim=1).item()

        return action

    def quantile_huber_loss(self, td_errors, tau):
        """
        Quantile Huber loss (QR-DQN loss)
        Args:
            td_errors: (batch, n_tau, n_tau')
            tau: (batch, n_tau, 1)
        """
        # Huber loss
        huber = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors.pow(2),
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )

        # Quantile regression
        quantile_weight = torch.abs(tau - (td_errors < 0).float())
        loss = (quantile_weight * huber).mean()

        return loss

    def train_step(self):
        """Single training step"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # Sample quantiles
        tau = torch.rand(self.batch_size, self.n_quantiles).to(device)
        tau_hat = torch.rand(self.batch_size, self.n_quantiles).to(device)

        # Current Q-values
        q_values = self.policy_net(state, tau)  # (batch, n_quantiles, 2)
        q_values = q_values.gather(2, action.unsqueeze(1).unsqueeze(2).expand(-1, self.n_quantiles, -1))
        q_values = q_values.squeeze(2)  # (batch, n_quantiles)

        # Target Q-values
        with torch.no_grad():
            # Double DQN: select action with policy net, evaluate with target net
            next_q_policy = self.policy_net(next_state, tau_hat)  # (batch, n_quantiles, 2)
            next_actions = next_q_policy.mean(dim=1).argmax(dim=1)  # (batch,)

            next_q_target = self.target_net(next_state, tau_hat)  # (batch, n_quantiles, 2)
            next_q_target = next_q_target.gather(2, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, self.n_quantiles, -1))
            next_q_target = next_q_target.squeeze(2)  # (batch, n_quantiles)

            target = reward.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * next_q_target

        # TD error
        # Expand for pairwise computation
        td_errors = target.unsqueeze(1) - q_values.unsqueeze(2)  # (batch, n_quantiles, n_quantiles)

        # Quantile Huber loss
        loss = self.quantile_huber_loss(td_errors, tau.unsqueeze(2))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train(self, n_episodes=2000, verbose=True):
        """Train the agent"""
        rewards_history = []
        steps_history = []
        losses_history = []

        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            train_steps = 0

            while True:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                # Store transition
                self.memory.push(state, action, reward, next_state, float(done))

                episode_reward += reward
                state = next_state

                # Train
                loss = self.train_step()
                episode_loss += loss
                train_steps += 1

                if done:
                    break

            rewards_history.append(episode_reward)
            steps_history.append(self.env.steps)
            losses_history.append(episode_loss / max(train_steps, 1))

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Logging
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                avg_loss = np.mean(losses_history[-100:])
                print(f"Episode {episode+1}/{n_episodes} | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Steps: {avg_steps:.1f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Buffer: {len(self.memory)}")

        return rewards_history, steps_history, losses_history

    def evaluate(self, theta_init=1.0, theta_dot_init=0.0, max_steps=200):
        """Evaluate trained policy (greedy)"""
        state = self.env.reset(theta_init, theta_dot_init)
        trajectory = [state.copy()]
        actions = []

        for _ in range(max_steps):
            action = self.select_action(state, epsilon=0.0)
            actions.append(action)
            state, reward, done = self.env.step(action)
            trajectory.append(state.copy())

            if done:
                break

        return np.array(trajectory), np.array(actions)

# ============================================================================
# ANALYTICAL SOLUTION
# ============================================================================
def analytical_bangbang_control(theta_0, theta_dot_0, lambda_param, omega_sq, dt, max_time=10):
    """Analytical bang-bang solution"""
    state = np.array([theta_0, theta_dot_0])
    trajectory = [state.copy()]
    actions = []
    time_points = [0]
    t = 0

    while t < max_time:
        theta, theta_dot = state
        switch_function = theta_dot + 2.0 * theta # The optimal switching curve when λ = 0 is θ̇ + 2.0 * θ in our case
 
        if switch_function > 0:
            alpha = -ALPHA_MAX
            action = 0
        else:
            alpha = ALPHA_MAX
            action = 1

        actions.append(action)

        def dynamics(s, a):
            th, th_dot = s
            th_ddot = a - lambda_param * th_dot - omega_sq * th
            return np.array([th_dot, th_ddot])

        k1 = dynamics(state, alpha)
        k2 = dynamics(state + 0.5 * dt * k1, alpha)
        k3 = dynamics(state + 0.5 * dt * k2, alpha)
        k4 = dynamics(state + dt * k3, alpha)

        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

        trajectory.append(state.copy())
        time_points.append(t)

        if abs(state[0]) < 0.05 and abs(state[1]) < 0.05:
            break

    return np.array(trajectory), np.array(actions), np.array(time_points)

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(trajectory_ml, actions_ml, trajectory_an, actions_an, 
                time_points_an, rewards_history, steps_history, losses_history,
                dt, filename='pendulum_iqn_torch_comparison.png'):
    """Create comprehensive comparison plots"""

    time_points_ml = np.arange(len(trajectory_ml)) * dt
    control_ml = np.where(actions_ml == 1, ALPHA_MAX, -ALPHA_MAX)
    control_an = np.where(actions_an == 1, ALPHA_MAX, -ALPHA_MAX)

    fig = plt.figure(figsize=(16, 12))

    # Training curves
    ax1 = plt.subplot(3, 3, 1)
    window = 50
    rewards_smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax1.plot(rewards_smooth, linewidth=2, color='#2E86DE')
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
    ax1.set_title('IQN Training (PyTorch)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    steps_smooth = np.convolve(steps_history, np.ones(window)/window, mode='valid')
    ax2.plot(steps_smooth, linewidth=2, color='#EE5A24')
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Steps to Goal', fontsize=11, fontweight='bold')
    ax2.set_title('Steps per Episode', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    losses_smooth = np.convolve(losses_history, np.ones(window)/window, mode='valid')
    ax3.plot(losses_smooth, linewidth=2, color='#9B59B6')
    ax3.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Quantile Huber Loss', fontsize=11, fontweight='bold')
    ax3.set_title('IQN Loss', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Phase portrait
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(trajectory_ml[:, 0], trajectory_ml[:, 1], 'b-', linewidth=2.5, 
             label='IQN (PyTorch)', alpha=0.8)
    ax4.plot(trajectory_an[:, 0], trajectory_an[:, 1], 'r--', linewidth=2.5, 
             label='Analytical', alpha=0.8)
    ax4.plot(trajectory_ml[0, 0], trajectory_ml[0, 1], 'go', markersize=10, label='Start', zorder=5)
    ax4.plot(0, 0, 'k*', markersize=15, label='Goal', zorder=5)
    ax4.set_xlabel('θ (angle)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('θ̇ (angular velocity)', fontsize=11, fontweight='bold')
    ax4.set_title('Phase Portrait', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='k', linewidth=0.5)
    ax4.axvline(0, color='k', linewidth=0.5)

    # Trajectories
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time_points_ml, trajectory_ml[:, 0], 'b-', linewidth=2.5, label='IQN', alpha=0.8)
    ax5.plot(time_points_an, trajectory_an[:, 0], 'r--', linewidth=2.5, label='Analytical', alpha=0.8)
    ax5.axhline(0, color='k', linewidth=1, linestyle=':', alpha=0.5)
    ax5.fill_between(time_points_ml, -0.05, 0.05, alpha=0.2, color='green')
    ax5.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('θ (angle)', fontsize=11, fontweight='bold')
    ax5.set_title('Angle Trajectory', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time_points_ml, trajectory_ml[:, 1], 'b-', linewidth=2.5, label='IQN', alpha=0.8)
    ax6.plot(time_points_an, trajectory_an[:, 1], 'r--', linewidth=2.5, label='Analytical', alpha=0.8)
    ax6.axhline(0, color='k', linewidth=1, linestyle=':', alpha=0.5)
    ax6.fill_between(time_points_ml, -0.05, 0.05, alpha=0.2, color='green')
    ax6.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('θ̇ (angular velocity)', fontsize=11, fontweight='bold')
    ax6.set_title('Angular Velocity', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Control signal
    ax7 = plt.subplot(3, 3, 7)
    time_control_ml = np.arange(len(control_ml)) * dt
    time_control_an = np.arange(len(control_an)) * dt
    ax7.step(time_control_ml, control_ml, 'b-', linewidth=2, label='IQN', where='post', alpha=0.8)
    ax7.step(time_control_an, control_an, 'r--', linewidth=2, label='Analytical', where='post', alpha=0.8)
    ax7.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Control α(t)', fontsize=11, fontweight='bold')
    ax7.set_title('Bang-Bang Control', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(-1.2, 1.2)
    ax7.axhline(0, color='k', linewidth=0.5)

    # Energy
    energy_ml = 0.5 * trajectory_ml[:, 1]**2 + 0.5 * OMEGA_SQ * trajectory_ml[:, 0]**2
    energy_an = 0.5 * trajectory_an[:, 1]**2 + 0.5 * OMEGA_SQ * trajectory_an[:, 0]**2

    ax8 = plt.subplot(3, 3, 8)
    ax8.semilogy(time_points_ml, energy_ml, 'b-', linewidth=2.5, label='IQN', alpha=0.8)
    ax8.semilogy(time_points_an, energy_an, 'r--', linewidth=2.5, label='Analytical', alpha=0.8)
    ax8.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Energy (log)', fontsize=11, fontweight='bold')
    ax8.set_title('Energy Dissipation', fontsize=12, fontweight='bold')
    ax8.legend(loc='best', fontsize=9)
    ax8.grid(True, alpha=0.3, which='both')

    # Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
PERFORMANCE SUMMARY
{'='*40}

IQN (PyTorch Implementation)
  • Quantile Huber loss
  • Double DQN
  • Experience replay
  • Target network

System: λ={LAMBDA}, ω²={OMEGA_SQ}
Initial: θ(0)={trajectory_ml[0,0]:.3f}
         θ̇(0)={trajectory_ml[0,1]:.3f}

IQN (Machine Learning):
  Time: {time_points_ml[-1]:.3f} s
  Switches: {np.sum(np.abs(np.diff(actions_ml)))}
  |θ|: {abs(trajectory_ml[-1,0]):.4f}
  |θ̇|: {abs(trajectory_ml[-1,1]):.4f}

Analytical:
  Time: {time_points_an[-1]:.3f} s
  Switches: {np.sum(np.abs(np.diff(actions_an)))}
  |θ|: {abs(trajectory_an[-1,0]):.4f}
  |θ̇|: {abs(trajectory_an[-1,1]):.4f}

Performance:
  {((time_points_ml[-1] - time_points_an[-1]) / time_points_an[-1] * 100):.1f}% slower
"""

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Pendulum Optimal Control: IQN (PyTorch) vs Analytical', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as '{filename}'")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("IQN (PYTORCH) FOR PENDULUM OPTIMAL CONTROL")
    print("="*70)
    print(f"System: θ̈ + {LAMBDA}θ̇ + {OMEGA_SQ}θ = α(t), |α| ≤ {ALPHA_MAX}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    env = PendulumEnv()
    agent = IQNAgent(env, hidden_dim=256, n_quantiles=32)

    print("Training IQN agent...")
    rewards_history, steps_history, losses_history = agent.train(n_episodes=2000, verbose=True)
    print("\n✓ Training completed!\n")

    print("="*70)
    print("EVALUATION")
    print("="*70)

    theta_0, theta_dot_0 = 1.0, 0.0
    print(f"Initial: θ(0) = {theta_0}, θ̇(0) = {theta_dot_0}\n")

    trajectory_ml, actions_ml = agent.evaluate(theta_0, theta_dot_0)
    time_ml = len(actions_ml) * env.dt

    print(f"IQN (PyTorch):")
    print(f"  Time: {time_ml:.3f} s")
    print(f"  Switches: {np.sum(np.abs(np.diff(actions_ml)))}")
    print(f"  Final: θ={trajectory_ml[-1, 0]:.4f}, θ̇={trajectory_ml[-1, 1]:.4f}\n")

    trajectory_an, actions_an, time_points_an = analytical_bangbang_control(
        theta_0, theta_dot_0, LAMBDA, OMEGA_SQ, env.dt
    )

    print(f"Analytical:")
    print(f"  Time: {time_points_an[-1]:.3f} s")
    print(f"  Switches: {np.sum(np.abs(np.diff(actions_an)))}")
    print(f"  Final: θ={trajectory_an[-1, 0]:.4f}, θ̇={trajectory_an[-1, 1]:.4f}\n")

    plot_results(trajectory_ml, actions_ml, trajectory_an, actions_an,
                time_points_an, rewards_history, steps_history, losses_history, env.dt)

    print("="*70)
    print("✓ Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
