"""
Pendulum Bang-Bang Control using Q-Learning
Simulation with Runge-Kutta 4 method

Problem: Minimize stopping time of a damped pendulum
Equation: θ̈ + λθ̇ + ω²θ = α(t)
Constraint: |α| ≤ 1 (bang-bang control: α ∈ {-1, +1})
Goal: Bring θ and θ̇ to zero in minimum time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import pickle

class PendulumEnvironment:
    """Pendulum environment with RK4 simulation"""
    
    def __init__(self, lambda_damping=0.5, omega_sq=4.0, dt=0.02, max_steps=1000):
        """
        Initialize pendulum environment
        
        Parameters:
        - lambda_damping: damping coefficient (λ)
        - omega_sq: ω² in the equation
        - dt: time step for RK4 integration
        - max_steps: maximum number of steps per episode
        """
        self.lambda_damping = lambda_damping
        self.omega_sq = omega_sq
        self.dt = dt
        self.max_steps = max_steps
        
        # Control actions: bang-bang
        self.actions = [-1.0, 1.0]  # α ∈ {-1, +1}
        self.num_actions = len(self.actions)
        
        # State bounds for normalization
        self.theta_max = np.pi
        self.theta_dot_max = 10.0
        
        # Tolerance for "at rest"
        self.rest_tolerance = 0.05
        
        self.reset()
    
    def dynamics(self, state, control):
        """
        Compute state derivatives
        state = [θ, θ̇]
        returns [θ̇, θ̈]
        """
        theta, theta_dot = state
        theta_ddot = -self.lambda_damping * theta_dot - self.omega_sq * theta + control
        return np.array([theta_dot, theta_ddot])
    
    def rk4_step(self, state, control):
        """
        Runge-Kutta 4th order integration step
        """
        dt = self.dt
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.dynamics(state + dt * k3, control)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def reset(self, initial_state=None):
        """Reset environment to initial state"""
        if initial_state is None:
            # Random initial condition
            self.state = np.array([
                np.random.uniform(-self.theta_max, self.theta_max),
                np.random.uniform(-self.theta_dot_max, self.theta_dot_max)
            ])
        else:
            self.state = np.array(initial_state, dtype=np.float64)
        
        self.steps = 0
        self.total_time = 0.0
        return self.discretize_state(self.state)
    
    def discretize_state(self, state):
        """
        Discretize continuous state for Q-table
        """
        theta, theta_dot = state
        
        # Discretize angle into bins
        theta_bins = 50
        theta_idx = int(np.clip((theta / self.theta_max + 1) * theta_bins / 2, 0, theta_bins - 1))
        
        # Discretize angular velocity into bins
        theta_dot_bins = 50
        theta_dot_idx = int(np.clip((theta_dot / self.theta_dot_max + 1) * theta_dot_bins / 2, 
                                     0, theta_dot_bins - 1))
        
        return (theta_idx, theta_dot_idx)
    
    def is_at_rest(self, state):
        """Check if pendulum is at rest"""
        return (np.abs(state[0]) < self.rest_tolerance and 
                np.abs(state[1]) < self.rest_tolerance)
    
    def step(self, action_idx):
        """
        Take a step in the environment
        
        Returns:
        - next_state_discrete: discretized next state
        - reward: reward signal
        - done: whether episode is finished
        - info: additional information
        """
        control = self.actions[action_idx]
        
        # Simulate one step with RK4
        self.state = self.rk4_step(self.state, control)
        self.steps += 1
        self.total_time += self.dt
        
        # Check if at rest (goal reached)
        done = False
        reward = -1.0  # Penalty for each time step (encourage faster convergence)
        
        if self.is_at_rest(self.state):
            reward = 100.0  # Large reward for reaching goal
            done = True
        elif self.steps >= self.max_steps:
            reward = -50.0  # Penalty for taking too long
            done = True
        
        next_state_discrete = self.discretize_state(self.state)
        info = {'continuous_state': self.state.copy(), 'time': self.total_time}
        
        return next_state_discrete, reward, done, info


class QLearningAgent:
    """Q-Learning agent for bang-bang control"""
    
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent
        
        Parameters:
        - num_actions: number of possible actions
        - learning_rate: α in Q-learning update
        - discount_factor: γ for future rewards
        - epsilon: exploration rate
        - epsilon_decay: rate of epsilon decay
        - epsilon_min: minimum epsilon value
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(num_actions))
    
    def get_action(self, state, training=True):
        """
        Get action using epsilon-greedy policy
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update rule
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.Q[state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.Q[state][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_qlearning(env, agent, num_episodes=5000, print_interval=500):
    """
    Train Q-Learning agent
    """
    episode_rewards = []
    episode_times = []
    success_count = 0
    
    print("Starting Q-Learning Training...")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        # Reset environment with random initial state
        state = env.reset()
        total_reward = 0
        
        done = False
        while not done:
            # Select and perform action
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_times.append(info['time'])
        
        if reward > 0:  # Successfully reached goal
            success_count += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_time = np.mean([t for t, r in zip(episode_times[-print_interval:], 
                                                   episode_rewards[-print_interval:]) if r > 0])
            success_rate = success_count / print_interval
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Time: {avg_time:.3f}s | "
                  f"Success Rate: {success_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            success_count = 0
    
    print("-" * 60)
    print("Training Complete!")
    
    return episode_rewards, episode_times


def test_policy(env, agent, initial_states, max_time=20.0):
    """
    Test learned policy on specific initial conditions
    """
    results = []
    
    for init_state in initial_states:
        state_continuous = init_state
        state_discrete = env.reset(initial_state=init_state)
        
        trajectory = [state_continuous.copy()]
        controls = []
        times = [0.0]
        
        done = False
        t = 0.0
        
        while not done and t < max_time:
            # Get action from learned policy (no exploration)
            action = agent.get_action(state_discrete, training=False)
            control = env.actions[action]
            controls.append(control)
            
            # Step environment
            state_discrete, reward, done, info = env.step(action)
            state_continuous = info['continuous_state']
            t = info['time']
            
            trajectory.append(state_continuous.copy())
            times.append(t)
            
            if done and reward > 0:
                break
        
        results.append({
            'initial_state': init_state,
            'trajectory': np.array(trajectory),
            'controls': np.array(controls),
            'times': np.array(times),
            'stopping_time': t,
            'success': done and reward > 0
        })
    
    return results


def plot_results(results, save_path='pendulum_qlearning_results.png'):
    """
    Plot trajectories and controls for multiple initial conditions
    """
    num_cases = len(results)
    fig = plt.figure(figsize=(16, 4 * num_cases))
    gs = GridSpec(num_cases, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, result in enumerate(results):
        init = result['initial_state']
        traj = result['trajectory']
        controls = result['controls']
        times = result['times']
        stop_time = result['stopping_time']
        success = result['success']
        
        # Plot angle θ(t)
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(times, traj[:, 0], 'b-', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Angle θ (rad)', fontsize=11)
        ax1.set_title(f'Case {i+1}: θ(0)={init[0]:.2f}, θ̇(0)={init[1]:.2f}\n' + 
                     f'Stopping Time: {stop_time:.3f}s {"✓" if success else "✗"}', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot angular velocity θ̇(t)
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(times, traj[:, 1], 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Angular Velocity θ̇ (rad/s)', fontsize=11)
        ax2.set_title('Angular Velocity vs Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot control α(t)
        ax3 = fig.add_subplot(gs[i, 2])
        control_times = times[:-1]
        ax3.step(control_times, controls, 'g-', linewidth=2, where='post')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Control α(t)', fontsize=11)
        ax3.set_title('Bang-Bang Control (Q-Learning)', fontsize=12, fontweight='bold')
        ax3.set_ylim([-1.5, 1.5])
        ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Pendulum Bang-Bang Control using Q-Learning with RK4 Simulation', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {save_path}")
    
    return fig


def plot_training_progress(episode_rewards, episode_times, save_path='training_progress.png'):
    """
    Plot training progress
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    window = 100
    if len(episode_rewards) >= window:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed_rewards, 'b-', linewidth=2)
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average Reward (100-episode window)', fontsize=12)
        ax1.set_title('Training Progress: Rewards', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot successful episode times
    successful_times = [t for t, r in zip(episode_times, episode_rewards) if r > 0]
    if successful_times:
        ax2.plot(successful_times, 'g.', alpha=0.5, markersize=3)
        if len(successful_times) >= window:
            smoothed_times = np.convolve(successful_times, np.ones(window)/window, mode='valid')
            ax2.plot(range(len(smoothed_times)), smoothed_times, 'r-', linewidth=2, 
                    label='100-episode moving average')
        ax2.set_xlabel('Successful Episode Number', fontsize=12)
        ax2.set_ylabel('Stopping Time (s)', fontsize=12)
        ax2.set_title('Stopping Time for Successful Episodes', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training progress saved to: {save_path}")
    
    return fig


def main():
    """
    Main function to train and test Q-Learning agent
    """
    print("="*60)
    print("PENDULUM BANG-BANG CONTROL WITH Q-LEARNING")
    print("="*60)
    
    # Initialize environment
    env = PendulumEnvironment(
        lambda_damping=0.5,
        omega_sq=4.0,
        dt=0.02,
        max_steps=1000
    )
    
    # Initialize Q-Learning agent
    agent = QLearningAgent(
        num_actions=env.num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agent
    episode_rewards, episode_times = train_qlearning(
        env, agent, 
        num_episodes=5000,
        print_interval=500
    )
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_times)
    
    # Test on specific initial conditions
    print("\n" + "="*60)
    print("TESTING LEARNED POLICY")
    print("="*60)
    
    test_initial_states = [
        np.array([1.0, 0.0]),      # Start at angle π/3, no velocity
        np.array([0.5, 2.0]),      # Start at angle π/6, positive velocity
        np.array([-1.0, -1.5]),    # Start at negative angle, negative velocity
        np.array([1.5, -2.0]),     # Start at large angle, negative velocity
    ]
    
    results = test_policy(env, agent, test_initial_states)
    
    # Print results
    print("\nTest Results:")
    print("-" * 60)
    for i, result in enumerate(results):
        init = result['initial_state']
        print(f"Case {i+1}: θ(0)={init[0]:.2f}, θ̇(0)={init[1]:.2f}")
        print(f"  Stopping time: {result['stopping_time']:.3f}s")
        print(f"  Success: {result['success']}")
        print()
    
    # Plot results
    plot_results(results)
    
    print("="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
