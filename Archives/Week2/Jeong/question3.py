import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Problem Configuration ---
T = 10.0           # Terminal time
k = 0.5            # Growth rate of reinvestment
x0 = 1.0           # Initial output
dt = 0.01          # Time step size
steps = int(T / dt)

# --- 2. Neural Network Policy ---
class ControlPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, t, x):
        t_norm = t / T 
        state_input = torch.cat([t_norm, x], dim=1)
        return self.net(state_input)

# --- 3. Training Loop (Differentiable Physics) ---
def train_optimal_control():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ControlPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.005)
    
    epochs = 2000
    
    print(f"{'Epoch':^10} | {'Loss (Negative Payoff)':^25}")
    print("-" * 40)

    for epoch in range(epochs):
        current_x = torch.tensor([[x0]], device=device, dtype=torch.float32)
        total_payoff = 0.0
        
        # Simulation loop (Euler Integration)
        for step in range(steps):
            t_val = step * dt
            t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
            
            alpha = policy(t_tensor, current_x)
            
            # Payoff: consume (1-alpha) * x
            instant_reward = (1.0 - alpha) * current_x
            total_payoff += instant_reward * dt
            
            # Dynamics: dx = k * alpha * x * dt
            dx = k * alpha * current_x * dt
            current_x = current_x + dx
        
        loss = -total_payoff
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"{epoch:^10} | {loss.item():^25.4f}")

    print(f"{epochs:^10} | {loss.item():^25.4f}")
    return policy

# --- 4. Analytical Solution Calculation ---
def get_analytical_solution(times):
    # Theoretical switching time t* = T - 1/k
    # If T < 1/k, t* = 0 (always consume)
    t_star = max(0, T - 1.0/k)
    
    analytical_alphas = []
    analytical_xs = []
    analytical_payoff = 0.0
    
    curr_x = x0
    
    for t in times:
        # Optimal Control: Bang-Bang
        if t < t_star:
            alpha = 1.0
        else:
            alpha = 0.0
            
        analytical_alphas.append(alpha)
        analytical_xs.append(curr_x)
        
        # Calculate payoff integral
        analytical_payoff += (1 - alpha) * curr_x * dt
        
        # Dynamics update
        dx = k * alpha * curr_x * dt
        curr_x += dx
        
    return analytical_alphas, analytical_xs, analytical_payoff, t_star

# --- 5. Visualization & Comparison ---
def visualize_comparison(policy):
    policy.eval()
    times = np.linspace(0, T, steps)
    
    # 1. Get Neural Net Solution
    nn_alphas = []
    nn_xs = []
    nn_payoff = 0.0
    
    current_x = torch.tensor([[x0]], dtype=torch.float32)
    with torch.no_grad():
        for t_val in times:
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
            alpha = policy(t_tensor, current_x)
            
            val_alpha = alpha.item()
            val_x = current_x.item()
            
            nn_alphas.append(val_alpha)
            nn_xs.append(val_x)
            nn_payoff += (1 - val_alpha) * val_x * dt
            
            dx = k * alpha * current_x * dt
            current_x = current_x + dx

    # 2. Get Analytical Solution
    ana_alphas, ana_xs, ana_payoff, t_star = get_analytical_solution(times)

    # 3. Print Comparison to Terminal
    print("\n" + "="*50)
    print(f"{'METRIC':<25} | {'ANALYTICAL':<15} | {'NEURAL NET':<15}")
    print("-" * 60)
    print(f"{'Total Payoff':<25} | {ana_payoff:<15.4f} | {nn_payoff:<15.4f}")
    print(f"{'Switching Time (t*)':<25} | {t_star:<15.4f} | {'~'+str(round(t_star, 1))+' (Learned)'}")
    print(f"{'Final Output x(T)':<25} | {ana_xs[-1]:<15.4f} | {nn_xs[-1]:<15.4f}")
    print("="*50 + "\n")

    # 4. Plot Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Control Strategy (Alpha)
    ax1.set_title(f'Control Strategy Comparison (Switching at t*={t_star:.2f})')
    ax1.plot(times, ana_alphas, 'k--', linewidth=2, label='Analytical Optimal (Bang-Bang)')
    ax1.plot(times, nn_alphas, 'r-', linewidth=2, alpha=0.8, label='Neural Network Policy')
    ax1.set_ylabel('Alpha (Reinvestment Rate)')
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.2)
    
    # Add annotation for switching time
    ax1.axvline(x=t_star, color='blue', linestyle=':', alpha=0.6)
    ax1.text(t_star+0.2, 0.5, f't* = {t_star}', color='blue')

    # Plot 2: State Trajectory (Output x)
    ax2.set_title('Production Output Trajectory x(t)')
    ax2.plot(times, ana_xs, 'k--', linewidth=2, label='Analytical x(t)')
    ax2.plot(times, nn_xs, 'b-', linewidth=2, alpha=0.8, label='Neural Network x(t)')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Output x(t)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question3.png', dpi=300)

if __name__ == "__main__":
    trained_policy = train_optimal_control()
    visualize_comparison(trained_policy)