import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class RocketNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze()

def main():
    model = RocketNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    dt = 0.02
    max_horizon = 250

    for episode in range(5000):
        state = (torch.rand(2) * 6) - 3
        state.requires_grad_(True)
        loss = 0
        
        for step in range(max_horizon):
            a_raw = model(state)
            a = torch.tanh(a_raw * 5)
            
            q, v = state
            v_new = v + dt * a
            q_new = q + dt * v_new
            state = torch.stack([q_new, v_new])
            
            loss += dt
            
            if state.detach().norm() < 0.08:
                break
        
        loss += state.norm()**2 * 20
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    test_state = torch.tensor([3.0, 2.0])
    path_q, path_v = [test_state[0].item()], [test_state[1].item()]
    
    with torch.no_grad():
        for t in range(1000):
            raw_out = model(test_state).item()
            if test_state.norm() < 0.05:
                alpha = 0.0
            else:
                alpha = 1.0 if raw_out >= 0 else -1.0
            
            q, v = test_state
            v_new = v + dt * alpha
            q_new = q + dt * v_new
            test_state = torch.tensor([q_new, v_new])
            path_q.append(q_new.item())
            path_v.append(v_new.item())
            if test_state.norm() < 0.01:
                break

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    q_arr, v_arr = np.array(path_q), np.array(path_v)
    pad = 1.0
    grid_q, grid_v = np.meshgrid(np.linspace(q_arr.min()-pad, q_arr.max()+pad, 100), 
                                 np.linspace(v_arr.min()-pad, v_arr.max()+pad, 100))
    grid_tensor = torch.tensor(np.c_[grid_q.ravel(), grid_v.ravel()], dtype=torch.float32)
    with torch.no_grad():
        grid_out = model(grid_tensor).reshape(grid_q.shape)
    
    v_plot = np.linspace(-3, 3, 200)
    q_switch_pos = -0.5 * v_plot * np.abs(v_plot)
    q_switch_neg = 0.5 * v_plot * np.abs(v_plot)
    
    plt.contourf(grid_q, grid_v, grid_out, levels=25, cmap='RdBu', alpha=0.5)
    plt.plot(q_switch_pos, v_plot, 'g--', linewidth=1.5, alpha=0.8, label='Analytical Switch')
    plt.plot(q_switch_neg, v_plot, 'g--', linewidth=1.5, alpha=0.8)
    plt.plot(path_q, path_v, 'k-', linewidth=2, label='Learned Trajectory')
    plt.plot(path_q[0], path_v[0], 'go', markersize=8, label='Start')
    plt.plot(0, 0, 'rx', markersize=10, markeredgewidth=2, label='Target')
    plt.title('Optimal Phase Portrait')
    plt.xlabel('Position (q)')
    plt.ylabel('Velocity (v)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(path_q, label='Position (q)', color='blue', linewidth=2)
    plt.plot(path_v, label='Velocity (v)', color='orange', linewidth=2, linestyle='--')
    plt.axhline(0, color='black', linewidth=1, alpha=0.7)
    plt.title('State Convergence')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
