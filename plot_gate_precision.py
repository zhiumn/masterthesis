import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.Sharpturncorridorhybrid import SharpTurnCorridorHybrid

def get_gate_crossings(model_path, seed=42, is_hybrid=False):
    env_class = SharpTurnCorridorHybrid if is_hybrid else SharpTurnCorridor
    env = env_class(gui=False, num_lookahead_gates=2, obs=ObservationType.KIN, act=ActionType.RPM)
    model = PPO.load(model_path)
    obs, info = env.reset(seed=seed)
    
    traj = []
    crossings = [] # List of (gate_idx, local_y, local_z)
    
    gates = []
    for i in range(len(env.gate_poses)):
        gates.append({'pos': env.gate_poses[i].copy(), 'quat': env.gate_quats[i]})
        
    passed_gates = 0
    prev_p_G_0 = [None] * len(gates)
    
    while True:
        pos = env._getDroneStateVector(0)[0:3]
        traj.append(pos.copy())
        
        for i, g in enumerate(gates):
            gate_pos = g['pos']
            gate_quat = g['quat']
            R_gate = Rotation.from_quat(gate_quat).as_matrix()
            p_G = R_gate.T @ (pos - gate_pos)
            
            if prev_p_G_0[i] is not None and prev_p_G_0[i] <= 0 and p_G[0] > 0:
                # Interpolate
                t = (0 - prev_p_G_0[i]) / (p_G[0] - prev_p_G_0[i] + 1e-6)
                cross_y = prev_p_G_y[i] + t * (p_G[1] - prev_p_G_y[i])
                cross_z = prev_p_G_z[i] + t * (p_G[2] - prev_p_G_z[i])
                crossings.append((i, cross_y, cross_z))
                
            prev_p_G_0[i] = p_G[0]
            if 'prev_p_G_y' not in locals():
                prev_p_G_y = [None] * len(gates)
                prev_p_G_z = [None] * len(gates)
            prev_p_G_y[i] = p_G[1]
            prev_p_G_z[i] = p_G[2]
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

    env.close()
    return np.array(traj), gates, crossings

methods = {
    'A0': {'path': 'models/sharp_ablation_A0_seed0/best_model.zip', 'color': '#999999', 'label': 'A0 (Static)', 'is_hybrid': False, 'ls': '--'},
    'Hybrid': {'path': 'models/sharp_hybrid_seed0/best_model.zip', 'color': '#E67E22', 'label': 'PBRS-Hybrid', 'is_hybrid': True, 'ls': '-.'},
    'Fixed': {'path': 'models/sharp_ablation_A7-S_seed0/best_model.zip', 'color': '#2ECC71', 'label': 'Additive-Fixed', 'is_hybrid': False, 'ls': '-'},
    'Step': {'path': 'models/sharp_ablation_P5_seed0/best_model.zip', 'color': '#9B59B6', 'label': 'Additive-Step', 'is_hybrid': False, 'ls': '-'}
}

# Run for multiple seeds
test_seeds = list(range(400, 430))

os.makedirs('trajectory_plots', exist_ok=True)

for seed in test_seeds:
    print(f"\nProcessing seed {seed}...")
    data = {}
    for name, m in methods.items():
        print(f"Running {name}...")
        traj, gates, crossings = get_gate_crossings(m['path'], seed=seed, is_hybrid=m['is_hybrid'])
        data[name] = {'traj': traj, 'gates': gates, 'crossings': crossings}

    # Plotting
    fig = plt.figure(figsize=(16, 8))

    # Subplot 1: Zoomed Top-Down Trajectory
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(f"Zoomed-in Top-Down Trajectory (Seed {seed})", fontsize=16)
    ax1.set_xlabel("X-axis (m)", fontsize=12)
    ax1.set_ylabel("Y-axis (m)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    all_gates = data['Step']['gates']
    # Draw gates top-down
    for i, gate in enumerate(all_gates):
        gx, gy, gz = gate['pos']
        yaw = Rotation.from_quat(gate['quat']).as_euler('xyz')[2]
        
        hw = 0.5
        ht = 0.1
        corners = np.array([[ht, hw], [-ht, hw], [-ht, -hw], [ht, -hw]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners_world = (R @ corners.T).T + np.array([gx, gy])
        
        poly = patches.Polygon(corners_world, closed=True, facecolor='cornflowerblue', edgecolor='darkblue', linewidth=2, alpha=0.3)
        ax1.add_patch(poly)
        ax1.text(gx, gy, f"Gate {i+1}", fontsize=12, ha='center', va='center', color='darkblue')

    for name, d in data.items():
        traj = d['traj']
        color = methods[name]['color']
        ax1.plot(traj[:, 0], traj[:, 1], color=color, label=methods[name]['label'], linewidth=3, alpha=0.8, linestyle=methods[name]['ls'])

    # Centerline (Ideal Path)
    gate_centers = np.array([g['pos'][:2] for g in all_gates])
    ax1.plot(gate_centers[:, 0], gate_centers[:, 1], 'k--', label='Ideal Centerline', alpha=0.5, linewidth=2)

    # Zoom in on Gate 2 and Gate 3 (usually where the sharp turn happens)
    if len(all_gates) >= 3:
        g1 = all_gates[0]['pos']
        g3 = all_gates[2]['pos']
        ax1.set_xlim(min(g1[0], g3[0]) - 1.5, max(g1[0], g3[0]) + 1.5)
        ax1.set_ylim(min(g1[1], g3[1]) - 1.5, max(g1[1], g3[1]) + 1.5)

    ax1.legend()

    # Subplot 2: Gate Crossing Precision (Strike Zone) for Gate 2
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f"Gate 2 Crossing Precision (Seed {seed})", fontsize=16)
    ax2.set_xlabel("Lateral Deviation (m)", fontsize=12)
    ax2.set_ylabel("Vertical Deviation (m)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Draw Gate Frame
    gate_w = 1.0
    gate_h = 1.0
    rect = patches.Rectangle((-gate_w/2, -gate_h/2), gate_w, gate_h, linewidth=3, edgecolor='darkblue', facecolor='aliceblue', alpha=0.5)
    ax2.add_patch(rect)
    # Draw center crosshair
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlim(-gate_w/2 - 0.2, gate_w/2 + 0.2)
    ax2.set_ylim(-gate_h/2 - 0.2, gate_h/2 + 0.2)

    target_gate_idx = 1 # Gate 2 (0-indexed)
    for name, d in data.items():
        crossings = d['crossings']
        color = methods[name]['color']
        for idx, cy, cz in crossings:
            if idx == target_gate_idx:
                ax2.scatter(cy, cz, color=color, s=200, edgecolors='black', label=methods[name]['label'], zorder=5)
                # Draw line to center to highlight error
                ax2.plot([0, cy], [0, cz], color=color, linestyle=':', linewidth=2)

    ax2.legend()

    plt.tight_layout()
    save_path = f'trajectory_plots/gate_precision_analysis_seed{seed}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

print("All seeds processed!")
