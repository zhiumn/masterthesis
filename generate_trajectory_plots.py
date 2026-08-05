import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
from gym_pybullet_drones.envs.RacingTrackAblation import RacingTrackAblation
from gym_pybullet_drones.envs.SharpTurnZigZag import SharpTurnZigZag
from gym_pybullet_drones.envs.Sharpturncorridorhybrid import SharpTurnCorridorHybrid
from gym_pybullet_drones.envs.Racingtrackhybrid import RacingTrackHybrid

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

def run_episode(env_class, model_path, seed, is_hybrid=False):
    env = env_class(gui=False, num_lookahead_gates=2, obs=ObservationType.KIN, act=ActionType.RPM)
    
    try:
        model = PPO.load(model_path)
    except:
        print(f"Failed to load {model_path}")
        return None, None, None

    obs, info = env.reset(seed=seed)
    traj = []
    
    gates = []
    
    def add_unique_gates():
        for i in range(len(env.gate_poses)):
            pos = env.gate_poses[i]
            quat = env.gate_quats[i]
            is_new = True
            for g in gates:
                if np.linalg.norm(g['pos'] - pos) < 0.1:
                    is_new = False
                    break
            if is_new:
                gates.append({'pos': pos.copy(), 'quat': quat})

    add_unique_gates()

    passed_gates = 0
    crossings = []
    prev_p_G_0 = [None] * 50  # allow up to 50 gates
    prev_p_G_y = [None] * 50
    prev_p_G_z = [None] * 50
    
    from scipy.spatial.transform import Rotation
    
    while True:
        pos = env._getDroneStateVector(0)[0:3]
        traj.append(pos.copy())
        
        for i, g in enumerate(gates):
            if i >= 50: break
            gate_pos = g['pos']
            gate_quat = g['quat']
            R_gate = Rotation.from_quat(gate_quat).as_matrix()
            p_G = R_gate.T @ (pos - gate_pos)
            
            if prev_p_G_0[i] is not None and prev_p_G_0[i] <= 0 and p_G[0] > 0:
                t = (0 - prev_p_G_0[i]) / (p_G[0] - prev_p_G_0[i] + 1e-6)
                cross_y = prev_p_G_y[i] + t * (p_G[1] - prev_p_G_y[i])
                cross_z = prev_p_G_z[i] + t * (p_G[2] - prev_p_G_z[i])
                crossings.append((i, cross_y, cross_z))
                
            prev_p_G_0[i] = p_G[0]
            prev_p_G_y[i] = p_G[1]
            prev_p_G_z[i] = p_G[2]
            
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        add_unique_gates()
        
        if terminated or truncated:
            passed_gates = info.get("gates_passed", 0)
            break

    traj = np.array(traj)
    env.close()
    return traj, gates, passed_gates, crossings

methods = {
    'A0': {'color': '#999999', 'label': 'A0 (Static)', 'ls': '--', 'marker': '^'},
    'Hybrid': {'color': '#E67E22', 'label': 'PBRS-Hybrid', 'ls': '-.', 'marker': 'v'},
    'Fixed': {'color': '#2ECC71', 'label': 'Additive-Fixed', 'ls': '-', 'marker': 'o'},
    'Step': {'color': '#9B59B6', 'label': 'Additive-Step', 'ls': '-', 'marker': 's'}
}

scenarios = {
    'sharp_turn': {
        'env': SharpTurnCorridor,
        'models': {
            'A0': 'models/sharp_ablation_A0_seed0/best_model.zip',
            'Hybrid': 'models/sharp_hybrid_seed0/best_model.zip',
            'Fixed': 'models/sharp_ablation_A7-S_seed0/best_model.zip',
            'Step': 'models/sharp_ablation_P5_seed0/best_model.zip'
        }
    },
    'racing_track': {
        'env': RacingTrackAblation,
        'models': {
            'A0': 'models/racing_ablation_A0_seed0/best_model.zip',
            'Hybrid': 'models/racing_hybrid_seed0/best_model.zip',
            'Fixed': 'models/racing_track_static_seed0/best_model.zip',
            'Step': 'models/racing_ablation_P5_seed0/best_model.zip'
        }
    },
    'zigzag': {
        'env': SharpTurnZigZag,
        'models': {
            'A0': 'models/sharp_ablation_A0_seed0/best_model.zip',
            'Hybrid': 'models/sharp_hybrid_seed0/best_model.zip',
            'Fixed': 'models/sharp_ablation_A7-S_seed0/best_model.zip',
            'Step': 'models/sharp_ablation_P5_seed0/best_model.zip'
        }
    }
}

os.makedirs('trajectory_plots', exist_ok=True)

test_seeds = [10, 42, 99, 123, 456]

for scene_name, scene_info in scenarios.items():
    env_cls = scene_info['env']
    
    for seed in test_seeds:
        print(f"Running {scene_name} seed {seed}...")
        
        trajectories = {}
        all_gates = None
        
        for method_key, model_path in scene_info['models'].items():
            is_hybrid = ('hybrid' in method_key.lower())
            
            # Note: For simplicity, we are passing the standard env_cls. If Hybrid requires the hybrid env class, we can swap it.
            # But the user mentioned PBRS-Hybrid just takes KIN observation since it's A1 or similar? Let's check. 
            # If it fails to load due to observation space, we will catch it. Actually PPO.load overrides env if not provided.
            # Let's use the standard env because they all use KIN state observation.
            
            actual_env_cls = env_cls
            if is_hybrid:
                if scene_name == 'sharp_turn':
                    actual_env_cls = SharpTurnCorridorHybrid
                elif scene_name == 'racing_track':
                    actual_env_cls = RacingTrackHybrid
                elif scene_name == 'zigzag':
                    actual_env_cls = SharpTurnZigZag # ZigZag has no hybrid yet, just use KIN
            
            traj, gates, passed, crossings = run_episode(actual_env_cls, model_path, seed, is_hybrid)
            if traj is not None:
                trajectories[method_key] = {'traj': traj, 'passed': passed, 'crossings': crossings}
                if all_gates is None:
                    all_gates = gates

        if all_gates is None or len(trajectories) == 0:
            continue
            
        # Plotting
        fig = plt.figure(figsize=(20, 6))
        ax_top = plt.subplot(1, 3, 1)
        ax_side = plt.subplot(1, 3, 2)
        ax_strike = plt.subplot(1, 3, 3)
        
        ax_top.set_title(f'Drone Trajectory Map - {scene_name.title()} (Seed {seed})', fontsize=14)
        ax_top.set_ylabel('Distance travel in Y-axis (m)', fontsize=12)
        ax_top.set_xlabel('Distance travel in X-axis (m)', fontsize=12)
        ax_top.grid(True, linestyle='--', alpha=0.6)
        
        ax_side.set_title('Side-View Trajectory', fontsize=14)
        ax_side.set_ylabel('Distance travel in Z-axis (m)', fontsize=12)
        ax_side.set_xlabel('Distance travel in X-axis (m)', fontsize=12)
        ax_side.grid(True, linestyle='--', alpha=0.6)
        
        ax_strike.set_title('Gate Crossing Precision (Strike Zone)', fontsize=14)
        ax_strike.set_xlabel('Lateral Deviation (m)', fontsize=12)
        ax_strike.set_ylabel('Vertical Deviation (m)', fontsize=12)
        ax_strike.grid(True, linestyle='--', alpha=0.6)
        
        gate_w = 1.0
        gate_h = 1.0
        rect = patches.Rectangle((-gate_w/2, -gate_h/2), gate_w, gate_h, linewidth=3, edgecolor='darkblue', facecolor='aliceblue', alpha=0.5)
        ax_strike.add_patch(rect)
        ax_strike.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_strike.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax_strike.set_xlim(-gate_w/2 - 0.2, gate_w/2 + 0.2)
        ax_strike.set_ylim(-gate_h/2 - 0.2, gate_h/2 + 0.2)

        # Plot gates
        gate_width = 1.0
        gate_height = 1.0
        gate_thickness = 0.2
        for gate in all_gates:
            gx, gy, gz = gate['pos']
            import pybullet as p
            yaw = p.getEulerFromQuaternion(gate['quat'])[2]
            
            # Top-Down gate projection (X, Y)
            # Create a rotated rectangle for the gate top-down
            hw = gate_width / 2.0
            ht = gate_thickness / 2.0
            
            # corners in local frame
            corners = np.array([
                [ht, hw], [-ht, hw], [-ht, -hw], [ht, -hw]
            ])
            # rotation matrix
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            # rotate and translate
            corners_world = (R @ corners.T).T + np.array([gx, gy])
            
            poly = patches.Polygon(corners_world, closed=True, 
                                   facecolor='cornflowerblue', edgecolor='darkblue', 
                                   linewidth=2, alpha=0.6, zorder=2)
            ax_top.add_patch(poly)
            
            # Side-View gate projection (X, Z)
            # Since gate might be angled, it projects to X as a thickness
            # We'll just draw it as a vertical rectangle based on thickness
            rect_side = patches.Rectangle((gx - ht, gz - gate_height/2), gate_thickness, gate_height, 
                                          facecolor='cornflowerblue', edgecolor='darkblue', 
                                          linewidth=2, alpha=0.6, zorder=2)
            ax_side.add_patch(rect_side)

        # Plot trajectories and strike zone
        for method_key, data in trajectories.items():
            traj = data['traj']
            crossings = data['crossings']
            style = methods[method_key]
            
            # Subsample for markers
            markevery = max(1, len(traj) // 10)
            
            ax_top.plot(traj[:, 0], traj[:, 1], color=style['color'], label=style['label'],
                        linestyle=style['ls'], marker=style['marker'], markevery=markevery, markersize=6, alpha=0.8)
            ax_side.plot(traj[:, 0], traj[:, 2], color=style['color'], label=style['label'],
                         linestyle=style['ls'], marker=style['marker'], markevery=markevery, markersize=6, alpha=0.8)
            
            # Strike Zone points
            for c in crossings:
                _, cy, cz = c
                ax_strike.scatter(cy, cz, color=style['color'], marker=style['marker'], s=100, edgecolors='black', alpha=0.7, zorder=5)
            
            # Mark end point
            end_marker = 'o' if data['passed'] > 0 and len(traj) > 100 else 'x'
            ax_top.scatter(traj[-1, 0], traj[-1, 1], color='black', marker=end_marker, s=80, zorder=5)
            ax_side.scatter(traj[-1, 0], traj[-1, 2], color='black', marker=end_marker, s=80, zorder=5)

        # Start point
        ax_top.scatter(0, 0, color='red', marker='o', s=100, label='Start', zorder=6)
        ax_side.scatter(0, 1.0, color='red', marker='o', s=100, zorder=6)
        
        ax_top.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        ax_strike.scatter([], [], color='white', label=' ') # dummy for alignment if needed
        
        plt.tight_layout()
        save_path = f"trajectory_plots/traj_{scene_name}_seed{seed}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

print("All trajectory plots generated successfully in trajectory_plots/ directory!")
