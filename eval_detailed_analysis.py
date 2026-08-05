import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from stable_baselines3 import PPO
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

# 定義不同場景對應的模型與環境
SCENE_CONFIGS = {
    'standard': {
        'name': 'Standard Corridor',
        'models': {
            'Base Only': 'ablation_A0_seed{seed}',
            'Additive-Fixed': 'ablation_A7-S_seed{seed}',
            'Additive-Step': 'ablation_P5_seed{seed}',
            'PBRS': 'baseline_hybrid_seed{seed}',
        }
    },
    'sharp': {
        'name': 'Sharp Turn',
        'models': {
            'Base Only': 'sharp_ablation_A0_seed{seed}',
            'Additive-Fixed': 'sharp_ablation_A7-S_seed{seed}',
            'Additive-Step': 'sharp_ablation_P5_seed{seed}',
            'PBRS': 'sharp_hybrid_seed{seed}',
        }
    },
    'racing': {
        'name': 'Racing Track',
        'models': {
            'Base Only': 'racing_ablation_A0_seed{seed}',
            'Additive-Fixed': 'racing_track_static_seed{seed}',
            'Additive-Step': 'racing_ablation_P5_seed{seed}',
            'PBRS': 'racing_hybrid_seed{seed}',
        }
    }
}

def load_env(scene, method, lookahead=2):
    """根據場景與方法載入對應的環境。"""
    is_pbrs = method == 'PBRS'
    
    if scene == 'standard':
        if is_pbrs:
            from gym_pybullet_drones.envs.InfiniteGateAviaryhybrid import InfiniteGateAviaryHybrid
            return InfiniteGateAviaryHybrid(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
        else:
            from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
            return InfiniteGateAviary(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
            
    elif scene == 'sharp':
        if is_pbrs:
            from gym_pybullet_drones.envs.SharpTurnCorridorPBRS import SharpTurnCorridorPBRS
            return SharpTurnCorridorPBRS(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
        else:
            from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
            return SharpTurnCorridor(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
            
    elif scene == 'racing':
        if is_pbrs:
            from gym_pybullet_drones.envs.RacingTrackPBRS import RacingTrackPBRS
            return RacingTrackPBRS(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
        else:
            from gym_pybullet_drones.envs.RacingTrack import RacingTrack
            return RacingTrack(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
            
    raise ValueError(f"Unknown scene: {scene}")

def get_termination_reason(env, truncated, pos, vel, ang_vel, rpy, current_target_idx, gates_passed, forced_miss_termination=False):
    """推斷深入的失敗原因"""
    speed = np.linalg.norm(vel)
    ang_speed = np.linalg.norm(ang_vel)
    
    # 1. 撞機分析 (區分一般撞機與轉向過度失控)
    is_collision = hasattr(env, '_checkCollision') and env._checkCollision()
    if is_collision:
        if ang_speed > 8.0:  # 若撞擊瞬間角速度極大，代表轉向過度/失控
            return "Collision (Oversteering Spin)"
        else:
            return "Collision (Crash)"
            
    # 2. 賽道偏移分析 (漏門或距離目標太遠)
    misses = getattr(env, 'consecutive_misses', 0)
    if forced_miss_termination or misses > 0:
        # 計算終止瞬間的診斷量 (theta_err)
        R_drone = R.from_euler('xyz', rpy).as_matrix()
        drone_dir = R_drone @ np.array([1.0, 0.0, 0.0])
        
        gate_quats = getattr(env, 'gate_quats', [])
        if len(gate_quats) > current_target_idx:
            gate_quat = gate_quats[current_target_idx]
            R_gate = R.from_quat(gate_quat).as_matrix()
            gate_normal = R_gate @ np.array([1.0, 0.0, 0.0])
            
            cos_theta = np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))
            
            # 使用 2x2 診斷矩陣 (此處採用 1.0 作為高低速分水嶺，並保留 Out of Bounds)
            if speed > 1.0:
                if theta_deg < 45:
                    return "Out of Bounds (Normal Flight)"
                else:
                    return "Undershoot Turn (轉向不足)"
            else:
                if theta_deg < 45:
                    return "Stall (速度不足)"
                else:
                    return "Overshoot Turn (轉向過度)"
        else:
            return "Missed Gates (Unknown)"
        
    gate_poses = getattr(env, 'gate_poses', [])
    if len(gate_poses) > current_target_idx:
        dist = np.linalg.norm(pos - gate_poses[current_target_idx])
        if dist > 8.0:
            return "Out of Bounds (Deviated)"
            
    # 3. 超時分析 (區分成功、速度不足、卡死)
    if truncated:
        if speed < 0.5:
            return "Stuck in Local Minima"
        # 依照不同場景給予不同標準，此處以 50 門作為基礎標準
        elif gates_passed < 50:
            return "Insufficient Speed (Timeout)"
        else:
            return "Success (Reached Max Time)"
            
    return "Unknown / Success"

def draw_gate(ax, pos, quat, half_width=0.5):
    """在 3D 圖上繪製實體門框"""
    rot = R.from_quat(quat).as_matrix()
    
    # 定義門的四個角落 (相對於中心)
    corners_local = np.array([
        [0, -half_width, -half_width],
        [0, half_width, -half_width],
        [0, half_width, half_width],
        [0, -half_width, half_width],
        [0, -half_width, -half_width] # 閉合
    ])
    
    # 旋轉並平移
    corners_world = (rot @ corners_local.T).T + pos
    
    ax.plot(corners_world[:, 0], corners_world[:, 1], corners_world[:, 2], color='black', linewidth=1.5, alpha=0.8)
    
    # 畫法線向量 (指示前進方向)
    normal = rot @ np.array([1.0, 0.0, 0.0])
    ax.quiver(pos[0], pos[1], pos[2], normal[0], normal[1], normal[2], 
              length=1.0, color='red', alpha=0.5, arrow_length_ratio=0.3)

def summarize_failure_modes_seed_level(df, output_path):
    # 固定所有可能的終止原因，避免未發生的類別被 groupby 消失
    all_reasons = [
        "Success (Reached Max Time)",
        "Collision (Crash)",
        "Collision (Oversteering Spin)",
        "Out of Bounds (Normal Flight)",
        "Out of Bounds (Deviated)",
        "Undershoot Turn (轉向不足)",
        "Overshoot Turn (轉向過度)",
        "Stall (速度不足)",
        "Stuck in Local Minima",
        "Insufficient Speed (Timeout)",
        "Missed Gates (Unknown)",
        "Unknown / Success"
    ]

    # 每個 Scene / Method / Train Seed 應有的總 episode 數
    totals = (
        df.groupby(["Scene", "Method", "Train Seed"])
          .size()
          .reset_index(name="Total Episodes")
    )

    # 建立完整索引：每個 seed 都要有所有 failure mode
    full_index = pd.MultiIndex.from_product(
        [
            totals["Scene"].unique(),
            totals["Method"].unique(),
            totals["Train Seed"].unique(),
            all_reasons
        ],
        names=["Scene", "Method", "Train Seed", "Termination Reason"]
    )

    # 實際計數
    counts = (
        df.groupby(["Scene", "Method", "Train Seed", "Termination Reason"])
          .size()
          .reindex(full_index, fill_value=0)
          .reset_index(name="Count")
    )

    # 只保留實際存在的 Scene / Method / Train Seed 組合
    counts = counts.merge(
        totals,
        on=["Scene", "Method", "Train Seed"],
        how="inner"
    )

    counts["Proportion"] = counts["Count"] / counts["Total Episodes"]
    counts["Percentage"] = counts["Proportion"] * 100

    counts.to_csv(
        f"{output_path}/failure_mode_seed_level_percentages.csv",
        index=False
    )

    return counts

def summarize_failure_modes_across_seeds(seed_summary, output_path):
    final_summary = (
        seed_summary
        .groupby(['Scene', 'Method', 'Termination Reason'])
        .agg(
            Mean_Percentage=('Percentage', 'mean'),
            SD_Percentage=('Percentage', 'std'),
            Num_Seeds=('Train Seed', 'nunique')
        )
        .reset_index()
    )

    # 如果有些方法成功完成某些種子，某些種子完全沒發生這個錯誤，
    # 則標準差可能會有 NaN，此處填補為 0
    final_summary['SD_Percentage'] = final_summary['SD_Percentage'].fillna(0)

    final_summary['SEM_Percentage'] = (
        final_summary['SD_Percentage'] / np.sqrt(final_summary['Num_Seeds'])
    )

    final_summary.to_csv(
        f"{output_path}/failure_mode_mean_sd_sem_across_seeds.csv",
        index=False
    )

    return final_summary

def run_analysis(scene, episodes=50, seed_start=150000, num_train_seeds=10):
    print(f"=== 開始執行 {scene} 場景的 seed-level 失敗分析與 3D 軌跡追蹤 ===")
    config = SCENE_CONFIGS[scene]
    
    results = []
    trajectories = {method: [] for method in config['models'].keys()}
    gate_info = None # 儲存一個賽道的門框配置用於繪圖
    
    # 顏色配置
    colors = {
        'Base Only': '#34495E',
        'Additive-Fixed': '#F5A623',
        'Additive-Step': '#7ED321',
        'PBRS': '#4A90E2'
    }
    
    output_dir = f"detailed_analysis_{scene}"
    os.makedirs(output_dir, exist_ok=True)
    
    for method, pattern in config['models'].items():
        print(f"\n[{method}]")
        
        for train_seed in range(num_train_seeds):
            # 尋找該方法的模型
            model_dir = pattern.format(seed=train_seed)
            model_path = f"models/{model_dir}/best_model.zip"
            
            if not os.path.exists(model_path):
                print(f"  警告: 找不到模型 {model_path}，跳過。")
                continue
                
            print(f"  載入模型: {model_path}")
            model = PPO.load(model_path)
            
            # 嘗試取得 lookahead (簡單處理，預設為 2)
            try:
                obs_shape = model.policy.observation_space.shape[-1]
                lookahead = 2 if obs_shape == 84 else (1 if obs_shape == 78 else 3)
            except:
                lookahead = 2
                
            env = load_env(scene, method, lookahead)
            
            for ep in range(episodes):
                test_seed = seed_start + ep
                obs, info = env.reset(seed=test_seed)
                
                traj = []
                
                # 如果是第一次測試，記錄一下環境的門框分佈 (因為同一個 test_seed 生成的門是一樣的)
                if ep == 0 and train_seed == 0 and gate_info is None and hasattr(env, 'gate_poses'):
                    gate_info = {
                        'poses': [p.copy() for p in getattr(env, 'gate_poses', [])],
                        'quats': [tuple(q) for q in getattr(env, 'gate_quats', [])]
                    }
                    
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    if len(action.shape) == 1: action = np.array([action])
                    
                    # 紀錄當前絕對座標
                    state = env._getDroneStateVector(0)
                    pos = state[0:3].copy()
                    vel = state[10:13].copy()
                    ang_vel = state[13:16].copy()
                    rpy = state[7:10].copy()
                    traj.append(pos)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # 嚴格模式: 如果有設定嚴格，漏門也算 terminated
                    forced_miss_termination = False
                    if hasattr(env, 'consecutive_misses') and env.consecutive_misses > 0:
                        terminated = True
                        forced_miss_termination = True
                        
                    if terminated or truncated:
                        final_speed = np.linalg.norm(vel)
                        target_idx = getattr(env, 'current_target_idx', 0)
                        gates_passed = info.get('gates_passed', 0)
                        reason = get_termination_reason(
                            env, truncated, pos, vel, ang_vel, rpy, target_idx, gates_passed,
                            forced_miss_termination=forced_miss_termination
                        )
                        
                        results.append({
                            'Scene': scene,
                            'Method': method,
                            'Train Seed': train_seed,
                            'Episode': ep,
                            'Test Seed': test_seed,
                            'Termination Reason': reason,
                            'Final Speed': final_speed,
                            'Gates Passed': gates_passed
                        })
                        
                        # 只記錄每個方法的第一局作為代表軌跡繪製
                        if ep == 0 and train_seed == 0:
                            trajectories[method] = traj
                            
                        break
                        
            env.close()

    if not results:
        print("沒有收集到任何資料，請確認模型路徑是否正確。")
        return
        
    # --- 1. 輸出 CSV 與 失敗原因統計 ---
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/failure_analysis_seed_level_raw.csv", index=False)
    
    # 計算 seed-level 統計
    seed_summary = summarize_failure_modes_seed_level(df, output_dir)
    summarize_failure_modes_across_seeds(seed_summary, output_dir)
    
    print(f"完成！原始與統計資料已存至: {output_dir}")
    
    # 統計圓餅圖
    methods = df['Method'].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5))
    if len(methods) == 1: axes = [axes]
    
    fig.suptitle(f'Failure Mode Distribution ({config["name"]})', fontsize=16, fontweight='bold', y=1.05)
    
    for ax, method in zip(axes, methods):
        sub_df = df[df['Method'] == method]
        counts = sub_df['Termination Reason'].value_counts()
        
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
               colors=plt.cm.Pastel1.colors)
        ax.set_title(method)
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/failure_distribution_pie.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 2. 繪製 2D 投影軌跡圖 (含 3D 視覺門框) ---
    if gate_info is not None and len(trajectories) > 0:
        fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
        fig.suptitle(f'Drone Trajectory Map ({config["name"]} - Seed {seed_start})', fontsize=18, fontweight='bold')
        
        def draw_gate_2d(ax, pos, quat, plane='XY'):
            rot = R.from_quat(quat).as_matrix()
            w, h = 0.5, 0.5
            d = 0.1 # 深度
            
            front = np.array([[d, w, h], [d, -w, h], [d, -w, -h], [d, w, -h]])
            back = np.array([[-d, w, h], [-d, -w, h], [-d, -w, -h], [-d, w, -h]])
            
            iw, ih = 0.35, 0.35
            front_in = np.array([[d, iw, ih], [d, -iw, ih], [d, -iw, -ih], [d, iw, -ih]])
            back_in = np.array([[-d, iw, ih], [-d, -iw, ih], [-d, -iw, -ih], [-d, iw, -ih]])
            
            def to_world(pts):
                return (rot @ pts.T).T + pos
                
            fw, bw = to_world(front), to_world(back)
            fiw, biw = to_world(front_in), to_world(back_in)
            
            if plane == 'XY': idx0, idx1 = 0, 1
            else: idx0, idx1 = 0, 2 # XZ
            
            def draw_face(face_out, face_in, facecolor, edgecolor, alpha):
                pts = np.vstack([
                    face_out[:, [idx0, idx1]],
                    face_out[0:1, [idx0, idx1]],
                    face_in[::-1, [idx0, idx1]],
                    face_in[-1:, [idx0, idx1]]
                ])
                poly = plt.Polygon(pts, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
                ax.add_patch(poly)
                
            draw_face(bw, biw, '#749ed4', '#3c6ea6', 0.5)
            for i in range(4):
                ax.plot([fw[i, idx0], bw[i, idx0]], [fw[i, idx1], bw[i, idx1]], color='#3c6ea6', lw=1, alpha=0.5)
            draw_face(fw, fiw, '#8ab3e6', '#3c6ea6', 0.8)

        # 繪製 XY (Top-down)
        for method, traj in trajectories.items():
            if len(traj) == 0: continue
            traj = np.array(traj)
            ax_xy.plot(traj[:, 0], traj[:, 1], label=method, color=colors.get(method, 'gray'), linewidth=2)
            # 打點
            ax_xy.scatter(traj[0, 0], traj[0, 1], color='red', s=40, zorder=5) # Start
            ax_xy.scatter(traj[-1, 0], traj[-1, 1], color='green', s=40, zorder=5) # End
            
        for pos, quat in zip(gate_info['poses'], gate_info['quats']):
            draw_gate_2d(ax_xy, pos, quat, plane='XY')
            
        ax_xy.set_ylabel('Distance travel in Y-axis (m)', fontsize=14)
        ax_xy.grid(True, linestyle='--', alpha=0.7)
        ax_xy.legend(loc='best')
        ax_xy.set_aspect('equal', 'datalim')

        # 繪製 XZ (Side view)
        for method, traj in trajectories.items():
            if len(traj) == 0: continue
            traj = np.array(traj)
            ax_xz.plot(traj[:, 0], traj[:, 2], label=method, color=colors.get(method, 'gray'), linewidth=2)
            # 打點
            ax_xz.scatter(traj[0, 0], traj[0, 2], color='red', s=40, zorder=5)
            ax_xz.scatter(traj[-1, 0], traj[-1, 2], color='green', s=40, zorder=5)
            
        for pos, quat in zip(gate_info['poses'], gate_info['quats']):
            draw_gate_2d(ax_xz, pos, quat, plane='XZ')
            
        ax_xz.set_xlabel('Distance travel in X-axis (m)', fontsize=14)
        ax_xz.set_ylabel('Distance travel in Z-axis (m)', fontsize=14)
        ax_xz.grid(True, linestyle='--', alpha=0.7)
        ax_xz.set_aspect('equal', 'datalim')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/trajectory_2d_projections.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"完成！圖表與數據已存至目錄: {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, choices=['standard', 'sharp', 'racing'])
    parser.add_argument('--episodes', type=int, default=50, help='要測試的 Seed 數量 (預設 50)')
    parser.add_argument('--seeds', type=int, default=150000, help='起始 Seed (預設 150000)')
    args = parser.parse_args()
    
    run_analysis(args.scene, args.episodes, args.seeds)
