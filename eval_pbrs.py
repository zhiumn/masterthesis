"""
PBRS vs Additive 評估腳本

使用方式：

  # 評估 Sharp-Turn 場景的 PBRS 模型
  python eval_pbrs.py --scene sharp_turn --episodes 50

  # 評估 Racing Track 場景的 PBRS 模型
  python eval_pbrs.py --scene racing_track --episodes 50

  # GUI 展示
  python eval_pbrs.py --scene sharp_turn --gui

  # 同時評估 PBRS 和 Additive (用於論文對比)
  python eval_pbrs.py --scene sharp_turn --compare_additive

輸出：
  - Fig1: Bar chart (Gates, Jerk, Distance)
  - Fig2: Box plot (分佈比較)
  - Fig3: Scatter (Performance vs Smoothness)
  - Summary Table (CSV + 圖片)
  - Pairwise Statistics (Cohen's d, Welch t-test)
"""

import argparse
import time
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from scipy import stats


def extract_lookahead(exp_name, default_val=2):
    match = re.search(r"_L(\d+)", exp_name)
    if match:
        return int(match.group(1))
    return default_val


def eval_model_worker(worker_args):
    """多進程評估 worker。"""
    model_path, exp_name, test_seeds, lookahead, scene = worker_args
    import numpy as np
    from stable_baselines3 import PPO

    model = PPO.load(model_path)

    # 動態推導 lookahead
    try:
        obs_shape = model.policy.observation_space.shape[-1]
        if obs_shape == 84:
            lookahead = 2
        elif obs_shape == 78:
            lookahead = 1
        elif obs_shape == 90:
            lookahead = 3
    except:
        pass

    # 根據場景和方法類型載入環境
    if scene == 'sharp_turn':
        if 'pbrs' in exp_name.lower():
            from gym_pybullet_drones.envs.SharpTurnCorridorPBRS import SharpTurnCorridorPBRS
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = SharpTurnCorridorPBRS(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM)
        else:
            from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = SharpTurnCorridor(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM)
    else:  # racing_track
        if 'pbrs' in exp_name.lower():
            from gym_pybullet_drones.envs.RacingTrackPBRS import RacingTrackPBRS
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = RacingTrackPBRS(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM)
        else:
            from gym_pybullet_drones.envs.RacingTrack import RacingTrack
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = RacingTrack(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM)

    seed_gates, seed_jerk, seed_dist, seed_laps = [], [], [], []

    for test_seed in test_seeds:
        obs, info = eval_env.reset(seed=test_seed)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            if len(action.shape) == 1:
                action = np.array([action])
            obs, reward, terminated, truncated, info = eval_env.step(action)

            # 嚴格連續穿梭模式
            if eval_env.consecutive_misses > 0:
                terminated = True

            if terminated or truncated:
                gp = info.get('gates_passed', 0)
                jk = info.get('avg_jerk', 0.0)
                ds = info.get('avg_pass_dist', 0.0)
                laps = info.get('laps_completed', 0)

                seed_gates.append(gp)
                seed_jerk.append(jk)
                seed_dist.append(ds)
                seed_laps.append(laps)
                break

    eval_env.close()
    return model_path, seed_gates, seed_jerk, seed_dist, seed_laps


def run_evaluation(args):
    """執行評估並產出圖表。"""

    scene = args.scene
    print(f"=== PBRS 評估模式 ({scene}) ===")

    # 搜尋模型
    experiments = {}
    
    # PBRS 模型
    scene_prefix = scene.split('_')[0]  # 'sharp_turn' -> 'sharp', 'racing_track' -> 'racing'
    pbrs_pattern = f"models/*{scene_prefix}*pbrs*_seed*/best_model.zip"
    for model_path in glob.glob(pbrs_pattern):
        exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
        if args.filter and not any(f.lower() in exp_name.lower() for f in args.filter):
            continue
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(model_path)

    # 如果需要比較 Additive
    if args.compare_additive:
        if scene == 'sharp_turn':
            additive_patterns = [
                "models/sharp_turn_dynamic_seed*/best_model.zip",
                "models/sharp_turn_static_seed*/best_model.zip",
                "models/sharp_ablation_*_seed*/best_model.zip",
            ]
        else:
            additive_patterns = [
                "models/racing_track_*_seed*/best_model.zip",
                "models/racing_ablation_*_seed*/best_model.zip",
            ]
        
        for pattern in additive_patterns:
            for model_path in glob.glob(pattern):
                if 'pbrs' in model_path.lower():
                    continue
                exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
                if exp_name not in experiments:
                    experiments[exp_name] = []
                experiments[exp_name].append(model_path)

    if not experiments:
        print(f"未找到任何 {scene} 模型!")
        return

    print(f"找到 {len(experiments)} 種實驗配置:")
    for name, paths in experiments.items():
        print(f"  - {name}: {len(paths)} seeds")

    # 評估
    collected_data = {}
    test_seeds = [150000 + i for i in range(args.episodes)]

    import concurrent.futures

    worker_tasks = []
    for exp_name, model_paths in experiments.items():
        lookahead = extract_lookahead(exp_name, default_val=2)
        for p in model_paths:
            worker_tasks.append((p, exp_name, test_seeds, lookahead, scene))

    results_map = {}
    if worker_tasks:
        # 為了避免 OOM (Out Of Memory) 與 分頁檔太小的錯誤 (WinError 1455)
        # Windows 的 PyTorch multiprocessing 非常吃記憶體，這裡將最大核心數限制為 6
        max_workers = min(max(1, (os.cpu_count() or 4) - 2), len(worker_tasks), 6)
        print(f"\n🚀 平行評估 {len(worker_tasks)} 個模型，使用 {max_workers} 核心...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, result in enumerate(executor.map(eval_model_worker, worker_tasks)):
                model_path, seed_g, seed_j, seed_d, seed_l = result
                results_map[model_path] = (seed_g, seed_j, seed_d, seed_l)
                print(f"  [{idx+1}/{len(worker_tasks)}] 完成: {os.path.basename(os.path.dirname(model_path))}")

    # 彙整數據
    for exp_name, model_paths in experiments.items():
        all_gates, all_jerk, all_dist, all_laps = [], [], [], []
        seed_stats = []

        for p in model_paths:
            if p not in results_map:
                continue
            seed_g, seed_j, seed_d, seed_l = results_map[p]

            seed_stats.append({
                'gates': np.mean([x for x in seed_g if np.isfinite(x)]),
                'jerk': np.mean([x for x in seed_j if np.isfinite(x)]),
                'dist': np.mean([x for x in seed_d if np.isfinite(x)]),
                'laps': np.mean([x for x in seed_l if np.isfinite(x)]),
            })

            all_gates.extend(seed_g)
            all_jerk.extend(seed_j)
            all_dist.extend(seed_d)
            all_laps.extend(seed_l)

        collected_data[exp_name] = {
            'gates': np.array(all_gates),
            'jerk': np.array(all_jerk),
            'dist': np.array(all_dist),
            'laps': np.array(all_laps),
            'seeds': seed_stats,
        }

    # 標籤映射
    def get_short_name(exp_name):
        if 'pbrs-full' in exp_name.lower():
            return 'PBRS-Full'
        if 'pbrs-dist' in exp_name.lower():
            return 'PBRS-Dist'
        if 'pbrs-cent' in exp_name.lower():
            return 'PBRS-Cent'
        if 'pbrs-head' in exp_name.lower():
            return 'PBRS-Head'
        if 'dynamic' in exp_name.lower():
            return 'Additive-Exp'
        if 'static' in exp_name.lower() or 'fixed' in exp_name.lower():
            return 'Additive-Fixed'
        if 'p4' in exp_name.lower():
            return 'Additive-Lin'
        if 'p5' in exp_name.lower():
            return 'Additive-Step'
        if 'p6' in exp_name.lower():
            return 'Additive-Sig'
        if '_a0' in exp_name.lower():
            return 'Base Only'
        if '_a1' in exp_name.lower():
            return '+Centering'
        if '_a2' in exp_name.lower():
            return '+Heading'
        if '_a3' in exp_name.lower():
            return '+Smooth'
        # 替換過時的 Full- 標籤
        name_map = {
            'Full-Exp': 'Additive-Exp',
            'Full-Lin': 'Additive-Lin',
            'Full-Step': 'Additive-Step',
            'Full-Sig': 'Additive-Sig'
        }
        for k, v in name_map.items():
            if k in exp_name:
                return exp_name.replace(k, v)
        return exp_name

    short_mapping = {name: get_short_name(name) for name in experiments.keys()}

    # 輸出目錄
    save_dir = os.path.join("evaluation_results", f"PBRS_{scene.title()}")
    os.makedirs(save_dir, exist_ok=True)

    # 繪圖
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

    exp_names = list(collected_data.keys())
    colors = ['#E24A4A', '#4A90E2', '#7ED321', '#F5A623', '#9B59B6',
              '#34495E', '#E67E22', '#16A085', '#2ECC71', '#F1C40F']
    labels = [short_mapping[n] for n in exp_names]

    # Fig.1 Bar Chart
    fig1, axes1 = plt.subplots(1, 3, figsize=(21, 7))
    fig1.suptitle(f'PBRS vs Additive: {scene.replace("_", " ").title()}',
                  fontsize=18, fontweight='bold', y=1.02)

    def make_barchart(ax, data_key, title, ylabel, fmt='.1f'):
        means, stds, bar_colors = [], [], []
        for idx, exp_name in enumerate(exp_names):
            data = collected_data[exp_name][data_key]
            d_valid = data[np.isfinite(data)]
            means.append(np.mean(d_valid) if len(d_valid) > 0 else 0)
            stds.append(stats.sem(d_valid) if len(d_valid) > 1 else 0)
            bar_colors.append(colors[idx % len(colors)])

        x_pos = np.arange(len(exp_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=bar_colors,
                      alpha=0.8, edgecolor='black', zorder=3)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_pos)
        xtick_font = 10 if len(exp_names) <= 6 else 8
        ax.set_xticklabels(labels, fontsize=xtick_font, rotation=20, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        max_h = max([m + s for m, s in zip(means, stds)]) if means else 1
        ax.set_ylim(0, max_h * 1.25)

        for i, bar in enumerate(bars):
            yval = bar.get_height()
            txt = f"{means[i]:{fmt}}±{stds[i]:{fmt}}"
            ax.text(bar.get_x() + bar.get_width()/2, yval + stds[i] + (max_h * 0.02),
                    txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    make_barchart(axes1[0], 'gates', 'Gates Passed', 'Gates', '.1f')
    make_barchart(axes1[1], 'jerk', 'Average Jerk (m/s³)', 'Jerk', '.2f')
    make_barchart(axes1[2], 'dist', 'Gate Distance (m)', 'Distance', '.3f')

    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "Fig1_BarChart.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Summary Table
    print("\n" + "=" * 90)
    print(f"  PBRS vs ADDITIVE COMPARISON ({scene.upper()})")
    print("=" * 90)
    print(f"{'Method':<25s} | {'Gates (Mean±SEM)':<18s} | {'Jerk':<15s} | {'Distance':<12s}")
    print("-" * 90)

    csv_path = os.path.join(save_dir, "Summary_Table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,Gates_Mean,Gates_SEM,Jerk_Mean,Jerk_SEM,Dist_Mean,Dist_SEM\n")

        for exp_name in exp_names:
            sname = short_mapping[exp_name]
            g_arr = collected_data[exp_name]['gates']
            g_val = g_arr[np.isfinite(g_arr)]
            j_arr = collected_data[exp_name]['jerk']
            j_val = j_arr[np.isfinite(j_arr)]
            d_arr = collected_data[exp_name]['dist']
            d_val = d_arr[np.isfinite(d_arr)]

            gm = np.mean(g_val) if len(g_val) > 0 else 0
            gs = stats.sem(g_val) if len(g_val) > 1 else 0
            jm = np.mean(j_val) if len(j_val) > 0 else 0
            js = stats.sem(j_val) if len(j_val) > 1 else 0
            dm = np.mean(d_val) if len(d_val) > 0 else 0
            ds = stats.sem(d_val) if len(d_val) > 1 else 0

            print(f"{sname:<25s} | {gm:5.1f} ± {gs:4.1f}       | "
                  f"{jm:5.2f} ± {js:4.2f}  | {dm:.3f} ± {ds:.3f}")
            f.write(f"{sname},{gm:.4f},{gs:.4f},{jm:.4f},{js:.4f},{dm:.4f},{ds:.4f}\n")

    print("=" * 90)

    # 統計分析
    if len(exp_names) >= 2:
        print(f"\n{'=' * 90}")
        print(f"  PAIRWISE STATISTICAL COMPARISON")
        print(f"{'=' * 90}")

        stat_csv = os.path.join(save_dir, "pairwise_stats.csv")
        with open(stat_csv, "w", encoding="utf-8") as f:
            f.write("Method_A,Method_B,Delta,Cohen_d,p_value,Significance\n")

            for i in range(len(exp_names)):
                for j in range(i + 1, len(exp_names)):
                    a_data = collected_data[exp_names[i]]['gates']
                    b_data = collected_data[exp_names[j]]['gates']
                    a_valid = a_data[np.isfinite(a_data)]
                    b_valid = b_data[np.isfinite(b_data)]

                    if len(a_valid) >= 2 and len(b_valid) >= 2:
                        ma, mb = np.mean(a_valid), np.mean(b_valid)
                        pooled_std = np.sqrt((np.var(a_valid) + np.var(b_valid)) / 2)
                        d_val = (mb - ma) / pooled_std if pooled_std > 1e-6 else 0.0
                        t_stat, p_val = stats.ttest_ind(a_valid, b_valid, equal_var=False)
                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'

                        la = short_mapping[exp_names[i]]
                        lb = short_mapping[exp_names[j]]
                        delta = mb - ma

                        print(f"  {la:15s} vs {lb:15s}: Δ={delta:+6.1f}, d={d_val:+.3f}, p={p_val:.4f} ({sig})")
                        f.write(f"{la},{lb},{delta:.2f},{d_val:.4f},{p_val:.6f},{sig}\n")

        print(f"{'=' * 90}")

    print(f"\n✅ 評估完成！結果已儲存至: {os.path.abspath(save_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description='PBRS vs Additive 評估腳本',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--scene', type=str, required=True,
                        choices=['sharp_turn', 'racing_track'],
                        help='評估場景')
    parser.add_argument('--episodes', type=int, default=50,
                        help='每個 Seed 的測試回合數')
    parser.add_argument('--gui', action='store_true',
                        help='GUI 展示模式')
    parser.add_argument('--compare_additive', action='store_true',
                        help='同時評估 Additive 方法 (用於對比)')
    parser.add_argument('--filter', type=str, nargs='+', default=[],
                        help='過濾特定實驗')
    args = parser.parse_args()

    if args.gui:
        print("GUI 模式尚未實作，請使用評估模式")
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()