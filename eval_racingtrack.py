"""
Racing Track 評估腳本

使用方式：

  # 論文數據採集模式 (產出 Fig1~4 + 統計表)
  python eval_racing_track.py --episodes 50

  # GUI 展示模式
  python eval_racing_track.py --gui

  # 過濾特定實驗
  python eval_racing_track.py --filter racing_ablation A3 Full

  # 與其他場景比較 (跨場景遷移測試)
  python eval_racing_track.py --cross_scene sharp_turn

評估特性：
  - 封閉賽道: 以「圈數 × 12 + 剩餘 gates」計算總穿門數
  - Episode 結束條件: miss 1 次即結束 (嚴格連續穿梭) 或 90 秒 timeout
  - 額外指標: laps_completed (完成圈數)
"""

import argparse
import time
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from scipy import stats


def extract_lookahead(exp_name, default_val=2):
    """從實驗名稱中提取 lookahead 數量。"""
    match = re.search(r"_L(\d+)", exp_name)
    if match:
        return int(match.group(1))
    return default_val


def eval_model_worker(worker_args):
    """多進程評估 worker。"""
    model_path, exp_name, test_seeds, lookahead = worker_args
    import traceback
    try:
        import numpy as np
        from stable_baselines3 import PPO, SAC, TD3

        # 1. 載入模型
        if "SAC" in exp_name:
            model = SAC.load(model_path)
        elif "TD3" in exp_name:
            model = TD3.load(model_path)
        else:
            model = PPO.load(model_path)

        # 2. 動態推導 lookahead 避免 shape mismatch
        obs_shape = model.policy.observation_space.shape[-1]
        if obs_shape == 84:
            lookahead = 2
        elif obs_shape == 78:
            lookahead = 1
        elif obs_shape == 90:
            lookahead = 3

        # 3. 根據實驗名稱載入對應環境
        if "racing_ablation" in exp_name.lower():
            from gym_pybullet_drones.envs.RacingTrackAblation import RacingTrackAblation
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType

            # 解析消融條件
            ablation_config = parse_ablation_condition(exp_name)
            eval_env = RacingTrackAblation(
                gui=False,
                num_lookahead_gates=lookahead,
                obs=ObservationType.KIN,
                act=ActionType.RPM,
                straight_length=10.0, turn_radius=2.5, height_variation=0.4,
                gate_half_width=0.4, gate_half_height=0.4,
                **ablation_config
            )
        elif "racing_track" in exp_name.lower():
            from gym_pybullet_drones.envs.RacingTrack import RacingTrack
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType

            is_dynamic = "dynamic" in exp_name.lower()
            eval_env = RacingTrack(
                gui=False,
                num_lookahead_gates=lookahead,
                obs=ObservationType.KIN,
                act=ActionType.RPM,
                use_dynamic_reward=is_dynamic,
                straight_length=10.0, turn_radius=2.5, height_variation=0.4,
                gate_half_width=0.4, gate_half_height=0.4,
            )
        else:
            # Fallback: 嘗試載入 RacingTrack
            from gym_pybullet_drones.envs.RacingTrack import RacingTrack
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType

            eval_env = RacingTrack(
                gui=False,
                num_lookahead_gates=lookahead,
                obs=ObservationType.KIN,
                act=ActionType.RPM,
                straight_length=10.0, turn_radius=2.5, height_variation=0.4,
                gate_half_width=0.4, gate_half_height=0.4,
            )

        seed_gates, seed_jerk, seed_dist, seed_laps = [], [], [], []

        for test_seed in test_seeds:
            obs, info = eval_env.reset(seed=test_seed)

            while True:
                action, _ = model.predict(obs, deterministic=True)
                if len(action.shape) == 1:
                    action = np.array([action])
                obs, reward, terminated, truncated, info = eval_env.step(action)

                # 嚴格連續穿梭模式：只要 miss 1 次就立即結束
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
    except Exception as e:
        print(f"\n[Worker Error] 處理 {model_path} 時發生錯誤:\n{traceback.format_exc()}")
        return worker_args[0], [], [], [], []



def parse_ablation_condition(exp_name):
    """從實驗名稱解析消融條件。"""
    exp_lower = exp_name.lower()

    # 預設值
    config = {
        'use_dynamic_reward': True,
        'enable_centering': True,
        'enable_heading': True,
        'enable_smooth': True,
        'proximity_type': 'exponential',
    }

    if 'a0' in exp_lower:
        config['enable_centering'] = False
        config['enable_heading'] = False
        config['enable_smooth'] = False
    elif 'a1' in exp_lower:
        config['enable_centering'] = True
        config['enable_heading'] = False
        config['enable_smooth'] = False
    elif 'a2' in exp_lower:
        config['enable_centering'] = False
        config['enable_heading'] = True
        config['enable_smooth'] = False
    elif 'a3' in exp_lower:
        config['enable_centering'] = False
        config['enable_heading'] = False
        config['enable_smooth'] = True
    elif 'a7-s' in exp_lower or 'static' in exp_lower:
        config['use_dynamic_reward'] = False
    elif 'p4' in exp_lower:
        config['proximity_type'] = 'linear'
    elif 'p5' in exp_lower:
        config['proximity_type'] = 'step'
    elif 'p6' in exp_lower:
        config['proximity_type'] = 'sigmoid'

    return config


def run_single_visual(args):
    """GUI 展示模式。"""
    print("=== GUI 展示模式 (Racing Track) ===")

    all_models = glob.glob("models/*racing*_seed*/best_model.zip")

    if args.filter:
        valid_models = []
        for p in all_models:
            folder_name = os.path.basename(os.path.dirname(p))
            if any(f.lower() in folder_name.lower() for f in args.filter):
                valid_models.append(p)
        all_models = valid_models

    if not all_models:
        print("未找到任何 Racing Track 模型!")
        return

    model_path = all_models[0]
    exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
    lookahead = extract_lookahead(exp_name, default_val=args.lookahead)

    print(f"載入模型: {model_path}")
    model = PPO.load(model_path)

    from gym_pybullet_drones.envs.RacingTrack import RacingTrack
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    env = RacingTrack(
        gui=True,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=lookahead,
        straight_length=10.0, turn_radius=2.5, height_variation=0.4,
        gate_half_width=0.4, gate_half_height=0.4,
    )

    for ep in range(3):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\n展示局數 {ep+1}/3")
        print(f"軌道結構: 膠囊形中等賽道 | 直道 10m | U-Turn R=2.5m | 12 gates/圈")

        while True:
            action, _ = model.predict(obs, deterministic=True)
            if len(action.shape) == 1:
                action = np.array([action])
            obs, reward, terminated, truncated, info = env.step(action)

            if args.fpv:
                import pybullet as p
                state = env._getDroneStateVector(0)
                drone_pos = state[0:3]
                drone_yaw = state[7:10][2]
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=np.degrees(drone_yaw) - 90,
                    cameraPitch=-20,
                    cameraTargetPosition=drone_pos,
                    physicsClientId=env.CLIENT
                )

            time.sleep(1/240.0)

            # 嚴格連續穿梭模式：只要 miss 1 次就立即結束
            if env.consecutive_misses > 0:
                terminated = True

            if terminated or truncated:
                laps = info.get('laps_completed', 0)
                gates = info.get('gates_passed', 0)
                jerk = info.get('avg_jerk', 0)
                dist = info.get('avg_pass_dist', 0)
                print(f"結果 -> 圈數: {laps}, 穿門: {gates}, "
                      f"Jerk: {jerk:.2f}, 過門誤差: {dist:.3f}m")
                break

    env.close()


def run_multi_eval(args):
    """論文數據採集模式。"""
    print("=== Racing Track 論文數據採集模式 ===")
    print("掃描所有 Racing Track 模型並產出統計圖表...")

    experiments = {}
    for s_dir in glob.glob("models/*_seed*"):
        exp_name = os.path.basename(s_dir).rsplit("_seed", 1)[0]

        # 只處理 racing 相關實驗
        if "racing" not in exp_name.lower():
            continue

        if args.filter:
            if not any(f.lower() in exp_name.lower() for f in args.filter):
                continue

        model_path = os.path.join(s_dir, "best_model.zip")
        if os.path.exists(model_path):
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(model_path)

    if not experiments:
        print("未找到任何 Racing Track 模型!")
        return

    print(f"找到 {len(experiments)} 種實驗配置:")
    for name, paths in experiments.items():
        print(f"  - {name}: {len(paths)} seeds")

    collected_data = {}
    test_seeds = [150000 + i for i in range(args.episodes)]

    import concurrent.futures

    worker_tasks = []
    for exp_name, model_paths in experiments.items():
        lookahead = extract_lookahead(exp_name, default_val=args.lookahead)
        for p in model_paths:
            worker_tasks.append((p, exp_name, test_seeds, lookahead))

    results_map = {}
    if worker_tasks:
        max_workers = min(max(1, (os.cpu_count() or 4) // 2), len(worker_tasks), 4) # 限制最多 4 個進程避免 OOM
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

    # ============================================================
    # 標籤映射
    # ============================================================
    FULL_NAMES = {
        'A0': 'Base Only (No Bonus)',
        'A1': 'Base + Centering',
        'A2': 'Base + Heading',
        'A3': 'Base + Smooth',
        'dynamic': 'Additive Bonus (Exp)',
        'static': 'Additive Bonus (Fixed)',
        'P4': 'Additive Bonus (Linear)',
        'P5': 'Additive Bonus (Step)',
        'P6': 'Additive Bonus (Sigmoid)',
    }

    SHORT_NAMES = {
        'A0': 'Base',
        'A1': '+Centering',
        'A2': '+Heading',
        'A3': '+Smooth',
        'dynamic': 'Additive-Exp',
        'static': 'Additive-Fixed',
        'P4': 'Additive-Lin',
        'P5': 'Additive-Step',
        'P6': 'Additive-Sig',
    }

    def parse_cond(ename):
        ename_lower = ename.lower()
        if "racing_ablation_" in ename_lower:
            cond = ename_lower.replace("racing_ablation_", "").upper()
            # 處理 A7-S, A7-D 等
            if 'A7-S' in cond or 'A7_S' in cond:
                return 'static'
            if 'A7-D' in cond or 'A7_D' in cond:
                return 'dynamic'
            return cond
        if "racing_track_dynamic" in ename_lower:
            return "dynamic"
        if "racing_track_static" in ename_lower:
            return "static"
        return ename

    short_mapping = {}
    full_mapping = {}

    for exp_name in experiments.keys():
        cond = parse_cond(exp_name)
        short_mapping[exp_name] = SHORT_NAMES.get(cond, cond)
        full_mapping[exp_name] = FULL_NAMES.get(cond, cond)

    # 建立輸出目錄
    save_dir = os.path.join("evaluation_results", "RacingTrack_Ablation")
    os.makedirs(save_dir, exist_ok=True)

    # 設定繪圖風格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

    exp_names = list(collected_data.keys())
    colors = ['#4A90E2', '#E24A4A', '#7ED321', '#F5A623', '#9B59B6',
              '#34495E', '#E67E22', '#16A085', '#2ECC71', '#F1C40F']
    labels = [short_mapping[n] for n in exp_names]

    # ============================================================
    # Fig.1 Bar Chart (包含 Laps)
    # ============================================================
    fig1, axes1 = plt.subplots(1, 4, figsize=(28, 7))
    fig1.suptitle('Racing Track: Method Comparison', fontsize=18, fontweight='bold', y=1.02)

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
        ax.set_xticklabels(labels, fontsize=xtick_font, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        max_h = max([m + s for m, s in zip(means, stds)]) if means else 1
        ax.set_ylim(0, max_h * 1.25)

        for i, bar in enumerate(bars):
            yval = bar.get_height()
            txt = f"{means[i]:{fmt}}±{stds[i]:{fmt}}"
            ax.text(bar.get_x() + bar.get_width()/2, yval + stds[i] + (max_h * 0.02),
                    txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    make_barchart(axes1[0], 'gates', 'Gates Passed', 'Gates', '.1f')
    make_barchart(axes1[1], 'laps', 'Laps Completed', 'Laps', '.2f')
    make_barchart(axes1[2], 'jerk', 'Average Jerk (m/s³)', 'Jerk', '.2f')
    make_barchart(axes1[3], 'dist', 'Gate Distance (m)', 'Distance', '.3f')

    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "Fig1_BarChart.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ============================================================
    # Fig.2 Box Plot
    # ============================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 7))
    fig2.suptitle('Racing Track: Distribution Analysis', fontsize=18, fontweight='bold', y=1.02)

    def make_boxplot(ax, data_key, title, ylabel):
        data_list = [collected_data[name][data_key] for name in exp_names]
        data_list = [d[np.isfinite(d)] for d in data_list]

        bplot = ax.boxplot(data_list, vert=True, patch_artist=True, widths=0.5,
                           medianprops=dict(color='black', linewidth=2.5))

        for i, patch in enumerate(bplot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(range(1, len(exp_names) + 1))
        xtick_font = 10 if len(exp_names) <= 6 else 8
        ax.set_xticklabels(labels, fontsize=xtick_font, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    make_boxplot(axes2[0], 'gates', 'Gates Passed Distribution', 'Gates')
    make_boxplot(axes2[1], 'jerk', 'Jerk Distribution', 'Jerk (m/s³)')
    make_boxplot(axes2[2], 'dist', 'Gate Distance Distribution', 'Distance (m)')

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "Fig2_BoxPlot.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ============================================================
    # Fig.3 Scatter (Performance vs Smoothness)
    # ============================================================
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    for idx, exp_name in enumerate(exp_names):
        seeds_data = collected_data[exp_name]['seeds']
        x_jerk = [s['jerk'] for s in seeds_data]
        y_gates = [s['gates'] for s in seeds_data]

        c = colors[idx % len(colors)]
        ax3.scatter(x_jerk, y_gates, color=c, label=short_mapping[exp_name],
                    s=150, alpha=0.8, edgecolors='k', zorder=3)

    ax3.set_title('Racing Track: Performance vs Smoothness Trade-off',
                  fontsize=16, fontweight='bold')
    ax3.set_xlabel('Average Jerk (m/s³)', fontsize=14)
    ax3.set_ylabel('Mean Gates Passed', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax3.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')

    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "Fig3_Scatter.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 90)
    print("  RACING TRACK EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Method':<25s} | {'Gates (Mean±SEM)':<18s} | {'Laps':<12s} | {'Jerk':<15s}")
    print("-" * 90)

    csv_path = os.path.join(save_dir, "Summary_Table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,ShortName,Gates_Mean,Gates_SEM,Laps_Mean,Laps_SEM,"
                "Jerk_Mean,Jerk_SEM,Dist_Mean,Dist_SEM\n")

        for exp_name in exp_names:
            fname = full_mapping[exp_name]
            sname = short_mapping[exp_name]

            g_arr = collected_data[exp_name]['gates']
            g_val = g_arr[np.isfinite(g_arr)]
            l_arr = collected_data[exp_name]['laps']
            l_val = l_arr[np.isfinite(l_arr)]
            j_arr = collected_data[exp_name]['jerk']
            j_val = j_arr[np.isfinite(j_arr)]
            d_arr = collected_data[exp_name]['dist']
            d_val = d_arr[np.isfinite(d_arr)]

            gm = np.mean(g_val) if len(g_val) > 0 else 0
            lm = np.mean(l_val) if len(l_val) > 0 else 0
            jm = np.mean(j_val) if len(j_val) > 0 else 0
            dm = np.mean(d_val) if len(d_val) > 0 else 0
            gs = stats.sem(g_val) if len(g_val) > 1 else 0
            ls = stats.sem(l_val) if len(l_val) > 1 else 0
            js = stats.sem(j_val) if len(j_val) > 1 else 0
            ds = stats.sem(d_val) if len(d_val) > 1 else 0

            print(f"{fname:<25s} | {gm:5.1f} ± {gs:4.1f}       | "
                  f"{lm:4.2f} ± {ls:4.2f} | {jm:5.2f} ± {js:4.2f}")
            f.write(f"{fname},{sname},{gm:.4f},{gs:.4f},{lm:.4f},{ls:.4f},"
                    f"{jm:.4f},{js:.4f},{dm:.4f},{ds:.4f}\n")

    print("=" * 90)

    # ============================================================
    # Fig.4 表格圖片
    # ============================================================
    fig4, ax4 = plt.subplots(figsize=(12, len(exp_names) * 0.6 + 1.5))
    ax4.axis('off')

    table_data = []
    columns = ['Method', 'Gates (Mean±SEM)', 'Laps', 'Jerk (m/s³)', 'Distance (m)']

    for exp_name in exp_names:
        g_arr = collected_data[exp_name]['gates']
        g_val = g_arr[np.isfinite(g_arr)]
        l_arr = collected_data[exp_name]['laps']
        l_val = l_arr[np.isfinite(l_arr)]
        j_arr = collected_data[exp_name]['jerk']
        j_val = j_arr[np.isfinite(j_arr)]
        d_arr = collected_data[exp_name]['dist']
        d_val = d_arr[np.isfinite(d_arr)]

        gm = np.mean(g_val) if len(g_val) > 0 else 0
        lm = np.mean(l_val) if len(l_val) > 0 else 0
        jm = np.mean(j_val) if len(j_val) > 0 else 0
        dm = np.mean(d_val) if len(d_val) > 0 else 0
        gs = stats.sem(g_val) if len(g_val) > 1 else 0
        ls = stats.sem(l_val) if len(l_val) > 1 else 0
        js = stats.sem(j_val) if len(j_val) > 1 else 0
        ds = stats.sem(d_val) if len(d_val) > 1 else 0

        table_data.append([
            full_mapping[exp_name],
            f"{gm:.1f} ± {gs:.1f}",
            f"{lm:.2f} ± {ls:.2f}",
            f"{jm:.2f} ± {js:.2f}",
            f"{dm:.3f} ± {ds:.3f}"
        ])

    table = ax4.table(cellText=table_data, colLabels=columns,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#B0BEC5')
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#F8F9F9')

    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, "Summary_Table_image.png"),
                 dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # ============================================================
    # 統計分析: Cohen's d + Welch t-test
    # ============================================================
    if len(exp_names) >= 2:
        print(f"\n{'=' * 90}")
        print(f"  PAIRWISE STATISTICAL COMPARISON (Gates Passed)")
        print(f"{'=' * 90}")

        stat_csv = os.path.join(save_dir, "pairwise_stats.csv")
        with open(stat_csv, "w", encoding="utf-8") as f:
            f.write("Method_A,Method_B,Mean_A,Mean_B,Delta,Cohen_d,p_value,Significance\n")

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

                        print(f"  {la:12s} vs {lb:12s}: Δ={delta:+6.1f}, d={d_val:+.3f}, p={p_val:.4f} ({sig})")
                        f.write(f"{la},{lb},{ma:.2f},{mb:.2f},{delta:.2f},{d_val:.4f},{p_val:.6f},{sig}\n")

        print(f"{'=' * 90}")
        print(f"  統計摘要已寫入: {stat_csv}")

    # ============================================================
    # Component Contribution Analysis (如果有消融數據)
    # ============================================================
    # 找出 A0, A1, A2, A3 和 Full 的數據
    a0_data = None
    full_data = None
    component_data = {}

    for exp_name in exp_names:
        cond = parse_cond(exp_name)
        gates = collected_data[exp_name]['gates']
        mean_gates = np.mean(gates[np.isfinite(gates)])

        if cond == 'A0':
            a0_data = mean_gates
        elif cond in ['dynamic', 'A7-D']:
            full_data = mean_gates
        elif cond in ['A1', 'A2', 'A3']:
            component_data[cond] = mean_gates

    if a0_data is not None and full_data is not None and len(component_data) > 0:
        print(f"\n{'=' * 90}")
        print(f"  COMPONENT CONTRIBUTION ANALYSIS (Racing Track)")
        print(f"{'=' * 90}")
        print(f"  Base Only (A0): {a0_data:.1f} gates")
        print(f"  Full Bonus:     {full_data:.1f} gates")
        print(f"  Total Effect:   {full_data - a0_data:+.1f} gates")
        print(f"  {'─' * 50}")

        for cond in ['A1', 'A2', 'A3']:
            if cond in component_data:
                contribution = component_data[cond] - a0_data
                pct = (contribution / (full_data - a0_data)) * 100 if (full_data - a0_data) > 0 else 0
                cond_name = {'A1': 'Centering', 'A2': 'Heading', 'A3': 'Smooth'}[cond]
                print(f"  {cond_name:12s}: {component_data[cond]:.1f} gates | "
                      f"Contribution: {contribution:+.1f} ({pct:.1f}%)")

        print(f"{'=' * 90}")

    print(f"\n✅ 評估完成！結果已儲存至: {os.path.abspath(save_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description='Racing Track 模型評估腳本',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--episodes', type=int, default=50,
                        help='每個 Seed 的測試回合數 (default: 50)')
    parser.add_argument('--gui', action='store_true',
                        help='GUI 展示模式')
    parser.add_argument('--fpv', action='store_true',
                        help='第一人稱視角 (需搭配 --gui)')
    parser.add_argument('--filter', type=str, nargs='+', default=[],
                        help='過濾特定實驗 (例: --filter A3 Full)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機數種子，用以在不同方法中固定相同的測試地圖 (default: 42)')
    parser.add_argument('--lookahead', type=int, default=2,
                        help='預設 lookahead gates 數量 (default: 2)')
    args = parser.parse_args()

    if args.gui:
        run_single_visual(args)
    else:
        run_multi_eval(args)


if __name__ == '__main__':
    main()