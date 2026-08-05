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
from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from scipy import stats


def finite_mean(xs):
    """安全地計算 mean，若無有效值則回傳 np.nan。"""
    xs = [x for x in xs if np.isfinite(x)]
    return np.mean(xs) if len(xs) > 0 else np.nan

def extract_lookahead(exp_name):
    match = re.search(r"_L(\d+)", exp_name)
    if match:
        return int(match.group(1))
    return 2

def eval_model_worker(worker_args):
    """Top-level worker function for multiprocessing."""
    model_path, exp_name, test_seeds, lookahead = worker_args
    import numpy as np
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType
    from stable_baselines3 import PPO, SAC, TD3

    # 1. 載入模型
    if "SAC" in exp_name:
        model = SAC.load(model_path)
    elif "TD3" in exp_name:
        model = TD3.load(model_path)
    else:
        model = PPO.load(model_path)

    # 2. 動態推導 lookahead
    obs_shape = model.policy.observation_space.shape[-1]
    if obs_shape == 84:
        lookahead = 2
    elif obs_shape == 78:
        lookahead = 1
    elif obs_shape == 90:
        lookahead = 3

    # 3. 根據模型名稱動態決定要讀取哪個難度的場景
    if "sharp" in exp_name.lower():
        from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
        is_strict = "_strict" in exp_name.lower()
        eval_env = SharpTurnCorridor(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM, strict_miss=is_strict, gate_half_width=0.4 if is_strict else 0.5, heading_bonus_scale=5.0 if is_strict else 0.5, centering_bonus_scale=3.0 if is_strict else 1.5)
    elif "baseline_hybrid" in exp_name:
        from gym_pybullet_drones.envs.InfiniteGateAviaryhybrid import InfiniteGateAviaryHybrid
        eval_env = InfiniteGateAviaryHybrid(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
    else:
        from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
        eval_env = InfiniteGateAviary(gui=False, num_lookahead_gates=lookahead, obs=ObservationType.KIN, act=ActionType.RPM)
        
    seed_g, seed_j, seed_d = [], [], []
    
    for test_seed in test_seeds:
        obs, info = eval_env.reset(seed=test_seed)
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            if len(action.shape) == 1: action = np.array([action])
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # [新增] 嚴格連續穿梭模式：只要漏掉一次門框，測試局立刻強制結束
            if eval_env.consecutive_misses > 0:
                terminated = True
                
            if terminated or truncated:
                gp = info.get('gates_passed', 0)
                jk = info.get('avg_jerk', 0.0)
                ds = info.get('avg_pass_dist', 0.0)
                
                seed_g.append(gp)
                seed_j.append(jk)
                seed_d.append(ds)
                break
                
    eval_env.close()
    return model_path, seed_g, seed_j, seed_d

def run_single_visual(args):
    print("=== GUI 展示模式 ===")
    print("在此模式下，將只尋找一個模型載入並在 PyBullet 視窗中實際展示飛行姿態。")
    all_models = glob.glob("models/*_seed*/best_model.zip")
    
    # [新增] 過濾機制
    if args.filter:
        valid_models = []
        for p in all_models:
            folder_name = os.path.basename(os.path.dirname(p))
            if any(f in folder_name for f in args.filter):
                valid_models.append(p)
        all_models = valid_models

    if not all_models:
        print("未找到任何符合條件的訓練好模型!")
        return
        
    model_path = all_models[0]
    exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
    lookahead = extract_lookahead(exp_name)
    
    model = PPO.load(model_path)
    
    if "sharp" in exp_name.lower():
        from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
        is_strict = "_strict" in os.path.basename(os.path.dirname(model_path)).lower()
        env = SharpTurnCorridor(gui=True, record=args.record, obs=ObservationType.KIN, act=ActionType.RPM, num_lookahead_gates=lookahead, strict_miss=is_strict, gate_half_width=0.4 if is_strict else 0.5, heading_bonus_scale=5.0 if is_strict else 0.5, centering_bonus_scale=3.0 if is_strict else 1.5)
    elif "baseline_hybrid" in exp_name:
        from gym_pybullet_drones.envs.InfiniteGateAviaryhybrid import InfiniteGateAviaryHybrid
        env = InfiniteGateAviaryHybrid(gui=True, record=args.record, obs=ObservationType.KIN, act=ActionType.RPM, num_lookahead_gates=lookahead)
    else:
        env = InfiniteGateAviary(gui=True, record=args.record, obs=ObservationType.KIN, act=ActionType.RPM, num_lookahead_gates=lookahead)
    
    for ep in range(3):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"展示局數 {ep+1}/3")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            if len(action.shape) == 1: action = np.array([action])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if args.fpv:
                import pybullet as p
                state = env._getDroneStateVector(0)
                drone_pos = state[0:3]
                drone_yaw = state[7:10][2]
                # 將相機強制鎖定在無人機後方，製造第一人稱 / 追尾視角 (Third-person follow)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.5,
                    cameraYaw=np.degrees(drone_yaw) - 90, 
                    cameraPitch=-15, 
                    cameraTargetPosition=drone_pos,
                    physicsClientId=env.CLIENT
                )
            
            # [新增] 嚴格連續穿梭模式：只要漏掉一次門框，測試局立刻強制結束
            if env.consecutive_misses > 0:
                terminated = True

            time.sleep(1/240.0)
            if terminated or truncated:
                print(f"結果 -> 穿門: {info.get('gates_passed', 0)}, Jerk: {info.get('avg_jerk', 0):.4f}, 過門誤差: {info.get('avg_pass_dist', 0):.4f}m")
                break
    env.close()

def run_multi_eval(args):
    print("=== 論文數據採集模式 ===")
    print("將掃描所有的 Seed 與 Method 並產出 Fig1(Bar), Fig2(Box), Fig3(Scatter) 圖表。")
    
    experiments = {}
    for s_dir in glob.glob("models/*_seed*"):
        exp_name = os.path.basename(s_dir).rsplit("_seed", 1)[0]
        
        # [修正] 過濾機制：使用前綴匹配，避免 'ablation_A0' 誤匹配 'sharp_ablation_A0'
        if args.filter:
            if not any(exp_name == f or exp_name.startswith(f + '_') or exp_name.startswith(f + '-') for f in args.filter):
                continue
                
        model_path = os.path.join(s_dir, "best_model.zip")
        if os.path.exists(model_path):
           if exp_name not in experiments:
               experiments[exp_name] = []
           experiments[exp_name].append(model_path)
           
    if not experiments:
        print("未找到任何模型 (best_model.zip)！")
        return
        
    collected_data = {}
    test_seeds = [150000 + i for i in range(args.episodes)]
    
    import concurrent.futures
    import multiprocessing
    
    worker_tasks = []
    for exp_name, model_paths in experiments.items():
        lookahead = extract_lookahead(exp_name)
        for p in model_paths:
            worker_tasks.append((p, exp_name, test_seeds, lookahead))
            
    results_map = {}
    if worker_tasks:
        max_workers = min(max(1, (os.cpu_count() or 4) - 1), len(worker_tasks), 16)
        print(f"\n [平行評估] 總共 {len(worker_tasks)} 個模型任務，分配 {max_workers} 個核心...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, result in enumerate(executor.map(eval_model_worker, worker_tasks)):
                model_path, seed_g, seed_j, seed_d = result
                results_map[model_path] = (seed_g, seed_j, seed_d)
                print(f"  - [{idx+1}/{len(worker_tasks)}] 完成評估: {model_path}")

    for exp_name, model_paths in experiments.items():
        all_gates, all_jerk, all_dist = [], [], []
        seed_stats = []
        
        for p in model_paths:
            if p not in results_map: continue
            seed_g, seed_j, seed_d = results_map[p]
            
            # 計算該 Seed 的各項平均
            seed_stats.append({
                'jerk': finite_mean(seed_j),
                'gates': finite_mean(seed_g),
                'dist': finite_mean(seed_d)
            })
            
            all_gates.extend(seed_g)
            all_jerk.extend(seed_j)
            all_dist.extend(seed_d)
        
        collected_data[exp_name] = {
            'gates': np.array(all_gates),
            'jerk': np.array(all_jerk),
            'dist': np.array(all_dist),
            'seeds': seed_stats
        }
        
    # ==========================
    # 標籤映射 (與 eval_hybrid.py 的 get_short_name 完全一致)
    # ==========================
    def get_short_name(exp_name):
        name = exp_name.lower()
        if 'hybrid' in name:
            return 'PBRS (shared)'
        if 'pbrs-full' in name or 'pbrs_full' in name:
            return 'PBRS-Full'
        if 'pbrs-dist' in name:
            return 'PBRS-Dist'
        if 'pbrs-cent' in name:
            return 'PBRS-Cent'
        if 'pbrs-head' in name:
            return 'PBRS-Head'
        if 'pbrs' in name:
            return 'PBRS-Full'
        if 'dynamic' in name or '_a7-d' in name or '_a7_d' in name:
            return 'Additive-Exp'
        if 'static' in name or 'fixed' in name:
            return 'Additive-Fixed'
        if '_a7-s' in name or '_a7_s' in name:
            return 'Additive-Fixed'
        if '_p4' in name:
            return 'Additive-Lin'
        if '_p5' in name:
            return 'Additive-Step'
        if '_p6' in name:
            return 'Additive-Sig'
        if '_a0' in name:
            return 'Base Only'
        if '_a1' in name:
            return '+Centering'
        if '_a2' in name:
            return '+Heading'
        if '_a3' in name:
            return '+Smooth'
        return exp_name

    short_mapping = {}
    full_mapping = {}
    folder_name_parts = []
    
    for exp_name in experiments.keys():
        label = get_short_name(exp_name)
        short_mapping[exp_name] = label
        full_mapping[exp_name] = label
        clean_name = label.replace("+", "").replace("-", "_")
        folder_name_parts.append(clean_name)

    subfolder_name = "-vs-".join(folder_name_parts)
    if not subfolder_name: subfolder_name = "Evaluation_Results"
    save_dir = os.path.join("evaluation_results", subfolder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try: plt.style.use('seaborn-whitegrid')
        except: pass

    exp_names = list(collected_data.keys())
    colors = ['#4A90E2', '#E24A4A', '#7ED321', '#F5A623', '#9B59B6', '#34495E', '#E67E22', '#16A085', '#2ECC71', '#F1C40F']
    labels = [short_mapping[n] for n in exp_names]

    # === 統計輔助函數 (與 eval_hybrid.py 完全一致) ===
    def get_seed_array(exp_name, key):
        """取得某方法的 seed-level mean array。"""
        vals = [s[key] for s in collected_data[exp_name]['seeds'] if np.isfinite(s[key])]
        return np.array(vals)

    def cohens_d(a, b):
        """Pooled Cohen's d (equal-weight pooled SD)."""
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return 0.0
        sa2 = np.var(a, ddof=1)
        sb2 = np.var(b, ddof=1)
        pooled_sd = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
        if pooled_sd < 1e-9:
            return 0.0
        return (np.mean(b) - np.mean(a)) / pooled_sd

    def hedges_g(a, b):
        """Hedges' g：Cohen's d 的小樣本修正版。"""
        d = cohens_d(a, b)
        df = len(a) + len(b) - 2
        if df <= 0:
            return d
        correction = 1 - (3 / (4 * df - 1))
        return d * correction

    def holm_bonferroni(p_values):
        """Holm-Bonferroni 多重比較修正。"""
        n = len(p_values)
        if n == 0:
            return []
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros(n)
        cummax = 0.0
        for rank, idx in enumerate(sorted_idx):
            adjusted = p_values[idx] * (n - rank)
            cummax = max(cummax, adjusted)
            corrected[idx] = min(cummax, 1.0)
        return corrected.tolist()

    def sig_label(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'

    # ==========================
    # Fig.1 主圖 (Bar chart) — seed-level Mean ± SEM
    # ==========================
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 7))
    fig1.suptitle('Method Comparison (Seed-Level Means)', fontsize=18, fontweight='bold', y=1.05)
    
    def make_barchart(ax, data_key, title, ylabel, fmt='.1f'):
        means, sems, bar_colors = [], [], []
        for idx, exp_name in enumerate(exp_names):
            seed_vals = get_seed_array(exp_name, data_key)
            means.append(np.mean(seed_vals) if len(seed_vals) > 0 else 0)
            sems.append(stats.sem(seed_vals) if len(seed_vals) > 1 else 0)
            bar_colors.append(colors[idx % len(colors)])
            
        x_pos = np.arange(len(exp_names))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=8, color=bar_colors, alpha=0.8, edgecolor='black', zorder=3)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_pos)
        xtick_font = 11 if len(exp_names) <= 5 else 9
        ax.set_xticklabels(labels, fontsize=xtick_font, rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        
        max_h = max([m + s for m, s in zip(means, sems)]) if means else 1
        ax.set_ylim(0, max_h * 1.25)
        
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            txt = f"{means[i]:{fmt}}±{sems[i]:{fmt}}"
            bar_font = 11 if len(exp_names) <= 5 else 9
            ax.text(bar.get_x() + bar.get_width()/2, yval + sems[i] + (max_h * 0.02),
                    txt, ha='center', va='bottom', fontsize=bar_font, fontweight='bold')

    make_barchart(axes1[0], 'gates', 'Gates Passed (seed-level)', 'Gates Passed', '.1f')
    make_barchart(axes1[1], 'jerk', 'Average Jerk (seed-level)', 'Average Jerk (m/s³)', '.2f')
    make_barchart(axes1[2], 'dist', 'Gate Distance (seed-level)', 'Distance to Gate Center (m)', '.3f')
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "Fig1_BarChart.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ==========================
    # Fig.2 分析圖 (Box plot) — 保留 episode-level 分佈供參考
    # ==========================
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
    fig2.suptitle('Deterministic Evaluation Across Methods (Distribution)', fontsize=18, fontweight='bold', y=1.05)
    
    def make_boxplot(ax, idx_key, title, ylabel):
        data_list = [collected_data[name][idx_key] for name in exp_names]
        data_list = [d[np.isfinite(d)] for d in data_list]
        
        bplot = ax.boxplot(data_list, vert=True, patch_artist=True, widths=0.5,
                           medianprops=dict(color='black', linewidth=2.5))
        
        for i, patch in enumerate(bplot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.8)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(range(1, len(exp_names) + 1))
        
        xtick_font = 11 if len(exp_names) <= 5 else 9
        ax.set_xticklabels(labels, fontsize=xtick_font, rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    make_boxplot(axes2[0], 'gates', 'Task Performance (Gates Passed)', 'Number of Gates Passed')
    make_boxplot(axes2[1], 'jerk', 'Flight Smoothness (Average Jerk)', 'Jerk ($m/s^3$)')
    make_boxplot(axes2[2], 'dist', 'Gate-Centering Precision (Distance to Gate Center)', 'Distance (m)')

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "Fig2_BoxPlot.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ==========================
    # Fig.3 散佈圖 (Scatter: Trade-off)
    # ==========================
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    for idx, exp_name in enumerate(exp_names):
        seeds_data = collected_data[exp_name]['seeds']
        x_jerk = [s['jerk'] for s in seeds_data if np.isfinite(s['jerk'])]
        y_gates = [s['gates'] for s in seeds_data if np.isfinite(s['gates'])]
        
        c = colors[idx % len(colors)]
        ax3.scatter(x_jerk, y_gates, color=c, label=short_mapping[exp_name], 
                    s=150, alpha=0.8, edgecolors='k', zorder=3)
        
    ax3.set_title('Performance vs Smoothness Trade-off (Per Seed)', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Average Jerk ($m/s^3$)', fontsize=14)
    ax3.set_ylabel('Mean Gates Passed', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax3.legend(loc='best', fontsize=12, framealpha=0.9, edgecolor='black')
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "Fig3_Scatter.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # ================================================================
    # Summary Table (seed-level Mean ± SEM) — 與 eval_hybrid.py 完全一致
    # ================================================================
    print("\n" + "=" * 100)
    print(f"  SEED-LEVEL SUMMARY  —  統計單位: 訓練 seed (避免 pseudo-replication)")
    print("=" * 100)
    print(f"{'Method':<32s} | {'N_seeds':>7s} | {'Gates (Mean±SEM)':<18s} | {'Jerk (Mean±SEM)':<18s} | {'Dist (Mean±SEM)':<18s}")
    print("-" * 100)
    
    csv_path = os.path.join(save_dir, "Summary_Table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,N_seeds,Gates_Mean,Gates_SEM,Jerk_Mean,Jerk_SEM,Dist_Mean,Dist_SEM\n")
        for exp_name in exp_names:
            sname = short_mapping[exp_name]

            g_seed = get_seed_array(exp_name, 'gates')
            j_seed = get_seed_array(exp_name, 'jerk')
            d_seed = get_seed_array(exp_name, 'dist')

            # 使用主要指標 (gates) 的有效 seed 數作為 N_seeds
            n_s = len(g_seed)

            gm = np.mean(g_seed) if len(g_seed) > 0 else 0
            gs = stats.sem(g_seed) if len(g_seed) > 1 else 0
            jm = np.mean(j_seed) if len(j_seed) > 0 else 0
            js = stats.sem(j_seed) if len(j_seed) > 1 else 0
            dm = np.mean(d_seed) if len(d_seed) > 0 else 0
            ds = stats.sem(d_seed) if len(d_seed) > 1 else 0
            
            print(f"{sname:<32s} | {n_s:>7d} | {gm:5.1f} ± {gs:4.1f}       | "
                  f"{jm:5.2f} ± {js:4.2f}      | {dm:.3f} ± {ds:.3f}")
            f.write(f"{sname},{n_s},{gm:.4f},{gs:.4f},{jm:.4f},{js:.4f},{dm:.4f},{ds:.4f}\n")
    print("=" * 100)

    # ==========================
    # Fig.4 自動產生高質感 CSV 表格圖片 (seed-level)
    # ==========================
    fig4, ax4 = plt.subplots(figsize=(10, len(exp_names) * 0.5 + 1.2))
    ax4.axis('off')
    ax4.axis('tight')

    table_data = []
    columns = ['Method', 'N', 'Gates Passed (Mean ± SEM)', 'Average Jerk (m/s³)', 'Gate Distance (m)']
    
    for exp_name in exp_names:
        g_seed = get_seed_array(exp_name, 'gates')
        j_seed = get_seed_array(exp_name, 'jerk')
        d_seed = get_seed_array(exp_name, 'dist')
        n_s = len(g_seed)
        
        gm = np.mean(g_seed) if len(g_seed) > 0 else 0
        gs = stats.sem(g_seed) if len(g_seed) > 1 else 0
        jm = np.mean(j_seed) if len(j_seed) > 0 else 0
        js = stats.sem(j_seed) if len(j_seed) > 1 else 0
        dm = np.mean(d_seed) if len(d_seed) > 0 else 0
        ds = stats.sem(d_seed) if len(d_seed) > 1 else 0
        
        table_data.append([
            full_mapping[exp_name],
            str(n_s),
            f"{gm:.1f} ± {gs:.1f}",
            f"{jm:.2f} ± {js:.2f}",
            f"{dm:.2f} ± {ds:.2f}"
        ])
        
    table = ax4.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.25, 2.2)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#B0BEC5')
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#F8F9F9')
            
    fig4.tight_layout()
    table_png_path = os.path.join(save_dir, "Summary_Table_image.png")
    fig4.savefig(table_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # ================================================================
    # Pairwise Statistical Comparison (seed-level)
    # 與 eval_hybrid.py 完全一致：Hedges' g + Welch t-test + Holm-Bonferroni
    # ================================================================
    if len(exp_names) >= 2:
        def run_pairwise(data_key, csv_filename, section_title):
            """對特定指標執行 planned comparisons (Family A & Family B)"""
            print(f"  [{section_title}]")
            
            name_to_exp = {short_mapping[e]: e for e in exp_names}
            
            family_A = [
                ('Base Only', 'Additive-Fixed'),
                ('Base Only', 'Additive-Step'),
                ('PBRS (shared)', 'Additive-Fixed'),
                ('PBRS (shared)', 'Additive-Step'),
                ('Additive-Fixed', 'Additive-Step')
            ]
            
            family_B = [
                ('Base Only', '+Centering'),
                ('Base Only', '+Heading'),
                ('Base Only', '+Smooth')
            ]
            
            families = {'Family A': family_A, 'Family B': family_B}
            all_results = []
            
            stat_csv = os.path.join(save_dir, csv_filename)
            with open(stat_csv, "w", encoding="utf-8") as f:
                f.write("Family,Method_A,Method_B,N_A,N_B,Delta,CI_95_Low,CI_95_High,Hedges_g,p_raw,p_holm,Significance\n")
                
                for fam_name, pairs_list in families.items():
                    print(f"\n    --- {fam_name} ---")
                    pairs_data = []
                    raw_p_values = []
                    
                    for la, lb in pairs_list:
                        if la in name_to_exp and lb in name_to_exp:
                            a_seed = get_seed_array(name_to_exp[la], data_key)
                            b_seed = get_seed_array(name_to_exp[lb], data_key)
                            
                            if len(a_seed) >= 2 and len(b_seed) >= 2:
                                delta = np.mean(b_seed) - np.mean(a_seed)
                                g_val = hedges_g(a_seed, b_seed)
                                t_stat, p_raw = stats.ttest_ind(a_seed, b_seed, equal_var=False)
                                
                                var_a = np.var(a_seed, ddof=1)
                                var_b = np.var(b_seed, ddof=1)
                                n_a = len(a_seed)
                                n_b = len(b_seed)
                                
                                if var_a + var_b == 0:
                                    ci_low, ci_high = delta, delta
                                else:
                                    se = np.sqrt(var_a/n_a + var_b/n_b)
                                    df = (var_a/n_a + var_b/n_b)**2 / ((var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1))
                                    t_crit = stats.t.ppf(0.975, df)
                                    margin = t_crit * se
                                    ci_low, ci_high = delta - margin, delta + margin
                                
                                pairs_data.append((la, lb, delta, ci_low, ci_high, g_val, p_raw, n_a, n_b))
                                raw_p_values.append(p_raw)
                            else:
                                pairs_data.append((la, lb, 0.0, 0.0, 0.0, 0.0, 1.0, len(a_seed), len(b_seed)))
                                raw_p_values.append(1.0)
                                
                    if not raw_p_values:
                        continue
                        
                    # Holm-Bonferroni 修正 (僅針對同一個 Family 內的比較)
                    corrected_p = holm_bonferroni(raw_p_values)
                    
                    for k, (la, lb, delta, ci_low, ci_high, g_val, p_raw, na, nb) in enumerate(pairs_data):
                        p_holm = corrected_p[k]
                        sig = sig_label(p_holm)
                        print(f"    {la:15s} vs {lb:15s}: "
                              f"n=({na},{nb}), Δ={delta:+6.1f} [{ci_low:+.1f}, {ci_high:+.1f}], g={g_val:+.3f}, "
                              f"p_raw={p_raw:.4f}, p_holm={p_holm:.4f} ({sig})")
                        f.write(f"{fam_name},{la},{lb},{na},{nb},{delta:.2f},{ci_low:.2f},{ci_high:.2f},{g_val:.4f},{p_raw:.6f},{p_holm:.6f},{sig}\n")
                        all_results.append((la, lb, p_holm))
                        
            print(f"  → 檢定結果已儲存至: {stat_csv}")
            return all_results

        # 主要指標: Gates Passed
        print(f"\n{'=' * 110}")
        print(f"  PAIRWISE STATISTICAL COMPARISON — 主要指標: Gates Passed (seed-level Welch's t-test + Holm-Bonferroni)")
        print(f"{'=' * 110}")
        run_pairwise('gates', 'pairwise_stats_gates.csv', 'Gates Passed')

        # 輔助指標: Jerk
        print(f"\n{'=' * 110}")
        print(f"  PAIRWISE — 輔助指標: Average Jerk (seed-level)")
        print(f"{'=' * 110}")
        run_pairwise('jerk', 'pairwise_stats_jerk.csv', 'Average Jerk')

        # 輔助指標: Gate Distance
        print(f"\n{'=' * 110}")
        print(f"  PAIRWISE — 輔助指標: Gate Distance (seed-level)")
        print(f"{'=' * 110}")
        run_pairwise('dist', 'pairwise_stats_dist.csv', 'Gate Distance')

        print(f"\n{'=' * 110}")
    
    print(f"\n [SUCCESS] 所有評估完成！")
    print(f"  - 數據表 (CSV & 表格圖片) 已匯出至: {os.path.abspath(save_dir)}")
    print(f"  - 論文圖表 (Fig1~3) 已輸出至: {os.path.abspath(save_dir)}")
    print(f"  - 統計檢定 (Pairwise) 已輸出至: {os.path.abspath(save_dir)}")
def main():
    parser = argparse.ArgumentParser(description='測試已訓練好的 PPO 多模型並輸出論文版多圖表')
    parser.add_argument('--episodes', type=int, default=50, help='每個 Seed 要測試的回合數')
    parser.add_argument('--gui', action='store_true', help='開啟 GUI 視窗只做視覺展示(不會產圖)')
    parser.add_argument('--fpv', action='store_true', help='在 GUI 模式下開啟無人機追尾/第一人稱視角 (需搭配 --gui)')
    parser.add_argument('--filter', type=str, nargs='+', default=[], help='只評估檔名包含這些字串的實驗 (例: --filter ablation 或 --filter SAC TD3 PPO Baseline)')
    parser.add_argument('--seed', type=int, default=42, help='隨機數種子，用以在不同方法中固定相同的門框排序 (default: 42)')
    parser.add_argument('--record', action='store_true', help='是否錄製展示影片 (儲存至 results/ 目錄)')
    args = parser.parse_args()
    
    if args.gui:
        run_single_visual(args)
    else:
        run_multi_eval(args)

if __name__ == '__main__':
    main()
