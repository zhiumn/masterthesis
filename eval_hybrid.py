"""
PBRS vs Additive vs Hybrid 評估腳本

使用方式：

  # 評估 Sharp-Turn 場景的所有方法
  python eval_pbrs.py --scene sharp_turn --episodes 50 --compare_all

  # 評估 Racing Track 場景的所有方法
  python eval_pbrs.py --scene racing_track --episodes 50 --compare_all

  # 只評估 PBRS
  python eval_pbrs.py --scene sharp_turn --episodes 50

  # 同時評估 PBRS 和 Additive
  python eval_pbrs.py --scene sharp_turn --compare_additive

  # GUI 展示
  python eval_pbrs.py --scene sharp_turn --gui

方法比較：
  ┌─────────────────────────────────────────────────────────────┐
  │  方法        │ R_base              │ Bonus                  │
  │─────────────────────────────────────────────────────────────│
  │  Additive    │ R_acquire + R_align │ f(s)                   │
  │  Hybrid      │ R_acquire + R_align │ γ·Φ(s') - Φ(s) (PBRS)  │
  │  Pure PBRS   │ r_approach          │ γ·Φ(s') - Φ(s) (PBRS)  │
  └─────────────────────────────────────────────────────────────┘

輸出：
  - Fig1: Bar chart (Gates, Jerk, Distance) — seed-level Mean ± SEM
  - Summary Table (CSV) — seed-level Mean ± SEM
  - Pairwise Statistics (Hedges' g, Welch t-test, Holm-Bonferroni)

統計設計 (避免 pseudo-replication)：
  Step 1：每個方法訓練多個 seed
  Step 2：每個 seed 選出 best model
  Step 3：每個 best model 測試 N episodes
  Step 4：對每個 seed 計算 N episodes 的平均 Gates、Distance、Jerk
  Step 5：以 seed-level mean 作為該方法的統計樣本
  Step 6：回報 Mean ± SEM across seeds
  Step 7：對 Gates Passed 進行 pairwise Welch's t-test (主要指標)
  Step 8：回報 effect size (Hedges' g)
  Step 9：對 pairwise p-values 做 Holm-Bonferroni 修正
  Step 10：用修正後 p-value 判定顯著性
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


def finite_mean(xs):
    """安全地計算 mean，若無有效值則回傳 np.nan。"""
    xs = [x for x in xs if np.isfinite(x)]
    return np.mean(xs) if len(xs) > 0 else np.nan


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

    # 根據場景和方法類型載入環境
    if scene == 'sharp_turn':
        if 'hybrid' in exp_name.lower():
            from gym_pybullet_drones.envs.Sharpturncorridorhybrid import SharpTurnCorridorHybrid
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = SharpTurnCorridorHybrid(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM)
        elif 'pbrs' in exp_name.lower():
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
        if 'hybrid' in exp_name.lower():
            from gym_pybullet_drones.envs.Racingtrackhybrid import RacingTrackHybrid
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = RacingTrackHybrid(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM,
                straight_length=10.0, turn_radius=2.5, height_variation=0.4,
                gate_half_width=0.4, gate_half_height=0.4)
        elif 'pbrs' in exp_name.lower():
            from gym_pybullet_drones.envs.RacingTrackPBRS import RacingTrackPBRS
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = RacingTrackPBRS(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM,
                straight_length=10.0, turn_radius=2.5, height_variation=0.4,
                gate_half_width=0.4, gate_half_height=0.4)
        else:
            from gym_pybullet_drones.envs.RacingTrack import RacingTrack
            from gym_pybullet_drones.utils.enums import ActionType, ObservationType
            eval_env = RacingTrack(
                gui=False, num_lookahead_gates=lookahead,
                obs=ObservationType.KIN, act=ActionType.RPM,
                straight_length=10.0, turn_radius=2.5, height_variation=0.4,
                gate_half_width=0.4, gate_half_height=0.4)

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"\n[Warning] 無法載入模型 {model_path}: {e}")
        eval_env.close()
        return model_path, [], [], [], []

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
    print(f"=== PBRS / Hybrid / Additive 評估模式 ({scene}) ===")

    # 搜尋模型
    experiments = {}
    
    # PBRS 模型
    pbrs_pattern = f"models/*{scene}*pbrs*_seed*/best_model.zip"
    for model_path in glob.glob(pbrs_pattern):
        exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
        if args.filter and not any(exp_name.lower() == f.lower() or exp_name.lower().startswith(f.lower() + '_') or exp_name.lower().startswith(f.lower() + '-') for f in args.filter):
            continue
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(model_path)

    # Hybrid 模型
    if args.compare_all or args.compare_hybrid:
        hybrid_pattern = f"models/*{scene}*hybrid*_seed*/best_model.zip"
        # 也嘗試另一種命名模式
        hybrid_pattern2 = f"models/*hybrid*{scene}*_seed*/best_model.zip"
        for pattern in [hybrid_pattern, hybrid_pattern2, "models/*_hybrid_seed*/best_model.zip"]:
            for model_path in glob.glob(pattern):
                if scene not in model_path and 'sharp' not in model_path.lower() and 'racing' not in model_path.lower():
                    continue
                if scene == 'sharp_turn' and 'racing' in model_path.lower():
                    continue
                if scene == 'racing_track' and 'sharp' in model_path.lower():
                    continue
                exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
                if args.filter and not any(exp_name.lower() == f.lower() or exp_name.lower().startswith(f.lower() + '_') or exp_name.lower().startswith(f.lower() + '-') for f in args.filter):
                    continue
                if exp_name not in experiments:
                    experiments[exp_name] = []
                if model_path not in experiments[exp_name]:
                    experiments[exp_name].append(model_path)

    # Additive 模型
    if args.compare_all or args.compare_additive:
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
                if 'pbrs' in model_path.lower() or 'hybrid' in model_path.lower():
                    continue
                exp_name = os.path.basename(os.path.dirname(model_path)).rsplit("_seed", 1)[0]
                if args.filter and not any(exp_name.lower() == f.lower() or exp_name.lower().startswith(f.lower() + '_') or exp_name.lower().startswith(f.lower() + '-') for f in args.filter):
                    continue
                if exp_name not in experiments:
                    experiments[exp_name] = []
                if model_path not in experiments[exp_name]:
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
        max_workers = min(max(1, (os.cpu_count() or 4) - 1), len(worker_tasks), 16)
        print(f"\n 平行評估 {len(worker_tasks)} 個模型，使用 {max_workers} 核心...")

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
                'gates': finite_mean(seed_g),
                'jerk': finite_mean(seed_j),
                'dist': finite_mean(seed_d),
                'laps': finite_mean(seed_l),
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
        if 'hybrid' in exp_name.lower():
            return 'PBRS (shared)'
        if 'pbrs-full' in exp_name.lower() or 'pbrs_full' in exp_name.lower():
            return 'PBRS-Full'
        if 'pbrs-dist' in exp_name.lower():
            return 'PBRS-Dist'
        if 'pbrs-cent' in exp_name.lower():
            return 'PBRS-Cent'
        if 'pbrs-head' in exp_name.lower():
            return 'PBRS-Head'
        if 'pbrs' in exp_name.lower():
            return 'PBRS-Full'
        if 'dynamic' in exp_name.lower() or 'a7-d' in exp_name.lower() or 'a7_d' in exp_name.lower():
            return 'Additive-Exp'
        if 'static' in exp_name.lower() or 'fixed' in exp_name.lower() or 'a7-s' in exp_name.lower() or 'a7_s' in exp_name.lower():
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
        # 若是早期的命名 'Full-Exp' 等等，也一併替換
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

    # 輸出目錄 (根據實際比較對象動態命名，避免不同組合互相覆蓋)
    folder_name_parts = []
    for name in experiments.keys():
        clean = short_mapping[name].replace("+", "").replace("-", "_").replace(" ", "_")
        if clean not in folder_name_parts:
            folder_name_parts.append(clean)
    subfolder_name = f"{scene}_" + "-vs-".join(folder_name_parts) if folder_name_parts else f"PBRS_{scene.title()}"
    save_dir = os.path.join("evaluation_results", subfolder_name)
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

    # === 統計輔助函數 ===
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
        """Holm-Bonferroni 多重比較修正。
        輸入: list of raw p-values
        輸出: list of corrected p-values (同樣順序)
        """
        n = len(p_values)
        if n == 0:
            return []
        # 1) 排序 indices
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros(n)
        # 2) 從小到大逐步修正
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

    # === Bar Chart (使用 seed-level Mean ± SEM) ===
    def make_barchart(ax, data_key, title, ylabel, fmt='.1f'):
        means, sems, bar_colors = [], [], []
        for idx, exp_name in enumerate(exp_names):
            seed_vals = get_seed_array(exp_name, data_key)
            means.append(np.mean(seed_vals) if len(seed_vals) > 0 else 0)
            sems.append(stats.sem(seed_vals) if len(seed_vals) > 1 else 0)
            bar_colors.append(colors[idx % len(colors)])

        x_pos = np.arange(len(exp_names))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=6, color=bar_colors,
                      alpha=0.8, edgecolor='black', zorder=3)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_pos)
        xtick_font = 10 if len(exp_names) <= 6 else 8
        ax.set_xticklabels(labels, fontsize=xtick_font, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        max_h = max([m + s for m, s in zip(means, sems)]) if means else 1
        ax.set_ylim(0, max_h * 1.25)

        for i, bar in enumerate(bars):
            yval = bar.get_height()
            txt = f"{means[i]:{fmt}}±{sems[i]:{fmt}}"
            ax.text(bar.get_x() + bar.get_width()/2, yval + sems[i] + (max_h * 0.02),
                    txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    make_barchart(axes1[0], 'gates', 'Gates Passed (seed-level)', 'Gates', '.1f')
    make_barchart(axes1[1], 'jerk', 'Average Jerk (seed-level)', 'Jerk (m/s³)', '.2f')
    make_barchart(axes1[2], 'dist', 'Gate Distance (seed-level)', 'Distance (m)', '.3f')

    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "Fig1_BarChart.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ================================================================
    # Summary Table (seed-level Mean ± SEM)
    # ================================================================
    print("\n" + "=" * 100)
    print(f"  SEED-LEVEL SUMMARY ({scene.upper()})  —  統計單位: 訓練 seed (避免 pseudo-replication)")
    print("=" * 100)
    print(f"{'Method':<25s} | {'N_seeds':>7s} | {'Gates (Mean±SEM)':<18s} | {'Jerk (Mean±SEM)':<18s} | {'Dist (Mean±SEM)':<18s}")
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

            print(f"{sname:<25s} | {n_s:>7d} | {gm:5.1f} ± {gs:4.1f}       | "
                  f"{jm:5.2f} ± {js:4.2f}      | {dm:.3f} ± {ds:.3f}")
            f.write(f"{sname},{n_s},{gm:.4f},{gs:.4f},{jm:.4f},{js:.4f},{dm:.4f},{ds:.4f}\n")

    print("=" * 100)

    # ================================================================
    # Pairwise Statistical Comparison (seed-level)
    # ================================================================
    if len(exp_names) >= 2:
        # --- 主要指標: Gates Passed ---
        print(f"\n{'=' * 110}")
        print(f"  PAIRWISE STATISTICAL COMPARISON — 主要指標: Gates Passed (seed-level Welch's t-test + Holm-Bonferroni)")
        print(f"{'=' * 110}")

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

    print(f"\n[SUCCESS] 所有評估完成！結果已儲存至: {os.path.abspath(save_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description='PBRS vs Additive vs Hybrid 評估腳本',
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
                        help='同時評估 Additive 方法')
    parser.add_argument('--compare_hybrid', action='store_true',
                        help='同時評估 Hybrid 方法')
    parser.add_argument('--compare_all', action='store_true',
                        help='同時評估所有方法 (PBRS + Hybrid + Additive)')
    parser.add_argument('--filter', type=str, nargs='+', default=[],
                        help='過濾特定實驗')
    args = parser.parse_args()

    if args.gui:
        print("GUI 模式尚未實作，請使用評估模式")
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()