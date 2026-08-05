"""
Zero-shot Cross-scenario Transfer Evaluation

不需要任何新的訓練。
拿已有的 best_model.zip 直接在不同場景上做 deterministic evaluation。

使用方式：

  # 方式 1：自動掃描 models/ 下所有主實驗模型，做完整交叉測試
  python eval_cross_scenario.py

  # 方式 2：手動指定模型和測試場景
  python eval_cross_scenario.py \
      --model_path models/sharp_turn/sharp_turn_dynamic_seed0/best_model.zip \
      --train_scene sharp_turn \
      --test_scene standard

  # 方式 3：手動指定多個模型目錄
  python eval_cross_scenario.py \
      --model_dirs models/sharp_turn models/PPO_InfiniteGate_L2_seed0

  # 調整測試 episode 數
  python eval_cross_scenario.py --episodes 100

測試矩陣：
  ┌─────────────────────────────────────────────────────┐
  │  Train Scene ╲ Test Scene │  Standard  │  Sharp Turn │
  │─────────────────────────────────────────────────────│
  │  Standard                 │  原始性能   │  遷移性能   │
  │  Sharp Turn               │  遷移性能   │  原始性能   │
  └─────────────────────────────────────────────────────┘

輸出：
  - Transfer Matrix 表格 (console)
  - 性能衰減率分析
  - CSV 詳細結果 (evaluation_results/transfer/)
"""

import os
import glob
import re
import argparse
import numpy as np
import csv
from collections import defaultdict

import concurrent.futures
import multiprocessing

def eval_model_worker(worker_args):
    """Top-level worker function for multiprocessing."""
    model_path, test_scene, test_seeds = worker_args
    import numpy as np
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType
    from stable_baselines3 import PPO, SAC, TD3

    if test_scene == 'sharp_turn':
        from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
        eval_env = SharpTurnCorridor(gui=False, num_lookahead_gates=2, obs=ObservationType.KIN, act=ActionType.RPM)
    else:
        from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
        eval_env = InfiniteGateAviary(gui=False, num_lookahead_gates=2, obs=ObservationType.KIN, act=ActionType.RPM)

    if "SAC" in model_path:
        model = SAC.load(model_path)
    elif "TD3" in model_path:
        model = TD3.load(model_path)
    else:
        model = PPO.load(model_path)
        
    seed_g, seed_j, seed_d = [], [], []
    
    for test_seed in test_seeds:
        obs, info = eval_env.reset(seed=test_seed)
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            if len(action.shape) == 1: action = np.array([action])
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # [嚴格機制] 漏掉門框即強制結束回合
            if eval_env.consecutive_misses > 0:
                terminated = True
                
            if terminated or truncated:
                seed_g.append(info.get('gates_passed', 0))
                seed_j.append(info.get('avg_jerk', 0.0))
                seed_d.append(info.get('avg_pass_dist', 0.0))
                break
                
    eval_env.close()
    return model_path, test_scene, seed_g, seed_j, seed_d

def detect_model_info(model_path):
    """從模型路徑自動推斷訓練場景、方法名稱、seed。"""
    path_lower = model_path.lower()
    parent_dir = os.path.basename(os.path.dirname(model_path)).lower()

    if 'sharp' in path_lower:
        train_scene = 'sharp_turn'
    else:
        train_scene = 'standard'

    if 'dynamic' in parent_dir or 'ppo_infinitegate' in parent_dir:
        method = 'Full Bonus (Exp)'
    elif 'static' in parent_dir or 'baseline' in parent_dir:
        method = 'Full Bonus (Fixed)'
    else:
        method = 'Unknown'

    match = re.search(r'seed(\d+)', model_path)
    seed = int(match.group(1)) if match else -1

    return {
        'train_scene': train_scene,
        'method': method,
        'seed': seed,
    }


def scan_main_experiment_models(model_dirs):
    """掃描所有主實驗的 best_model.zip，排除 ablation 模型。"""
    models = []
    exclude_keywords = ['ablation', '_a0_', '_a1_', '_a2_', '_a3_',
                        '_p4_', '_p5_', '_p6_', 'algo_gen', 'sac_', 'td3_']

    for base_dir in model_dirs:
        if not os.path.exists(base_dir):
            print(f"[WARN] 目錄不存在: {base_dir}")
            continue

        for path in glob.glob(os.path.join(base_dir, '**', 'best_model.zip'), recursive=True):
            dirname_lower = os.path.dirname(path).lower()
            if any(kw in dirname_lower for kw in exclude_keywords):
                continue

            info = detect_model_info(path)
            if info['method'] == 'Unknown':
                continue

            models.append({'path': path, **info})

    return models


def run_full_transfer(model_dirs, episodes=50):
    """執行完整的交叉場景遷移測試。"""

    scene_labels = {
        'standard': 'Standard Corridor',
        'sharp_turn': 'Sharp-Turn Corridor',
    }

    print("=" * 70)
    print("  Zero-shot Cross-scenario Transfer Evaluation")
    print("=" * 70)

    models = scan_main_experiment_models(model_dirs)

    if not models:
        print("\n未找到任何主實驗模型！")
        print("請確認目錄結構，或手動指定:")
        print("  python eval_cross_scenario.py \\")
        print("      --model_path <path> \\")
        print("      --train_scene <standard|sharp_turn> \\")
        print("      --test_scene <standard|sharp_turn>")
        return

    print(f"\n找到 {len(models)} 個主實驗模型:")
    for m in models:
        print(f"  Train: {scene_labels[m['train_scene']]:22s} | "
              f"Method: {m['method']:20s} | "
              f"Seed: {m['seed']:2d} | {m['path']}")

    results = []
    test_scenes = ['standard', 'sharp_turn']
    test_seeds = [15000 + i for i in range(episodes)]
    
    worker_tasks = []
    for model_info in models:
        for test_scene in test_scenes:
            worker_tasks.append((model_info['path'], test_scene, test_seeds))
            
    total_evals = len(worker_tasks)
    done = 0

    if worker_tasks:
        max_workers = min(max(1, (os.cpu_count() or 4) - 1), total_evals, 16)
        print(f"\n🚀 啟動平行化 Zero-shot 測試！共 {total_evals} 個任務，分配給 {max_workers} 個核心...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(eval_model_worker, worker_tasks):
                res_path, res_scene, seed_g, seed_j, seed_d = result
                done += 1
                
                # 找到對應的 model_info
                info = next(m for m in models if m['path'] == res_path)
                
                is_transfer = (info['train_scene'] != res_scene)
                label = "TRANSFER" if is_transfer else "Same-scene"
                
                print(f"[{done}/{total_evals}] {label} | Train: {info['train_scene']} | Test: {res_scene} | Method: {info['method']} Seed: {info['seed']}")
                
                gates = np.array(seed_g)
                jerk = np.array([j for j in seed_j if np.isfinite(j)])
                dist = np.array([d for d in seed_d if np.isfinite(d)])
                
                g_mean = float(np.mean(gates)) if len(gates) > 0 else 0.0
                g_std = float(np.std(gates)) if len(gates) > 0 else 0.0
                g_median = float(np.median(gates)) if len(gates) > 0 else 0.0
                j_mean = float(np.mean(jerk)) if len(jerk) > 0 else 0.0
                d_mean = float(np.mean(dist)) if len(dist) > 0 else 0.0

                result_dict = {
                    'train_scene': info['train_scene'],
                    'test_scene': res_scene,
                    'method': info['method'],
                    'seed': info['seed'],
                    'is_transfer': is_transfer,
                    'gates_mean': g_mean,
                    'gates_std': g_std,
                    'gates_median': g_median,
                    'jerk_mean': j_mean,
                    'dist_mean': d_mean,
                }
                results.append(result_dict)
                print(f"  -> Gates: {g_mean:.1f} +/- {g_std:.1f} (median: {g_median:.0f})")

    if not results:
        print("\n沒有成功的評估結果！")
        return

    # ---- Transfer Matrix ----
    print("\n" + "=" * 70)
    print("  TRANSFER MATRIX")
    print("=" * 70)

    grouped = defaultdict(list)
    for r in results:
        key = (r['train_scene'], r['test_scene'], r['method'])
        grouped[key].append(r['gates_mean'])

    print(f"\n{'Train Scene':>22s} | {'Test Scene':>22s} | "
          f"{'Method':>20s} | {'Gates':>16s} | {'N':>3s} | {'Type':>8s}")
    print("-" * 100)

    for key in sorted(grouped.keys()):
        train_s, test_s, method = key
        vals = np.array(grouped[key])
        is_transfer = train_s != test_s
        t_label = "TRANSFER" if is_transfer else "Same"
        print(f"{scene_labels[train_s]:>22s} | {scene_labels[test_s]:>22s} | "
              f"{method:>20s} | {np.mean(vals):5.1f} +/- {np.std(vals):4.1f}  | "
              f"{len(vals):>3d} | {t_label:>8s}")

    # ---- Degradation Analysis ----
    print(f"\n{'=' * 70}")
    print("  TRANSFER DEGRADATION ANALYSIS")
    print("=" * 70)

    methods_seen = sorted(set(r['method'] for r in results))

    for method in methods_seen:
        print(f"\n  {method}:")
        for train_s in ['standard', 'sharp_turn']:
            same_key = (train_s, train_s, method)
            if same_key not in grouped:
                continue
            same_perf = np.mean(grouped[same_key])

            for test_s in ['standard', 'sharp_turn']:
                if test_s == train_s:
                    continue
                transfer_key = (train_s, test_s, method)
                if transfer_key not in grouped:
                    continue
                transfer_perf = np.mean(grouped[transfer_key])

                if same_perf > 0:
                    change_pct = (transfer_perf - same_perf) / same_perf * 100
                else:
                    change_pct = 0

                arrow = "+" if change_pct > 0 else ""
                print(f"    {scene_labels[train_s]:>22s} -> {scene_labels[test_s]:<22s}: "
                      f"{same_perf:5.1f} -> {transfer_perf:5.1f} "
                      f"({arrow}{change_pct:.1f}%)")

    # ---- Save CSV ----
    os.makedirs('evaluation_results/transfer', exist_ok=True)
    csv_path = 'evaluation_results/transfer/cross_scenario_results.csv'

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['train_scene', 'test_scene', 'method', 'seed',
                      'is_transfer', 'gates_mean', 'gates_std',
                      'gates_median', 'jerk_mean', 'dist_mean']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nCSV 已儲存: {csv_path}")
    print("\n完成！")


def run_single(model_path, train_scene, test_scene, episodes=50):
    """測試單一模型。"""
    scene_labels = {'standard': 'Standard Corridor', 'sharp_turn': 'Sharp-Turn Corridor'}
    is_transfer = (train_scene != test_scene)

    print(f"Model:       {model_path}")
    print(f"Train scene: {scene_labels[train_scene]}")
    print(f"Test scene:  {scene_labels[test_scene]}")
    print(f"Transfer:    {'YES' if is_transfer else 'No (same scene)'}")
    print(f"Episodes:    {episodes}")
    print()

    test_seeds = [15000 + i for i in range(episodes)]
    worker_args = (model_path, test_scene, test_seeds)
    _, _, seed_g, seed_j, seed_d = eval_model_worker(worker_args)

    gates = np.array(seed_g)
    jerk = np.array([j for j in seed_j if np.isfinite(j)])
    dist = np.array([d for d in seed_d if np.isfinite(d)])

    g_mean, g_std, g_median = float(np.mean(gates)), float(np.std(gates)), float(np.median(gates))
    j_mean, j_std = float(np.mean(jerk)), float(np.std(jerk))
    d_mean, d_std = float(np.mean(dist)), float(np.std(dist))

    print(f"Gates Passed: {g_mean:.2f} +/- {g_std:.2f} (median: {g_median:.0f})")
    print(f"Avg Jerk:     {j_mean:.2f} +/- {j_std:.2f}")
    print(f"Avg Distance: {d_mean:.4f} +/- {d_std:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot Cross-scenario Transfer Evaluation',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model_path', type=str, default=None,
                        help='單一模型路徑')
    parser.add_argument('--train_scene', type=str, default=None,
                        choices=['standard', 'sharp_turn'],
                        help='模型的訓練場景 (搭配 --model_path)')
    parser.add_argument('--test_scene', type=str, default=None,
                        choices=['standard', 'sharp_turn'],
                        help='要測試的場景')
    parser.add_argument('--episodes', type=int, default=50,
                        help='每個模型的測試 episode 數 (default: 50)')
    parser.add_argument('--model_dirs', type=str, nargs='+',
                        default=['models/'],
                        help='搜尋模型的根目錄 (default: models/)')
    args = parser.parse_args()

    if args.model_path:
        if not args.train_scene or not args.test_scene:
            print("手動模式需要同時指定 --train_scene 和 --test_scene")
            return
        run_single(args.model_path, args.train_scene, args.test_scene, args.episodes)
    else:
        run_full_transfer(args.model_dirs, args.episodes)


if __name__ == '__main__':
    main()