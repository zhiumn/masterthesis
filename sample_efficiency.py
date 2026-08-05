"""
Sample Efficiency 分析腳本 — AUC + 多門檻 TTT

計算指標:
  1. AUC (Area Under Curve): 學習曲線正規化後的積分面積
     → 反映整體學習效率 (學習速度 + 最終效能)
  2. TTT (Time To Threshold): 首次達到指定門檻所需的訓練步數
     → 門檻自動設定為各場景最佳方法最終效能的 50%, 70%, 90%
     → 消除單一門檻的選擇偏差

門檻計算方式 (基於 Deterministic Reward):
  threshold = best_final_reward × percentage
  Standard: best_final=16271.1 → 50%=8135.6, 70%=11389.8, 90%=14644.0
  Sharp:    best_final=10554.4 → 50%=5277.2, 70%=7388.1, 90%=9499.0
  Racing:   best_final=16589.6 → 50%=8294.8, 70%=11612.7, 90%=14930.6

使用方式:
  # 分析單一場景的主實驗
  python analyze_sample_efficiency.py --scene standard --exp main

  # 分析急彎場景的消融實驗
  python analyze_sample_efficiency.py --scene sharp --exp ablation

  # 分析所有場景所有實驗
  python sample_efficiency.py --scene all --exp all

  # 自訂百分比門檻 (例如 60% 和 80%)
  python analyze_sample_efficiency.py --scene sharp --exp main --percentages 0.6 0.8

  # 指定資料來源
  python analyze_sample_efficiency.py --scene standard --exp main --source monitor
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# ================================================================
# 嘗試匯入 TensorBoard 讀取器
# ================================================================
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[WARNING] tensorboard 未安裝，將僅使用 Monitor CSV")

# ================================================================
# 場景與實驗配置
# ================================================================

# 主實驗的資料夾命名慣例 (根據你的訓練腳本)
SCENE_CONFIGS = {
    'standard': {
        'name': '基準走廊 (Standard Corridor)',
        'main_methods': {
            'Base Only':       'ablation_A0_seed{seed}',
            'Additive-Fixed':  'ablation_A7-S_seed{seed}',
            'Additive-Step':   'ablation_P5_seed{seed}',
            'PBRS (shared)':   'baseline_hybrid_seed{seed}',
        },
        'ablation_methods': {
            'Base Only':    'ablation_A0_seed{seed}',
            '+Centering':   'ablation_A1_seed{seed}',
            '+Heading':     'ablation_A2_seed{seed}',
            '+Smooth':      'ablation_A3_seed{seed}',
            'Additive-Step': 'ablation_P5_seed{seed}',
        },
        'variant_methods': {
            'Additive-Exp':  'ablation_A7-D_seed{seed}',       # P1 = exponential
            'Additive-Lin':  'ablation_P4_seed{seed}',
            'Additive-Step': 'ablation_P5_seed{seed}',
            'Additive-Sig':  'ablation_P6_seed{seed}',
        },
        'best_final': 14514.7,   # 最佳方法 (Additive-sig) 的最終 Reward
        'episode_len': 60,
    },
    'sharp': {
        'name': '急彎走廊 (Sharp Turn)',
        'main_methods': {
            'Base Only':       'sharp_ablation_A0_seed{seed}',
            'Additive-Fixed':  'sharp_ablation_A7-S_seed{seed}',
            'Additive-Step':   'sharp_ablation_P5_seed{seed}',
            'PBRS (shared)':   'sharp_hybrid_seed{seed}',
        },
        'ablation_methods': {
            'Base Only':     'sharp_ablation_A0_seed{seed}',
            '+Centering':    'sharp_ablation_A1_seed{seed}',
            '+Heading':      'sharp_ablation_A2_seed{seed}',
            '+Smooth':       'sharp_ablation_A3_seed{seed}',
            'Additive-Step':  'sharp_ablation_P5_seed{seed}',
        },
        'variant_methods': {
            'Additive-Exp':  'sharp_ablation_A7-D_seed{seed}',
            'Additive-Lin':  'sharp_ablation_P4_seed{seed}',
            'Additive-Step': 'sharp_ablation_P5_seed{seed}',
            'Additive-Sig':  'sharp_ablation_P6_seed{seed}',
        },
        'best_final': 6057.9,   # 最佳方法 (Additive-exp) 的最終 Reward
        'episode_len': 60,
    },
    'racing': {
        'name': 'Racing Track',
        'main_methods': {
            'Base Only':       'racing_ablation_A0_seed{seed}',
            'Additive-Fixed':  'racing_track_static_seed{seed}',
            'Additive-Step':    'racing_ablation_P5_seed{seed}',
            'PBRS (shared)':   'racing_hybrid_seed{seed}',
        },
        'ablation_methods': {
            'Base Only':     'racing_ablation_A0_seed{seed}',
            '+Centering':    'racing_ablation_A1_seed{seed}',
            '+Heading':      'racing_ablation_A2_seed{seed}',
            '+Smooth':       'racing_ablation_A3_seed{seed}',
            'Additive-Step': 'racing_ablation_P5_seed{seed}',
        },
        'variant_methods': {
            'Additive-Exp':  'racing_track_dynamic_seed{seed}',
            'Additive-Lin':  'racing_ablation_P4_seed{seed}',
            'Additive-Step': 'racing_ablation_P5_seed{seed}',
            'Additive-Sig':  'racing_ablation_P6_seed{seed}',
        },
        'best_final': 7493.6,   # 最佳方法 (Additive-step) 的最終 Reward
        'episode_len': 90,
    },
}

# 多門檻百分比: 50%, 70%, 90%
TTT_PERCENTAGES = [0.5, 0.7, 0.9]

SEEDS = list(range(10))


# ================================================================
# 資料讀取函數
# ================================================================

def read_tensorboard_scalar(log_dir, tag='eval/mean_reward', max_steps=2000000):
    """從 TensorBoard events 檔案讀取指定 scalar。

    Parameters
    ----------
    log_dir : str
        TensorBoard log 目錄
    tag : str
        要讀取的 scalar tag (常見: 'eval/mean_reward', 'rollout/ep_rew_mean')
    max_steps : int
        最大步數

    Returns
    -------
    steps : np.ndarray
        訓練步數
    values : np.ndarray
        對應的數值
    """
    if not HAS_TB:
        return None, None

    event_files = glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True)
    if not event_files:
        return None, None

    # 嘗試每個 event file
    for event_file in sorted(event_files):
        try:
            ea = EventAccumulator(os.path.dirname(event_file))
            ea.Reload()

            if tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                steps = np.array([e.step for e in events])
                values = np.array([e.value for e in events])

                # 過濾到 max_steps
                mask = steps <= max_steps
                return steps[mask], values[mask]
        except Exception:
            continue

    return None, None


def read_monitor_csv(monitor_dir):
    """從 Monitor CSV 讀取 episode 資訊。

    Parameters
    ----------
    monitor_dir : str
        monitor CSV 所在目錄

    Returns
    -------
    df : pd.DataFrame or None
        包含 r (reward), l (length), t (time), 以及 info 欄位
    """
    csv_files = glob.glob(os.path.join(monitor_dir, '*.monitor.csv'))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            # Monitor CSV 的第一行是 metadata (以 # 開頭)
            df = pd.read_csv(f, skiprows=1)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def extract_gates_passed_from_monitor(monitor_dir, ctrl_freq=30, n_envs=8):
    """從 monitor CSV 提取 gates_passed 隨訓練步數的變化。

    Returns
    -------
    steps : np.ndarray
        累計訓練步數
    gates : np.ndarray
        每個 episode 的 gates_passed
    """
    df = read_monitor_csv(monitor_dir)
    if df is None or 'gates_passed' not in df.columns:
        return None, None

    gates = df['gates_passed'].values
    ep_lengths = df['l'].values

    # 計算累計步數
    cumulative_steps = np.cumsum(ep_lengths) * n_envs  # 乘以並行環境數
    return cumulative_steps, gates


def load_eval_results(log_dir):
    """讀取 EvalCallback 儲存的 evaluations.npz。

    Returns
    -------
    steps : np.ndarray
        評估時的訓練步數
    mean_rewards : np.ndarray
        每次評估的平均獎勵
    """
    npz_path = os.path.join(log_dir, 'evaluations.npz')
    if not os.path.exists(npz_path):
        return None, None

    data = np.load(npz_path)
    timesteps = data['timesteps']
    results = data['results']  # shape: (n_evals, n_eval_episodes)

    mean_rewards = np.mean(results, axis=1)
    return timesteps, mean_rewards


# ================================================================
# 指標計算函數
# ================================================================

def compute_auc(steps, values, max_steps=2000000, normalize=True):
    """計算學習曲線的 AUC (Area Under Curve)。

    Parameters
    ----------
    steps : np.ndarray
    values : np.ndarray
    max_steps : int
    normalize : bool
        是否除以 max_steps 做正規化

    Returns
    -------
    auc : float
    """
    if steps is None or values is None or len(steps) < 2:
        return np.nan

    # 確保在 max_steps 範圍內
    mask = steps <= max_steps
    s = steps[mask]
    v = values[mask]

    if len(s) < 2:
        return np.nan

    # 梯形積分
    auc = np.trapz(v, s)

    if normalize:
        auc /= max_steps

    return auc


def compute_ttt(steps, values, threshold):
    """計算 TTT (Time To Threshold)。

    Parameters
    ----------
    steps : np.ndarray
    values : np.ndarray
    threshold : float
        效能門檻

    Returns
    -------
    ttt : float or np.nan
        首次達到 threshold 所需的訓練步數
        若從未達到則回傳 np.nan
    """
    if steps is None or values is None or len(steps) == 0:
        return np.nan

    above = np.where(values >= threshold)[0]
    if len(above) == 0:
        return np.nan

    return steps[above[0]]


def smooth_curve(values, window=10):
    """滑動平均平滑化。"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


# ================================================================
# 分析流程
# ================================================================

def analyze_methods(scene_key, exp_type, thresholds=None, source='eval',
                    seeds=None, base_model_dir='models', base_log_dir='logs',
                    max_steps=2000000):
    """分析指定場景與實驗類型的 sample efficiency。

    Parameters
    ----------
    scene_key : str
        'standard', 'sharp', 'racing'
    exp_type : str
        'main', 'ablation', 'variant'
    thresholds : dict or None
        TTT 門檻字典 {label: value}，例如 {'50%': 36, '70%': 51, '90%': 65}
        None = 根據 best_final 自動計算 50%/70%/90%
    source : str
        'tensorboard', 'monitor', 'eval', 'auto'
    """
    if seeds is None:
        seeds = SEEDS

    config = SCENE_CONFIGS[scene_key]
    method_key = f'{exp_type}_methods'
    methods = config[method_key]

    # 計算多門檻
    if thresholds is None:
        best = config['best_final']
        thresholds = {f'{int(p*100)}%': round(best * p, 1) for p in TTT_PERCENTAGES}

    print(f"\n{'='*70}")
    print(f"  Sample Efficiency Analysis")
    print(f"  場景: {config['name']}")
    print(f"  實驗: {exp_type}")
    print(f"  TTT 門檻: {thresholds}")
    print(f"  Seeds: {seeds}")
    print(f"  資料來源: {source}")
    print(f"{'='*70}\n")

    results = {}

    for method_name, dir_pattern in methods.items():
        method_aucs = []
        method_ttts = {label: [] for label in thresholds}  # 每個門檻一個列表
        method_curves = []

        for seed in seeds:
            dir_name = dir_pattern.format(seed=seed)
            model_dir = os.path.join(base_model_dir, dir_name)
            log_dir = os.path.join(base_log_dir, dir_name)
            monitor_dir = os.path.join(model_dir, 'monitor')

            steps, values = None, None

            # 優先使用 evaluations.npz (無隨機噪聲的 Deterministic Reward)
            if source in ('eval', 'auto'):
                steps, values = load_eval_results(log_dir)

            if (steps is None) and source in ('tensorboard', 'auto') and HAS_TB:
                steps, values = read_tensorboard_scalar(
                    log_dir, tag='rollout/ep_rew_mean', max_steps=max_steps)

            if steps is not None and values is not None and len(steps) > 0:
                auc = compute_auc(steps, values, max_steps=max_steps)
                method_aucs.append(auc)
                method_curves.append((steps, values))

                # 對每個門檻計算 TTT
                for label, thresh_val in thresholds.items():
                    ttt = compute_ttt(steps, values, threshold=thresh_val)
                    method_ttts[label].append(ttt)
            else:
                print(f"  [SKIP] {method_name} seed={seed}: 找不到資料 ({model_dir})")

        if method_aucs:
            result_entry = {
                'auc_mean': np.nanmean(method_aucs),
                'auc_std': np.nanstd(method_aucs),
                'auc_sem': np.nanstd(method_aucs) / np.sqrt(np.sum(~np.isnan(method_aucs))),
                'n_seeds': len(method_aucs),
                'curves': method_curves,
                'thresholds': {},  # 多門檻 TTT 結果
            }
            for label, ttt_list in method_ttts.items():
                arr = np.array(ttt_list)
                n_valid = np.sum(~np.isnan(arr))
                result_entry['thresholds'][label] = {
                    'value': thresholds[label],
                    'ttt_mean': np.nanmean(arr),
                    'ttt_std': np.nanstd(arr),
                    'ttt_sem': np.nanstd(arr) / np.sqrt(n_valid) if n_valid > 0 else np.nan,
                    'reach_rate': np.mean(~np.isnan(arr)),
                }
            results[method_name] = result_entry
        else:
            print(f"  [ERROR] {method_name}: 所有 seeds 都沒有資料")

    return results


def print_results_table(results, scene_name, exp_type):
    """印出多門檻結果表格。"""
    if not results:
        print("  沒有結果可顯示。")
        return

    # 取得門檻列表 (從第一個方法)
    first_method = next(iter(results.values()))
    thresh_labels = list(first_method['thresholds'].keys())

    print(f"\n{'='*110}")
    print(f"  {scene_name} — {exp_type} — Sample Efficiency")
    thresh_info = ", ".join([f"{l}={first_method['thresholds'][l]['value']:.0f}" for l in thresh_labels])
    print(f"  TTT Thresholds: {thresh_info} (deterministic reward)")
    print(f"{'='*110}")

    # 表頭
    ttt_headers = "".join([f"{'TTT-'+l:<16s}" for l in thresh_labels])
    reach_headers = "".join([f"{'Reach-'+l:<12s}" for l in thresh_labels])
    header = f"  {'Method':<18s} {'AUC':<14s} {ttt_headers}{reach_headers}"
    print(header)
    print("  " + "-" * 106)

    # 按 AUC 排序
    sorted_methods = sorted(results.items(), key=lambda x: -x[1]['auc_mean'])

    for method_name, data in sorted_methods:
        auc_str = f"{data['auc_mean']:.1f}±{data['auc_sem']:.1f}"

        ttt_strs = ""
        reach_strs = ""
        for label in thresh_labels:
            td = data['thresholds'][label]
            if np.isnan(td['ttt_mean']):
                ttt_strs += f"{'N/A':<16s}"
            else:
                ttt_strs += f"{td['ttt_mean']/1000:.0f}k±{td['ttt_sem']/1000:.0f}k".ljust(16)
            reach_strs += f"{td['reach_rate']*100:.0f}%".ljust(12)

        print(f"  {method_name:<18s} {auc_str:<14s} {ttt_strs}{reach_strs}")

    print()


def save_results_csv(results, output_path, scene_name, exp_type):
    """將多門檻結果儲存為 CSV。"""
    rows = []
    for method_name, data in results.items():
        row = {
            'Scene': scene_name,
            'Experiment': exp_type,
            'Method': method_name,
            'AUC_Mean': data['auc_mean'],
            'AUC_Std': data['auc_std'],
            'AUC_SEM': data['auc_sem'],
            'N_Seeds': data['n_seeds'],
        }
        for label, td in data['thresholds'].items():
            row[f'TTT_{label}_Threshold'] = td['value']
            row[f'TTT_{label}_Mean'] = td['ttt_mean']
            row[f'TTT_{label}_Std'] = td['ttt_std']
            row[f'TTT_{label}_SEM'] = td['ttt_sem']
            row[f'TTT_{label}_ReachRate'] = td['reach_rate']
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  已儲存: {output_path}")


def plot_results(results, scene_name, exp_type, output_dir='figures'):
    """繪製 AUC 與多門檻 TTT 的比較圖。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib 未安裝，跳過繪圖")
        return

    os.makedirs(output_dir, exist_ok=True)

    methods = list(results.keys())
    n = len(methods)
    if n == 0:
        return

    first_data = next(iter(results.values()))
    thresh_labels = list(first_data['thresholds'].keys())
    n_thresh = len(thresh_labels)

    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))
    thresh_colors = ['#4CAF50', '#FF9800', '#F44336']  # 50%=綠, 70%=橙, 90%=紅

    # ---- 圖 1: AUC + 多門檻 TTT ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AUC 長條圖
    aucs = [results[m]['auc_mean'] for m in methods]
    auc_errs = [results[m]['auc_sem'] for m in methods]

    bars = axes[0].bar(range(n), aucs, yerr=auc_errs, capsize=4,
                       color=colors[:n], edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('AUC (Normalized)')
    axes[0].set_title(f'AUC Comparison — {scene_name}')

    for bar, val, err in zip(bars, aucs, auc_errs):
        if not np.isnan(val):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # 多門檻 TTT 分組長條圖
    bar_width = 0.8 / n_thresh
    x_base = np.arange(n)

    for t_idx, label in enumerate(thresh_labels):
        thresh_val = first_data['thresholds'][label]['value']
        ttts = []
        ttt_errs = []
        bar_colors = []

        for m in methods:
            td = results[m]['thresholds'][label]
            if np.isnan(td['ttt_mean']):
                ttts.append(0)
                ttt_errs.append(0)
                bar_colors.append('lightgray')
            else:
                ttts.append(td['ttt_mean'] / 1000)
                ttt_errs.append(td['ttt_sem'] / 1000)
                bar_colors.append(thresh_colors[t_idx % len(thresh_colors)])

        x_pos = x_base + t_idx * bar_width - (n_thresh - 1) * bar_width / 2
        bars = axes[1].bar(x_pos, ttts, bar_width * 0.9, yerr=ttt_errs,
                          capsize=3, color=bar_colors, edgecolor='black',
                          linewidth=0.3, label=f'{label} ({thresh_val:.0f}g)')

        # 標註 N/A
        for bar, m in zip(bars, methods):
            td = results[m]['thresholds'][label]
            if np.isnan(td['ttt_mean']):
                axes[1].text(bar.get_x() + bar.get_width()/2, 1,
                            '✗', ha='center', va='bottom', fontsize=7, color='red')

    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('TTT (×10³ steps)')
    axes[1].set_title(f'Time to Threshold (Multi) — {scene_name}')
    axes[1].legend(fontsize=8, title='Threshold')

    plt.tight_layout()
    fname = f'{output_dir}/sample_efficiency_{scene_name.replace(" ", "_")}_{exp_type}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存圖表: {fname}")

    # ---- 圖 2: 學習曲線 + 多門檻水平線 ----
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method_name in enumerate(methods):
        curves = results[method_name].get('curves', [])
        if not curves:
            continue

        common_steps = np.linspace(0, 2000000, 200)
        interpolated = []

        for steps, values in curves:
            if len(steps) > 1:
                interp_vals = np.interp(common_steps, steps, values)
                interpolated.append(interp_vals)

        if interpolated:
            arr = np.array(interpolated)
            mean_curve = np.mean(arr, axis=0)
            std_curve = np.std(arr, axis=0)

            ax.plot(common_steps / 1000, mean_curve, color=colors[i],
                   label=method_name, linewidth=1.5)
            ax.fill_between(common_steps / 1000,
                          mean_curve - std_curve,
                          mean_curve + std_curve,
                          color=colors[i], alpha=0.15)

    # 繪製多門檻水平線
    for t_idx, label in enumerate(thresh_labels):
        thresh_val = first_data['thresholds'][label]['value']
        ax.axhline(y=thresh_val, color=thresh_colors[t_idx % len(thresh_colors)],
                  linestyle='--', alpha=0.5, linewidth=1,
                  label=f'{label} = {thresh_val:.0f}')

    ax.set_xlabel('Timesteps (×10³)')
    ax.set_ylabel('Mean Evaluation Reward')
    ax.set_title(f'Learning Curves — {scene_name} ({exp_type})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{output_dir}/learning_curves_{scene_name.replace(" ", "_")}_{exp_type}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存學習曲線: {fname}")


# ================================================================
# 統計檢定
# ================================================================

def pairwise_ttest(results):
    """對 AUC 做成對 t 檢定。"""
    from scipy import stats

    methods = list(results.keys())
    n = len(methods)

    print(f"\n  Pairwise t-test on AUC:")
    print(f"  {'Method A':<20s} {'Method B':<20s} {'Δ AUC':<10s} {'p-value':<10s} {'Sig':<6s}")
    print("  " + "-" * 66)

    for i in range(n):
        for j in range(i+1, n):
            a_name = methods[i]
            b_name = methods[j]

            # 如果有原始 AUC 值就做 t-test
            a_auc = results[a_name]['auc_mean']
            b_auc = results[b_name]['auc_mean']
            delta = a_auc - b_auc

            # 近似 p-value (用 SEM)
            a_sem = results[a_name]['auc_sem']
            b_sem = results[b_name]['auc_sem']
            se_diff = np.sqrt(a_sem**2 + b_sem**2)

            if se_diff > 0:
                t_stat = delta / se_diff
                # 近似 df
                n_a = results[a_name]['n_seeds']
                n_b = results[b_name]['n_seeds']
                df = n_a + n_b - 2
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
            else:
                p_val = 1.0

            sig = ''
            if p_val < 0.001: sig = '***'
            elif p_val < 0.01: sig = '**'
            elif p_val < 0.05: sig = '*'
            else: sig = 'n.s.'

            print(f"  {a_name:<20s} {b_name:<20s} {delta:<+10.1f} {p_val:<10.4f} {sig:<6s}")


# ================================================================
# 主程式
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sample Efficiency 分析 (AUC + 多門檻 TTT)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--scene', type=str, default='all',
                        choices=['standard', 'sharp', 'racing', 'all'],
                        help='場景 (default: all)')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['main', 'ablation', 'variant', 'all'],
                        help='實驗類型 (default: all)')
    parser.add_argument('--source', type=str, default='auto',
                        choices=['tensorboard', 'monitor', 'eval', 'auto'],
                        help='資料來源 (default: auto)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seed 列表 (default: 0-9)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='模型根目錄 (default: models)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log 根目錄 (default: logs)')
    parser.add_argument('--output_dir', type=str, default='sample_efficiency_results',
                        help='輸出目錄 (default: sample_efficiency_results)')
    parser.add_argument('--no_plot', action='store_true',
                        help='不繪製圖表')
    parser.add_argument('--max_steps', type=int, default=2000000,
                        help='最大訓練步數 (default: 2000000)')
    parser.add_argument('--percentages', type=float, nargs='+', default=None,
                        help='自訂 TTT 百分比門檻 (default: 0.5 0.7 0.9)')
    args = parser.parse_args()

    # 自訂百分比
    if args.percentages:
        global TTT_PERCENTAGES
        TTT_PERCENTAGES = args.percentages

    os.makedirs(args.output_dir, exist_ok=True)

    scenes = ['standard', 'sharp', 'racing'] if args.scene == 'all' else [args.scene]
    exps = ['main', 'ablation', 'variant'] if args.exp == 'all' else [args.exp]
    seeds = args.seeds if args.seeds else SEEDS

    all_results = {}

    for scene in scenes:
        config = SCENE_CONFIGS[scene]

        # 自動計算多門檻
        best = config['best_final']
        thresholds = {f'{int(p*100)}%': round(best * p, 1) for p in TTT_PERCENTAGES}

        for exp in exps:
            print(f"\n{'#'*70}")
            print(f"  分析: {config['name']} — {exp}")
            print(f"  多門檻: {thresholds}")
            print(f"{'#'*70}")

            results = analyze_methods(
                scene_key=scene,
                exp_type=exp,
                thresholds=thresholds,
                source=args.source,
                seeds=seeds,
                base_model_dir=args.model_dir,
                base_log_dir=args.log_dir,
                max_steps=args.max_steps,
            )

            if results:
                # 印出表格
                print_results_table(results, config['name'], exp)

                # 儲存 CSV
                csv_path = os.path.join(args.output_dir,
                                       f'sample_efficiency_{scene}_{exp}.csv')
                save_results_csv(results, csv_path, config['name'], exp)

                # 統計檢定
                try:
                    pairwise_ttest(results)
                except ImportError:
                    print("  [SKIP] scipy 未安裝，跳過統計檢定")

                # 繪圖
                if not args.no_plot:
                    plot_results(results, config['name'], exp,
                              output_dir=args.output_dir)

                all_results[f'{scene}_{exp}'] = results

    # ---- 跨場景彙總表 ----
    if len(all_results) > 1:
        print(f"\n{'='*110}")
        print(f"  跨場景彙總")
        print(f"{'='*110}")

        summary_rows = []
        for key, results in all_results.items():
            scene, exp = key.rsplit('_', 1)
            for method, data in results.items():
                row = {
                    'Scene': scene,
                    'Experiment': exp,
                    'Method': method,
                    'AUC': f"{data['auc_mean']:.1f}±{data['auc_sem']:.1f}",
                }
                for label, td in data['thresholds'].items():
                    if np.isnan(td['ttt_mean']):
                        row[f'TTT_{label}'] = 'N/A'
                    else:
                        row[f'TTT_{label}'] = f"{td['ttt_mean']/1000:.0f}k"
                    row[f'Reach_{label}'] = f"{td['reach_rate']*100:.0f}%"
                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.output_dir, 'sample_efficiency_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  彙總表已儲存: {summary_path}")
        print(summary_df.to_string(index=False))

    print(f"\n{'='*70}")
    print(f"  分析完成！結果已儲存至: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()