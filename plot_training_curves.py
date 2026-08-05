import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def load_results(seed_dir):
    """
    從 monitor.csv 中讀取穩定的過程數據。
    """
    monitor_files = glob.glob(os.path.join(seed_dir, 'monitor', '*.csv'))
    if not monitor_files:
        return None
        
    df_list = []
    num_envs = len(monitor_files)
    for f in monitor_files:
        # Stable Baselines3 的 monitor.csv 第一行是 JSON meta，第二行才是標題，故跳過1行
        #df = pd.read_csv(f, skiprows=1)
        df = pd.read_csv(
            f,
            skiprows=1,
            on_bad_lines="skip",
            engine="python"
        )

        # 🔥 關鍵：強制所有欄位轉數值
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # 🔥 丟掉轉換失敗的行
        df = df.dropna()
        # 各環境獨立計算步數，再乘以總環境數作為該組數據採樣當下的 global_step
        df['total_steps'] = df['l'].cumsum() * num_envs
        df_list.append(df)
        
    # 將平行環境各自分別計算好時間軸的資料合併，然後按時間先後排序
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values(by='total_steps').reset_index(drop=True)
    return df_all

def main():
    import argparse
    parser = argparse.ArgumentParser(description='繪製訓練學習曲線')
    parser.add_argument('--filter', type=str, nargs='+', default=[], help='只處理檔名包含這些字串的實驗 (例: --filter ablation 或 --filter SAC TD3 Baseline)')
    args = parser.parse_args()

    base_dir = "models"
    # 尋找所有底下的資料夾 (只要名稱包含 _seed 就算數)
    all_seed_dirs = glob.glob(os.path.join(base_dir, "*_seed*"))
    
    seed_dirs = []
    if args.filter:
        for s_dir in all_seed_dirs:
            exp_name = os.path.basename(s_dir).rsplit("_seed", 1)[0]
            if any(exp_name == f or exp_name.startswith(f + '_') or exp_name.startswith(f + '-') for f in args.filter):
                seed_dirs.append(s_dir)
    else:
        seed_dirs = all_seed_dirs
    
    if not seed_dirs:
        print("未找到任何含有 '_seed' 的資料夾。")
        return

    # 將資料夾按實驗名稱分組 (例如 'PPO_InfiniteGate_L1' 是一組, 'PPO_InfiniteGate_L2' 是一組)
    experiments = {}
    
    for s_dir in seed_dirs:
        dir_name = os.path.basename(s_dir)
        # 取出 _seed 前面的字串當作實驗組別名稱
        exp_name = dir_name.rsplit("_seed", 1)[0]
        if exp_name not in experiments:
            experiments[exp_name] = []
            
        df = load_results(s_dir)
        if df is not None:
            experiments[exp_name].append((dir_name, df))
            print(f"✅ 已載入: {dir_name} (共 {len(df)} 筆)")

    if not any(experiments.values()):
        print("未在資料夾內發現 monitor.csv，請確保訓練有正常運作。")
        return

    # 將資料切割到固定大小的 Step Bin 進行平均
    bin_size = 10000
    metrics = {
        'r': ('Mean Reward', 'Reward'),
        'gates_passed': ('Mean Gates Passed', 'Gates Passed'),
        'collision_rate': ('Collision Rate', 'Collision Rate'),
        'avg_jerk': ('Average Jerk(m/s³)', 'Avg Jerk (m/s³)'),
        'avg_pass_dist': ('Average Distance to Gate Center (m)', 'Avg Distance to Gate Center (m)')
    }

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

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

    legend_mapping = {}
    folder_name_parts = []
    
    for exp_name in experiments.keys():
        leg = get_short_name(exp_name)
        legend_mapping[exp_name] = leg
        
        safe_part = leg.replace("+", "").replace("-", "_")
        if safe_part not in folder_name_parts:
            folder_name_parts.append(safe_part)

    # ==========================
    # Legend 順序控制
    # ==========================
    master_order = [
        'Base Only', 
        'PBRS (shared)', 
        'Additive-Fixed', 
        '+Centering', 
        '+Heading', 
        '+Smooth', 
        'Additive-Exp', 
        'Additive-Lin', 
        'Additive-Step', 
        'Additive-Sig'
    ]

    def sort_key(item):
        short = legend_mapping.get(item[0], item[0])
        try:
            return master_order.index(short)
        except ValueError:
            return 999
            
    experiments = dict(sorted(experiments.items(), key=sort_key))


    # 建立 result/對比較量 名稱的子資料夾
    # 🕵️ 偵測場景前綴，避免不同場景互相覆蓋
    scenario_prefixes = ['baseline', 'sharp', 'racing']
    detected_scenarios = set()
    for exp_raw in experiments.keys():
        low_exp = exp_raw.lower()
        matched = False
        for sp in scenario_prefixes:
            if sp in low_exp:
                detected_scenarios.add(sp)
                matched = True
                break
        if not matched:
            detected_scenarios.add("infinite") # 預設場景
    
    scenario_prefix = "_".join(sorted(list(detected_scenarios)))
    subfolder_name = "-vs-".join(folder_name_parts)
    if scenario_prefix:
        subfolder_name = f"{scenario_prefix}_{subfolder_name}"
        
    if not subfolder_name: 
        subfolder_name = "Training_Results"
        
    save_dir = os.path.join("result", subfolder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📁 圖表準備輸出至實體分類資料夾: {save_dir}/")

    # ==========================
    # 開始繪製圖表 (包含總表與獨立 5 張)
    # ==========================
    fig_big, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig_big.suptitle('Learning Curves Comparison', fontsize=18, fontweight='bold', y=1.02)
    axes_flat = axes.flatten()
    axes_flat[5].set_visible(False) # 隱藏第6格
    
    exp_colors = ['#4A90E2', '#E24A4A', '#7ED321', '#F5A623', '#9B59B6']

    for i, (m, (title, ylabel)) in enumerate(metrics.items()):
        
        # 開啟一張獨立的 Figure 準備存給論文單點插入用
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        
        # 同時也抓取大總表的 Ax
        ax_big = axes_flat[i]
        
        for idx, (exp_name, seed_data_list) in enumerate(experiments.items()):
            if not seed_data_list:
                continue
            legend_name = legend_mapping[exp_name]
            color = exp_colors[idx % len(exp_colors)]
            
            # 取得該實驗組所有 df 並計算資料範圍
            dfs = [item[1] for item in seed_data_list]
            max_steps = max(df['total_steps'].max() for df in dfs)
            step_bins = np.arange(0, max_steps + bin_size, bin_size)

            interpolated_data = []
            for df in dfs:
                if m not in df.columns:
                    continue
                # 滾動平均算分桶
                df['bins'] = pd.cut(df['total_steps'], bins=step_bins)
                binned = df.groupby('bins')[m].mean()
                binned = binned.ffill().bfill()
                interpolated_data.append(binned.values)
                
            if not interpolated_data:
                continue
                
            data_matrix = np.array(interpolated_data)
            mean = np.mean(data_matrix, axis=0)
            std = np.std(data_matrix, axis=0)
            
            # 除以 1,000 以符合論文 (10^3) 單位需求
            x_axis = np.array(step_bins[1:]) / 1e3
            
            min_len = min(len(x_axis), len(mean))
            x_axis = x_axis[:min_len]
            mean = mean[:min_len]
            std = std[:min_len]

            # >>> 在大圖上繪製 <<<
            ax_big.plot(x_axis, mean, color=color, linewidth=2.5, label=legend_name)
            ax_big.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.08)
            if len(x_axis) > 0:
                ax_big.plot(x_axis[-1], mean[-1], marker='o', markersize=6, color=color)
                
            # >>> 在單次獨立圖上繪製 <<<
            ax_single.plot(x_axis, mean, color=color, linewidth=2.5, label=legend_name)
            ax_single.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.08)
            if len(x_axis) > 0:
                ax_single.plot(x_axis[-1], mean[-1], marker='o', markersize=6, color=color)
                
        # 共同設定大圖的排版
        ax_big.set_title(title, fontsize=14, fontweight='bold')
        ax_big.set_xlabel('Timesteps ($x10^3$)', fontsize=12)
        ax_big.set_ylabel(ylabel, fontsize=12)
        ax_big.set_axisbelow(True)
        ax_big.grid(True, linestyle='--', color='#D3D3D3', alpha=1.0)
        # 固定 y 軸範圍，確保跨場景可比較
        if m == 'avg_pass_dist':
            ax_big.set_ylim(0.0, 0.6)
        elif m == 'gates_passed':
            ax_big.set_ylim(0, 70)
        elif m == 'avg_jerk':
            ax_big.set_ylim(15, 30)
        if i == len(metrics) - 1: # 全部疊加畫完最後一筆時再顯示圖例
            ax_big.legend(loc='best', fontsize=10)
            
        # 設定獨立圖的排版並單獨存檔
        ax_single.set_title(title, fontsize=14, fontweight='bold')
        ax_single.set_xlabel('Timesteps ($x10^3$)', fontsize=12)
        ax_single.set_ylabel(ylabel, fontsize=12)
        ax_single.set_axisbelow(True)
        ax_single.grid(True, linestyle='--', color='#D3D3D3', alpha=1.0)
        # 固定 y 軸範圍，確保跨場景可比較
        if m == 'avg_pass_dist':
            ax_single.set_ylim(0.0, 0.6)
        elif m == 'gates_passed':
            ax_single.set_ylim(0, 70)
        elif m == 'avg_jerk':
            ax_single.set_ylim(15, 30)
        ax_single.legend(loc='best', fontsize=10) # 獨立圖每一張都要有圖例
        
        # 確保單獨存檔不越界並關閉避免記憶體溢出
        fig_single.tight_layout()
        single_path = os.path.join(save_dir, f"{m}.png")
        fig_single.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"  └─ 獨立圖表儲存為: {m}.png")
        
    # 最後對大總表存檔
    fig_big.tight_layout()
    big_path = os.path.join(save_dir, "comparative_training_curves_paper.png")
    fig_big.savefig(big_path, dpi=300, bbox_inches='tight')
    plt.close(fig_big)
    print(f"  └─ 整體面板儲存為: comparative_training_curves_paper.png")
    
    print("\n[SUCCESS] 一鍵化資料搜集、比對分組與獨立圖片輸出全數搞定！")

if __name__ == '__main__':
    main()
