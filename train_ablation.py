"""
Ablation Study Training Script — Exp 4a & 4b

使用方式：

  # 跑單一條件 (手動指定)
  python train_ablation.py --condition A0 --seed 0
  python train_ablation.py --condition A1 --seed 0
  python train_ablation.py --condition P5 --seed 0

  # 跑全部 Exp 4a 條件 (所有 bonus 消融)
  python train_ablation.py --run_exp 4a --seeds 0 1 2

  # 跑全部 Exp 4b 條件 (所有 proximity 函數消融)
  python train_ablation.py --run_exp 4b --seeds 0 1 2

  # 跑全部消融實驗
  python train_ablation.py --run_exp all --seeds 0 1 2 3 4

所有條件對照表：
──────────────────────────────────────────────────────────────
Exp 4a — Bonus Term Ablation:
  A0:   純 R_base，不加任何 bonus               (下限基線)
  A1:   只加 centering bonus (dynamic)          (centering 的獨立貢獻)
  A2:   只加 heading bonus (dynamic)            (heading 的獨立貢獻)
  A3:   只加 smooth bonus (dynamic)             (smooth 的獨立貢獻)
  A7-S: 三個 bonus 全開, 固定權重 (static)       (你已有的 Static baseline)
  A7-D: 三個 bonus 全開, 動態權重 (dynamic)      (你已有的 Dynamic proposed)

Exp 4b — Proximity Function Ablation:
  P1:   exponential exp(-0.8d)  (proposed)       (你已有的 A7-D)
  P4:   linear max(0, 1-d/4)                    (平滑但衰減恆定)
  P5:   hard step (d<1.5 → 1, else → 0)         (不連續跳變)
  P6:   sigmoid (k=3, d_mid=1.5)                (高區分度+平滑)
──────────────────────────────────────────────────────────────
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import argparse
import subprocess
import sys
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from gym_pybullet_drones.envs.InfiniteGateAviaryAblation import InfiniteGateAviaryAblation
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


# ================================================================
# 消融條件定義
# ================================================================

ABLATION_CONDITIONS = {
    # ============================================================
    # Exp 4a: Bonus Term Ablation
    # ============================================================

    # A0: 純 R_base，無任何 bonus
    # 目的: 建立下限基線，測量 bonus shaping 的整體改善幅度
    'A0': dict(
        use_dynamic_reward=True,    # 此處無意義 (因為所有 bonus 都關了)
        enable_centering=False,
        enable_heading=False,
        enable_smooth=False,
        proximity_type='exponential',
        description='Pure R_base only (no bonus)',
    ),

    # A1: 只開 centering bonus (使用 dynamic 權重)
    # 目的: 測量 centering bonus 的獨立貢獻
    'A1': dict(
        use_dynamic_reward=True,
        enable_centering=True,      # ← 只開這個
        enable_heading=False,
        enable_smooth=False,
        proximity_type='exponential',
        description='Centering bonus only (dynamic)',
    ),

    # A2: 只開 heading bonus (使用 dynamic 權重)
    # 目的: 測量 heading bonus 的獨立貢獻
    'A2': dict(
        use_dynamic_reward=True,
        enable_centering=False,
        enable_heading=True,        # ← 只開這個
        enable_smooth=False,
        proximity_type='exponential',
        description='Heading bonus only (dynamic)',
    ),

    # A3: 只開 smooth bonus (兩組權重都是 1.0，所以 dynamic/static 無差)
    # 目的: 測量 smooth bonus 的獨立貢獻
    'A3': dict(
        use_dynamic_reward=True,
        enable_centering=False,
        enable_heading=False,
        enable_smooth=True,         # ← 只開這個
        proximity_type='exponential',
        description='Smooth bonus only (dynamic)',
    ),

    # A7-S: 三個 bonus 全開，固定權重 (Static baseline)
    # 目的: 完整的 Static 方法，與 A7-D 構成核心對照
    'A7-S': dict(
        use_dynamic_reward=False,   # ← Static
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='exponential',
        description='Full Static (all bonus, fixed weights)',
    ),

    # A7-D: 三個 bonus 全開，動態權重 (Dynamic proposed)
    # 目的: 完整的 Dynamic 方法 = P1
    'A7-D': dict(
        use_dynamic_reward=True,    # ← Dynamic
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='exponential',
        description='Full Dynamic (all bonus, proximity weights)',
    ),

    # ============================================================
    # Exp 4b: Proximity Function Ablation
    # ============================================================
    # 注意: P1 就是 A7-D (exponential)，不需要重複跑

    # P4: 線性衰減 proximity
    # 目的: 測試「也是 smooth 但形狀不同」是否一樣好
    'P4': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='linear',    # ← 線性衰減
        description='Linear proximity: max(0, 1-d/4)',
    ),

    # P5: Hard step proximity
    # 目的: 測試「不連續跳變」是否導致失敗 (核心消融)
    'P5': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='step',      # ← 硬切換
        description='Hard step proximity: d<1.5→1, else→0',
    ),

    # P6: Sigmoid proximity
    'P6': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='sigmoid',
        description='Sigmoid proximity: 1/(1+exp(3*(d-1.5)))',
    ),
}

# 實驗分組
EXP_GROUPS = {
    '4a': ['A1', 'A2'],
    '4b': ['A7-D', 'P4', 'P5', 'P6'],  # A7-D = P1 (exponential)
    'all': ['A0','A1', 'A2','A3', 'A7-S', 'A7-D', 'P4', 'P5', 'P6'],
}


def train_single_condition(condition_name, seed, timesteps=2000000, n_envs=8):
    """訓練單一消融條件。"""

    config = ABLATION_CONDITIONS[condition_name]

    # 建立資料夾
    exp_dir = f"ablation_{condition_name}_seed{seed}"
    models_dir = f"models/{exp_dir}"
    log_dir = f"logs/{exp_dir}"
    monitor_dir = f"{models_dir}/monitor"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    print("=" * 65)
    print(f"  Ablation: {condition_name} | Seed: {seed}")
    print(f"  {config['description']}")
    print(f"  dynamic={config['use_dynamic_reward']}, "
          f"centering={config['enable_centering']}, "
          f"heading={config['enable_heading']}, "
          f"smooth={config['enable_smooth']}, "
          f"proximity={config['proximity_type']}")
    print("=" * 65)

    # 環境參數 (從 config 中取出，排除 description)
    env_kwargs = dict(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2,
        use_dynamic_reward=config['use_dynamic_reward'],
        enable_centering=config['enable_centering'],
        enable_heading=config['enable_heading'],
        enable_smooth=config['enable_smooth'],
        proximity_type=config['proximity_type'],
    )

    monitor_kwargs = dict(
        info_keywords=('gates_passed', 'collision_rate', 'avg_jerk', 'avg_pass_dist')
    )

    # 建立訓練環境
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    train_env = make_vec_env(
        InfiniteGateAviaryAblation,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        seed=seed,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
    )

    # VecNormalize (只 normalize reward)
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.995,
    )

    # 建立評估環境
    eval_env = make_vec_env(
        InfiniteGateAviaryAblation,
        n_envs=1,
        env_kwargs=env_kwargs,
        seed=seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        gamma=0.995,
    )

    # PPO 模型 (所有條件完全相同的超參數)
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        seed=seed,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    # 訓練
    print(f"開始訓練 {condition_name} seed={seed}...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    elapsed = time.time() - start_time
    print(f"完成！耗時: {elapsed:.1f}s ({elapsed/3600:.2f}h)")

    # 儲存
    model.save(os.path.join(models_dir, "final_model"))
    train_env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    print(f"已儲存至: {models_dir}/")

    train_env.close()
    eval_env.close()
    del model

    return models_dir


def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study 訓練腳本 (Exp 4a: Bonus Term / Exp 4b: Proximity Function)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--condition', type=str, default=None,
                        choices=list(ABLATION_CONDITIONS.keys()),
                        help='單一消融條件名稱 (e.g., A0, A1, P4, P5)')
    parser.add_argument('--run_exp', type=str, default=None,
                        choices=['4a', '4b', 'all'],
                        help='批次跑整組實驗: 4a (bonus ablation), 4b (proximity ablation), all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='要跑的 seed 列表 (default: 0 1 2)')
    parser.add_argument('--timesteps', type=int, default=2000000,
                        help='每個條件的訓練步數 (default: 2000000)')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='平行環境數 (default: 8)')
    args = parser.parse_args()

    if args.condition is None and args.run_exp is None:
        parser.print_help()
        print("\n請指定 --condition 或 --run_exp")
        print("\n可用條件:")
        for name, config in ABLATION_CONDITIONS.items():
            print(f"  {name:6s}: {config['description']}")
        print("\n可用實驗組:")
        for group, conditions in EXP_GROUPS.items():
            print(f"  {group:4s}: {', '.join(conditions)}")
        return

    # 決定要跑哪些條件
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = EXP_GROUPS[args.run_exp]
        # 去重 (例如 4b 包含 A7-D，如果之前 4a 已跑過可跳過)
        conditions = list(dict.fromkeys(conditions))

    # 計算總量
    total_runs = len(conditions) * len(args.seeds)
    print(f"\n{'='*65}")
    print(f"  消融實驗計劃")
    print(f"  條件: {', '.join(conditions)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  總訓練 runs: {total_runs}")
    print(f"  每 run: {args.timesteps/1e6:.1f}M steps")
    print(f"{'='*65}\n")

    completed = 0
    for condition_name in conditions:
        for seed in args.seeds:

            # 檢查是否已跑過 (如果 best_model.zip 存在就跳過)
            exp_dir = f"ablation_{condition_name}_seed{seed}"
            best_model_path = f"models/{exp_dir}/best_model.zip"
            if os.path.exists(best_model_path):
                print(f"[SKIP] {condition_name} seed={seed} — 已存在 {best_model_path}")
                completed += 1
                continue

            print(f"\n[{completed+1}/{total_runs}] Training {condition_name} seed={seed}")
            train_single_condition(
                condition_name,
                seed,
                timesteps=args.timesteps,
                n_envs=args.n_envs,
            )
            completed += 1

    print(f"\n{'='*65}")
    print(f"  全部完成！共 {completed} 個 runs")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()