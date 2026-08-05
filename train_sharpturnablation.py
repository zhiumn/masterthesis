"""
急彎走廊 Ablation 訓練腳本

使用方式：

  # 跑單一條件
  python train_sharpturnablation.py --condition A0 --seed 0

  # 跑全部 Exp 4a (Bonus 消融)
  python train_sharpturnablation.py --run_exp 4a --seeds 0 1 2 3 4

  # 跑全部 Exp 4b (Proximity 函數消融)
  python train_sharpturnablation.py --run_exp 4b --seeds 0 1 2 3 4

  # 跑全部
  python train_sharpturnablation.py --run_exp all --seeds 0 1 2 3 4

消融條件 (急彎場景):
──────────────────────────────────────────────────────────────
  Exp 4a — Bonus Term Ablation:
    A0:   純 R_base (無 bonus)                    → 下限基線
    A1:   只有 centering bonus (dynamic)          → centering 獨立貢獻
    A2:   只有 heading bonus (dynamic)            → heading 獨立貢獻
    A3:   只有 smooth bonus                       → smooth 獨立貢獻
    A7-S: 三個 bonus 全開, 固定權重               → Static baseline
    A7-D: 三個 bonus 全開, 動態權重               → Dynamic proposed

  Exp 4b — Proximity Function Ablation:
    P1 = A7-D: exponential exp(-0.8d)             → proposed
    P4:        linear max(0, 1-d/4)               → 低區分度
    P5:        step (d<1.5→1, else→0)             → 高區分度但不連續
    P6:        sigmoid 1/(1+exp(3(d-1.5)))        → 高區分度+平滑
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
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from gym_pybullet_drones.envs.SharpTurnCorridorAblation import SharpTurnCorridorAblation
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


# ================================================================
# 消融條件定義
# ================================================================

ABLATION_CONDITIONS = {

    # ============================================================
    # Exp 4a: Bonus Term Ablation
    # ============================================================

    'A0': dict(
        use_dynamic_reward=True,
        enable_centering=False,
        enable_heading=False,
        enable_smooth=False,
        proximity_type='exponential',
        description='Pure R_base only (no bonus) — lower bound',
    ),

    'A1': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=False,
        enable_smooth=False,
        proximity_type='exponential',
        description='Centering bonus only (dynamic)',
    ),

    'A2': dict(
        use_dynamic_reward=True,
        enable_centering=False,
        enable_heading=True,
        enable_smooth=False,
        proximity_type='exponential',
        description='Heading bonus only (dynamic)',
    ),

    'A3': dict(
        use_dynamic_reward=True,
        enable_centering=False,
        enable_heading=False,
        enable_smooth=True,
        proximity_type='exponential',
        description='Smooth bonus only',
    ),

    'A7-S': dict(
        use_dynamic_reward=False,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='exponential',
        description='Full Static (all bonus, fixed weights) — baseline',
    ),

    'A7-D': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='exponential',
        description='Full Dynamic (all bonus, proximity weights) — proposed = P1',
    ),

    # ============================================================
    # Exp 4b: Proximity Function Ablation
    # ============================================================

    'P4': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='linear',
        description='Linear proximity: max(0, 1-d/4) — low differentiation',
    ),

    'P5': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='step',
        description='Hard step proximity: d<1.5→1, else→0 — high diff, non-smooth',
    ),

    'P6': dict(
        use_dynamic_reward=True,
        enable_centering=True,
        enable_heading=True,
        enable_smooth=True,
        proximity_type='sigmoid',
        description='Sigmoid proximity: 1/(1+exp(3(d-1.5))) — high diff + smooth',
    ),
}

# 實驗分組
EXP_GROUPS = {
    '4a': ['A1', 'A2'], # 為了節省時間，將原本會重複訓練的 A7-S 和 A7-D 刪除，只需依賴之前跑好的 sharp_turn_static/dynamic 即可
    '4b': ['A7-D', 'P4', 'P5', 'P6'],  # 已將 A7-S 與 A7-D 移除，避免與 4a 平行執行時發生存取衝突
    'all': ['A0','A1', 'A2', 'A3','A7-S','A7-D', 'P4', 'P5', 'P6'],
}


def train_single_condition(condition_name, seed, timesteps=2000000, n_envs=8):
    """訓練單一消融條件。"""

    config = ABLATION_CONDITIONS[condition_name]

    exp_dir = f"sharp_ablation_{condition_name}_seed{seed}"
    models_dir = f"models/{exp_dir}"
    log_dir = f"logs/{exp_dir}"
    monitor_dir = f"{models_dir}/monitor"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    # 跳過已完成的
    if os.path.exists(f"{models_dir}/best_model.zip"):
        print(f"[SKIP] {condition_name} seed={seed} — already exists")
        return

    print("=" * 65)
    print(f"  Sharp Turn Ablation: {condition_name} | Seed: {seed}")
    print(f"  {config['description']}")
    print(f"  Bonus Scale: centering=1.5, heading=0.5, smooth=0.3") # 標準全對齊設定
    print(f"  dynamic={config['use_dynamic_reward']}, "
          f"C={config['enable_centering']}, "
          f"H={config['enable_heading']}, "
          f"S={config['enable_smooth']}, "
          f"prox={config['proximity_type']}")
    print("=" * 65)

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

    # 訓練環境
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    train_env = make_vec_env(
        SharpTurnCorridorAblation,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        seed=seed,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
    )
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True,
        clip_reward=10.0, gamma=0.995,
    )

    # 評估環境
    eval_env = make_vec_env(
        SharpTurnCorridorAblation,
        n_envs=1,
        env_kwargs=env_kwargs,
        seed=seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=False, norm_reward=False, gamma=0.995,
    )

    # PPO (所有條件完全相同的超參數)
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

    print(f"開始訓練 {condition_name} seed={seed}...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    elapsed = time.time() - start_time
    print(f"完成！耗時: {elapsed:.1f}s ({elapsed / 3600:.2f}h)")

    model.save(os.path.join(models_dir, "final_model"))
    train_env.save(os.path.join(models_dir, "vec_normalize.pkl"))

    train_env.close()
    eval_env.close()
    del model


def main():
    parser = argparse.ArgumentParser(
        description='急彎走廊 Ablation 訓練',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--condition', type=str, default=None,
                        choices=list(ABLATION_CONDITIONS.keys()),
                        help='單一消融條件')
    parser.add_argument('--run_exp', type=str, default=None,
                        choices=['4a', '4b', 'all'],
                        help='批次跑: 4a (bonus), 4b (proximity), all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Seed 列表 (default: 0 1 2 3 4)')
    parser.add_argument('--timesteps', type=int, default=2000000,
                        help='訓練步數 (default: 2000000)')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='平行環境數 (default: 8)')
    args = parser.parse_args()

    if args.condition is None and args.run_exp is None:
        parser.print_help()
        print("\n可用條件:")
        for name, config in ABLATION_CONDITIONS.items():
            print(f"  {name:6s}: {config['description']}")
        print("\n可用實驗組:")
        for group, conditions in EXP_GROUPS.items():
            print(f"  {group:4s}: {', '.join(conditions)}")
        print("\n建議流程:")
        print("  1. python train_sharpturnablation.py --run_exp 4a --seeds 0 1 2 3 4")
        print("  2. python train_sharpturnablation.py --run_exp 4b --seeds 0 1 2 3 4")
        return

    # 決定要跑的條件
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = EXP_GROUPS[args.run_exp]
        conditions = list(dict.fromkeys(conditions))  # 去重

    total_runs = len(conditions) * len(args.seeds)

    print(f"\n{'=' * 65}")
    print(f"  急彎走廊 Ablation")
    print(f"  場景: dist U(1.8,5.0) | yaw ±34° | z ±0.8m")
    print(f"  Bonus Scale: centering=1.5, heading=0.5, smooth=0.3")
    print(f"  條件: {', '.join(conditions)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  總 runs: {total_runs}")
    print(f"{'=' * 65}\n")

    completed = 0
    for condition_name in conditions:
        for seed in args.seeds:
            completed += 1
            print(f"\n[{completed}/{total_runs}]")
            train_single_condition(
                condition_name, seed,
                timesteps=args.timesteps,
                n_envs=args.n_envs,
            )

    print(f"\n{'=' * 65}")
    print(f"  全部完成！共 {completed} runs")
    print(f"  模型: models/sharp_ablation/")
    print(f"  TensorBoard: tensorboard --logdir logs/sharp_ablation/")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()