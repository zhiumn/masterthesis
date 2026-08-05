"""
Infinite Gate Aviary Hybrid 訓練腳本（基準場景）

Hybrid 方法：R_base (Additive) + Bonus (PBRS)

使用方式：

  # 先驗測試
  python train_infinite_gate_hybrid.py --pilot --seeds 0 1

  # 完整實驗
  python train_infinite_gate_hybrid.py --run_all --seeds 0 1 2 3 4 5 6 7 8 9

  # 單一 seed
  python train_infinite_gate_hybrid.py --seed 0

公平比較設計：
  ┌─────────────────────────────────────────────────────────────┐
  │  方法        │ R_base              │ Bonus                  │
  │─────────────────────────────────────────────────────────────│
  │  Additive    │ R_acquire + R_align │ f(s)                   │
  │  Hybrid      │ R_acquire + R_align │ γ·Φ(s') - Φ(s) (PBRS)  │
  └─────────────────────────────────────────────────────────────┘
  
  Hybrid vs Additive: 相同 R_base，只有 bonus 形式不同
  → 這是最公平的 PBRS vs Additive 比較

基準場景參數：
  - 門框間距: U(2.5, 4.0) m
  - 偏航變化: ±20° (±0.35 rad)
  - 高度變化: ±0.5 m
  - 難度: Easy
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

from gym_pybullet_drones.envs.InfiniteGateAviaryhybrid import InfiniteGateAviaryHybrid
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def train_single(seed, timesteps=2000000, n_envs=8):
    """訓練單一 Hybrid 配置（基準場景）。"""

    exp_name = f"baseline_hybrid_seed{seed}"
    models_dir = f"models/{exp_name}"
    log_dir = f"logs/{exp_name}"
    monitor_dir = f"{models_dir}/monitor"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    if os.path.exists(f"{models_dir}/best_model.zip"):
        print(f"[SKIP] {exp_name} — already exists")
        return

    print("=" * 70)
    print(f"  Infinite Gate Aviary Hybrid Training (基準場景)")
    print(f"  Method: R_base (Additive) + Bonus (PBRS)")
    print(f"  ")
    print(f"  場景參數:")
    print(f"    - 門框間距: U(2.5, 4.0) m")
    print(f"    - 偏航變化: ±20° (±0.35 rad)")
    print(f"    - 高度變化: ±0.5 m")
    print(f"    - 難度: Easy")
    print(f"  ")
    print(f"  R_base = 0.5 * R_acquire + 0.5 * R_align  (same as Additive)")
    print(f"  Bonus  = γ·Φ(s') - Φ(s)                   (PBRS form)")
    print(f"  ")
    print(f"  PBRS weights: centering=2.0, heading=1.5, smooth=1.0")
    print(f"  Seed: {seed}")
    print("=" * 70)

    env_kwargs = dict(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2,
        # PBRS Bonus 權重
        pbrs_centering_weight=2.0,
        pbrs_heading_weight=1.5,
        pbrs_smooth_weight=1.0,
        pbrs_gamma=0.995,
        gate_half_width=0.5,
    )

    monitor_kwargs = dict(
        info_keywords=('gates_passed', 'collision_rate', 'avg_jerk', 'avg_pass_dist')
    )

    train_env = make_vec_env(
        InfiniteGateAviaryHybrid,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
        seed=seed,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
    )
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True,
        clip_reward=10.0, gamma=0.995,
    )

    eval_env = make_vec_env(
        InfiniteGateAviaryHybrid, n_envs=1,
        env_kwargs=env_kwargs, seed=seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=False, norm_reward=False, gamma=0.995,
    )

    # PPO 超參數（與 InfiniteGateAviary / Additive 完全相同）
    model = PPO(
        'MlpPolicy', train_env,
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
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
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

    print(f"開始訓練 {exp_name}...")
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
        description='Infinite Gate Aviary Hybrid 訓練（基準場景）',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--pilot', action='store_true',
                        help='先驗測試: 只跑 2 seeds')
    parser.add_argument('--run_all', action='store_true',
                        help='完整實驗: 所有 seeds')
    parser.add_argument('--timesteps', type=int, default=2000000)
    parser.add_argument('--n_envs', type=int, default=8)
    args = parser.parse_args()

    seeds_to_run = []

    if args.pilot:
        seeds_to_run = args.seeds[:2] if len(args.seeds) >= 2 else [0, 1]
        print(f"\n先驗測試模式: Hybrid (基準場景) × {len(seeds_to_run)} seeds")

    elif args.run_all:
        seeds_to_run = args.seeds
        print(f"\n完整實驗: Hybrid (基準場景) × {len(seeds_to_run)} seeds")

    elif args.seed is not None:
        seeds_to_run = [args.seed]

    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Hybrid 方法說明（基準場景）:")
        print("=" * 70)
        print("  R_base: R_acquire + R_align (與 Additive 完全相同)")
        print("  Bonus:  PBRS 形式 γ·Φ(s') - Φ(s)")
        print("")
        print("  這是最公平的 PBRS vs Additive 比較：")
        print("  - 相同的 R_base")
        print("  - 相同的 observation space")
        print("  - 相同的 PPO 超參數")
        print("  - 唯一差異：bonus 的計算形式")
        print("")
        print("基準場景參數:")
        print("  - 門框間距: U(2.5, 4.0) m")
        print("  - 偏航變化: ±20° (±0.35 rad)")
        print("  - 高度變化: ±0.5 m")
        print("  - 回合時限: 60 s")
        print("  - 難度: Easy")
        print("")
        print("建議流程:")
        print("  1. python train_infinite_gate_hybrid.py --pilot --seeds 0 1")
        print("  2. python train_infinite_gate_hybrid.py --run_all --seeds 0 1 2 3 4 5 6 7 8 9")
        return

    total = len(seeds_to_run)
    for i, seed in enumerate(seeds_to_run):
        print(f"\n[{i+1}/{total}]")
        train_single(seed, args.timesteps, args.n_envs)

    print(f"\n{'=' * 70}")
    print(f"全部完成！共 {total} runs")
    print(f"模型位置: models/baseline_hybrid_seed*/")
    print(f"TensorBoard: tensorboard --logdir logs/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()