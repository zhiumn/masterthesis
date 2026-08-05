"""
場景 C: Racing Track 訓練腳本

使用方式：

  # Step 1: 先驗測試 (2 seeds，確認難度合適)
  python train_racingtrack.py --pilot --seeds 0 1

  # Step 2: 如果結果符合預期，擴展到完整實驗
  python train_racingtrack.py --run_all --seeds 0 1 2 3 4 5 6 7 8 9

  # 單一配置
  python train_racingtrack.py --reward dynamic --seed 0
  python train_racingtrack.py --reward static --seed 0

先驗測試的成功標準：
  - 能完成至少 1 圈 (12 gates) → 場景難度合適
  - Dynamic 應該與 Static 有差異 (不論誰優)
  - 如果平均 < 6 gates → 場景太難，考慮增加 turn_radius
  - 如果平均 > 36 gates (3 圈) → 場景太簡單，考慮減少 turn_radius

場景特性：
  - 膠囊形封閉賽道 (2 直道 + 2 U-Turn)
  - 每圈 12 gates
  - 連續曲率變化 (vs Sharp-Turn 的離散轉折)
  - 預期挑戰: Smooth bonus 應更重要
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

from gym_pybullet_drones.envs.RacingTrack import RacingTrack
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def train_single(reward_type, seed, timesteps=2000000, n_envs=8):
    """訓練單一 Racing Track 配置。"""

    use_dynamic = (reward_type == 'dynamic')

    exp_name = f"racing_track_{reward_type}_seed{seed}"

    models_dir = f"models/{exp_name}"
    log_dir = f"logs/{exp_name}"
    monitor_dir = f"{models_dir}/monitor"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    if os.path.exists(f"{models_dir}/best_model.zip"):
        print(f"[SKIP] {exp_name} — already exists")
        return

    print("=" * 65)
    print(f"  Scene C: Racing Track (Closed Circuit)")
    print(f"  Track: Pill-shaped Medium | L=10m | R=2.5m | V=0.4m | 12 gates/lap | w=0.4m")
    print(f"  Structure: 2 straights + 2 U-Turns (180° each)")
    print(f"  Reward: {reward_type} | Seed: {seed}")
    print("=" * 65)

    env_kwargs = dict(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2,
        use_dynamic_reward=use_dynamic,
        # Track 參數
        straight_length=10.0,
        turn_radius=2.5,
        n_gates_straight=3,
        n_gates_turn=3,
        gate_height=1.0,
        height_variation=0.4,
        # Reward 參數
        gate_half_width=0.4,
        gate_half_height=0.4,
        heading_bonus_scale=0.5,
        centering_bonus_scale=1.5,
    )

    monitor_kwargs = dict(
        info_keywords=('gates_passed', 'laps_completed', 'collision_rate',
                       'avg_jerk', 'avg_pass_dist')
    )

    train_env = make_vec_env(
        RacingTrack,
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
        RacingTrack, n_envs=1,
        env_kwargs=env_kwargs, seed=seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=False, norm_reward=False, gamma=0.995,
    )

    # PPO 超參數 (與其他場景完全相同)
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
        description='場景 C: Racing Track 訓練',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--reward', type=str, default=None,
                        choices=['dynamic', 'static'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--pilot', action='store_true',
                        help='先驗測試: 只跑 dynamic + static 各 2 seeds')
    parser.add_argument('--run_all', action='store_true',
                        help='完整實驗: dynamic + static × 所有 seeds')
    parser.add_argument('--timesteps', type=int, default=2000000)
    parser.add_argument('--n_envs', type=int, default=8)
    args = parser.parse_args()

    configs = []

    if args.pilot:
        pilot_seeds = args.seeds[:2] if len(args.seeds) >= 2 else [0, 1]
        for reward in ['dynamic', 'static']:
            for seed in pilot_seeds:
                configs.append((reward, seed))
        print(f"\n先驗測試模式: 2 rewards × {len(pilot_seeds)} seeds = {len(configs)} runs")
        print(f"成功標準: 能完成 ≥1 圈 (12 gates)，且 Dynamic/Static 有差異\n")

    elif args.run_all:
        for reward in ['dynamic', 'static']:
            for seed in args.seeds:
                configs.append((reward, seed))

    elif args.reward:
        if args.seed is not None:
            configs.append((args.reward, args.seed))
        else:
            for seed in args.seeds:
                configs.append((args.reward, seed))

    else:
        parser.print_help()
        print("\n" + "=" * 65)
        print("Racing Track 場景特性:")
        print("  - 膠囊形封閉極限賽道 (Hardcore)")
        print("  - 2 直道 (12m) + 2 U-Turn (R=2m)")
        print("  - 每圈 12 gates")
        print("  - 連續曲率變化 (vs Sharp-Turn 的離散轉折)")
        print("=" * 65)
        print("\n建議流程:")
        print("  1. python train_racing_track.py --pilot --seeds 0 1")
        print("  2. 檢查結果 (應能完成 ≥1 圈)")
        print("  3. python train_racing_track.py --run_all --seeds 0 1 2 3 4 5 6 7 8 9")
        return

    total = len(configs)
    for i, (reward, seed) in enumerate(configs):
        print(f"\n[{i+1}/{total}]")
        train_single(reward, seed, args.timesteps, args.n_envs)

    print(f"\n{'=' * 65}")
    print(f"全部完成！共 {total} runs")
    print(f"模型位置: models/racing_track_*/")
    print(f"TensorBoard: tensorboard --logdir logs/")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()