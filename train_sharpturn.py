"""
場景 A: 急彎走廊 訓練腳本

使用方式：

  # Step 1: 先驗測試 (2 seeds，確認難度合適)
  python train_sharp_turn.py --pilot --seeds 0 1

  # Step 2: 如果結果符合預期，擴展到完整實驗
  python train_sharp_turn.py --run_all --seeds 0 1 2 3 4

  # 單一配置
  python train_sharp_turn.py --reward dynamic --seed 0
  python train_sharp_turn.py --reward static --seed 0

驗測試的成功標準：
  - A0 (無 bonus) 在急彎場景應該 < 40 門 (原版場景約 55 門)
  - Dynamic 應該明顯 > Static
  - 如果兩者都 < 15 門 → 場景太難，需降低 yaw_shift
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

from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def train_single(reward_type, seed, timesteps=2000000, n_envs=8, strict=False):

    use_dynamic = (reward_type == 'dynamic')

    exp_name = f"sharp_turn_{reward_type}"
    if strict:
        exp_name += "_strict"
    exp_name += f"_seed{seed}"
        
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
    print(f"  Scene A: Sharp Turn Corridor")
    print(f"  Gate dist: U(1.8, 5.0)m | Yaw: ±34° | Z: ±0.8m")
    print(f"  Reward: {reward_type} | Seed: {seed}")
    print("=" * 65)

    env_kwargs = dict(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2, # [解除盲視] 恢復神經網路的跨彎視覺
        use_dynamic_reward=use_dynamic,
        strict_miss=strict,
        gate_half_width=0.4 if strict else 0.5,
        heading_bonus_scale=5.0 if strict else 0.5,
        centering_bonus_scale=3.0 if strict else 1.5,
    )

    monitor_kwargs = dict(
        info_keywords=('gates_passed', 'collision_rate', 'avg_jerk', 'avg_pass_dist')
    )

    train_env = make_vec_env(
        SharpTurnCorridor,
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
        SharpTurnCorridor, n_envs=1,
        env_kwargs=env_kwargs, seed=seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=False, norm_reward=False, gamma=0.995,
    )

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
        description='場景 A: 急彎走廊訓練',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--reward', type=str, default=None,
                        choices=['dynamic', 'static'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--pilot', action='store_true',
                        help='先驗測試: 只跑 dynamic + static 各 2 seeds')
    parser.add_argument('--run_all', action='store_true',
                        help='完整實驗: dynamic + static × 所有 seeds')
    parser.add_argument('--strict', action='store_true',
                        help='開啟嚴格過門條件 (容錯1次, 門寬0.4)')
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
        print(f"成功標準: Dynamic > Static, 且兩者都 > 15 門\n")

    elif args.run_all:
        for reward in ['dynamic', 'static']:
            for seed in args.seeds:
                configs.append((reward, seed))

    elif args.reward:
        # 優先檢查單個 --seed
        if args.seed is not None:
            configs.append((args.reward, args.seed))
        else:
            for seed in args.seeds:
                configs.append((args.reward, seed))

    else:
        parser.print_help()
        print("\n建議流程:")
        print("  1. python train_sharp_turn.py --pilot --seeds 0 1")
        print("  2. 檢查結果")
        print("  3. python train_sharp_turn.py --run_all --seeds 0 1 2 3 4")
        return

    total = len(configs)
    for i, (reward, seed) in enumerate(configs):
        print(f"\n[{i+1}/{total}]")
        train_single(reward, seed, args.timesteps, args.n_envs, args.strict)

    print(f"\n全部完成！共 {total} runs")
    print(f"模型位置: models/sharp_turn/")
    print(f"TensorBoard: tensorboard --logdir logs/sharp_turn/")


if __name__ == '__main__':
    main()