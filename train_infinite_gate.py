import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

def train_single(baseline, seed, timesteps, lookahead, n_envs):
    """訓練單一配置。"""
    if baseline:
        exp_prefix = f"Baseline_L{lookahead}"
    else:
        exp_prefix = f"PPO_InfiniteGate_L{lookahead}"
        
    models_dir = f"models/{exp_prefix}_seed{seed}"
    log_dir = f"logs/{exp_prefix}_seed{seed}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(os.path.join(models_dir, "best_model.zip")):
        print(f"[SKIP] {exp_prefix}_seed{seed} — already exists")
        return

    print("=" * 60)
    print(f"環境: InfiniteGateAviary V2 (Lookahead: {lookahead})")
    print(f"Reward: {'Static (Baseline)' if baseline else 'Additive Dynamic (Proposed)'}")
    print(f"平行環境: {n_envs} | Seed: {seed}")
    print(f"總訓練步數: {timesteps}")
    print("=" * 60)

    env_kwargs = dict(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=lookahead,
        use_dynamic_reward=not baseline
    )
    
    monitor_dir = f"{models_dir}/monitor"
    os.makedirs(monitor_dir, exist_ok=True)
    monitor_kwargs = dict(info_keywords=('gates_passed', 'collision_rate', 'avg_jerk', 'avg_pass_dist'))

    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    train_env = make_vec_env(
        InfiniteGateAviary, 
        n_envs=n_envs, 
        env_kwargs=env_kwargs, 
        vec_env_cls=vec_env_cls, 
        seed=seed,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs
    )
    
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.995
    )

    eval_env = make_vec_env(
        InfiniteGateAviary, 
        n_envs=1, 
        env_kwargs=env_kwargs, 
        seed=seed+1000
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        gamma=0.995
    )

    model = PPO('MlpPolicy',
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
                ))

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=log_dir, 
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True, 
        render=False
    )

    print(f"開始訓練 {exp_prefix}_seed{seed}...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    print(f"訓練完成！耗時: {time.time() - start_time:.2f} 秒")

    final_model_path = os.path.join(models_dir, "ppo_infinite_gate_final")
    model.save(final_model_path)
    
    vec_norm_path = os.path.join(models_dir, "vec_normalize.pkl")
    train_env.save(vec_norm_path)

    train_env.close()
    eval_env.close()
    del model

def main():
    parser = argparse.ArgumentParser(description='訓練 InfiniteGateAviary V2')
    parser.add_argument('--timesteps', type=int, default=2000000, help='訓練的總 steps 數量')
    parser.add_argument('--lookahead', type=int, default=2)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--baseline', action='store_true', help='使用固定權重 Baseline')
    parser.add_argument('--run_all', action='store_true', help='同時跑 Proposed 與 Baseline × 所有 seeds')
    args = parser.parse_args()

    configs = []
    if args.run_all:
        for b in [True, False]:
            for s in args.seeds:
                configs.append((b, s))
    elif args.seed is not None:
        configs.append((args.baseline, args.seed))
    else:
        for s in args.seeds:
            configs.append((args.baseline, s))

    total = len(configs)
    for i, (b, s) in enumerate(configs):
        print(f"\n[{i+1}/{total}]")
        train_single(b, s, args.timesteps, args.lookahead, args.n_envs)

if __name__ == '__main__':
    main()