"""
RL Algorithm Generalization Training Script — Exp 3

使用方式：

  # 跑單一配置
  python train_algo_generalization.py --algo PPO --reward dynamic --seed 0
  python train_algo_generalization.py --algo SAC --reward static --seed 0

  # 批次跑全部 (3 algos × 2 rewards × 5 seeds = 30 runs)
  python train_algo_generalization.py --run_all --seeds 0 1 2 3 4

  # 只補跑 SAC 和 TD3 (PPO 已跑過)
  python train_algo_generalization.py --algo SAC --run_both_rewards --seeds 0 1 2 3 4
  python train_algo_generalization.py --algo TD3 --run_both_rewards --seeds 0 1 2 3 4

實驗設計：
──────────────────────────────────────────────────────────────
  環境: InfiniteGateAviary (V3 Fair) — 完全不改動
  唯一自變量: RL 算法 × Reward 類型
  控制變因: 環境、obs、act、termination、network 大小 (256×256)

  PPO:  on-policy,  stochastic  (已有結果，可跳過)
  SAC:  off-policy, stochastic  (entropy-regularized)
  TD3:  off-policy, deterministic (twin critics)

  每個算法都跑 Dynamic 和 Static 兩組
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

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


# ================================================================
# 算法配置
# ================================================================
# 設計原則：所有算法共用的超參數盡量統一
# 只有算法特有的參數才各自設定

# 共用參數
SHARED = dict(
    gamma=0.995,
    learning_rate=3e-4,
    device="auto",
    verbose=1,
)

def create_model(algo_name, env, seed, log_dir):
    """根據算法名稱建立對應的 SB3 模型。
    
    PPO/SAC/TD3 三者的關鍵差異：
    
    PPO:
      - On-policy → 不需要 replay buffer
      - 每次 update 用整個 rollout batch
      - n_steps × n_envs = 一次收集的 transition 數
    
    SAC:
      - Off-policy → 有 replay buffer (1M transitions)
      - Entropy regularization → 自動調節探索力度
      - train_freq=1 → 每個環境 step 都做一次 gradient update
    
    TD3:
      - Off-policy → 有 replay buffer (1M transitions)
      - Deterministic policy → 需要外加 action noise 探索
      - policy_delay=2 → 每兩次 critic update 才做一次 actor update
    """
    
    if algo_name == 'PPO':
        # ---- PPO: On-policy, Stochastic ----
        # n_envs=8 時，每次 rollout 收集 8×1024=8192 transitions
        model = PPO(
            'MlpPolicy',
            env,
            seed=seed,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            **SHARED,
        )
        
    elif algo_name == 'SAC':
        # ---- SAC: Off-policy, Stochastic, Max-Entropy ----
        # SAC 自帶 entropy bonus，不需要手動設定探索噪聲
        # ent_coef='auto' → 自動調節 entropy 的溫度參數 α
        model = SAC(
            'MlpPolicy',
            env,
            seed=seed,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,             # soft target update 速率
            ent_coef='auto',       # 自動調節 entropy 溫度
            train_freq=1,          # 每 step 做一次 update
            learning_starts=10000, # 收集 10K transitions 後才開始訓練
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[256, 256]),
            **SHARED,
        )
        
    elif algo_name == 'TD3':
        # ---- TD3: Off-policy, Deterministic, Twin Critics ----
        # TD3 用 deterministic policy，需要手動加 action noise 做探索
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),   # 10% 的高斯噪聲
        )
        
        model = TD3(
            'MlpPolicy',
            env,
            seed=seed,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            policy_delay=2,        # 每兩次 critic update 做一次 actor update
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            action_noise=action_noise,
            gradient_steps=1,
            learning_starts=10000,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[256, 256]),
            **SHARED,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    
    return model


def train_single(algo_name, reward_type, seed, timesteps=1000000, n_envs=8):
    """訓練單一配置 (算法 × 獎勵類型 × seed)。"""
    
    use_dynamic = (reward_type == 'dynamic')
    
    exp_name = f"{algo_name}_{reward_type}_seed{seed}"
    models_dir = f"models/{exp_name}"
    log_dir = f"logs/{exp_name}"
    monitor_dir = f"{models_dir}/monitor"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    
    # 檢查是否已跑過
    if os.path.exists(f"{models_dir}/best_model.zip"):
        print(f"[SKIP] {exp_name} — already exists")
        return
    
    print("=" * 65)
    print(f"  Algorithm: {algo_name}")
    print(f"  Reward:    {reward_type} ({'proximity-conditioned' if use_dynamic else 'fixed weights'})")
    print(f"  Seed:      {seed}")
    print(f"  Timesteps: {timesteps/1e6:.1f}M")
    print("=" * 65)
    
    # 環境參數
    env_kwargs = dict(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2,
        use_dynamic_reward=use_dynamic,
    )
    
    monitor_kwargs = dict(
        info_keywords=('gates_passed', 'collision_rate', 'avg_jerk', 'avg_pass_dist')
    )
    
    # ---- 建立訓練環境 ----
    # PPO: 使用多進程加速 (on-policy 需要大量 rollout)
    # SAC/TD3: off-policy 不需要多進程，但多進程也能加速收集
    #          注意 SAC/TD3 的 SubprocVecEnv 需要 n_envs=1 或配合 train_freq 調整
    if algo_name == 'PPO':
        vec_env_cls = SubprocVecEnv if n_envs > 1 else None
        actual_n_envs = n_envs
    else:
        # SAC/TD3: off-policy 算法用單一環境較穩定
        # 多環境會讓 replay buffer 的 transition 品質更分散
        vec_env_cls = None
        actual_n_envs = 1
    
    train_env = make_vec_env(
        InfiniteGateAviary,
        n_envs=actual_n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        seed=seed,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
    )
    
    # VecNormalize (只 normalize reward)
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True,
        clip_reward=10.0, gamma=0.995,
    )
    
    # 建立評估環境
    eval_env = make_vec_env(
        InfiniteGateAviary, n_envs=1,
        env_kwargs=env_kwargs, seed=seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=False, norm_reward=False, gamma=0.995,
    )
    
    # ---- 建立模型 ----
    model = create_model(algo_name, train_env, seed, log_dir)
    
    # ---- Eval Callback ----
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    
    # ---- 訓練 ----
    print(f"開始訓練 {exp_name}...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    elapsed = time.time() - start_time
    print(f"完成！耗時: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    
    # 儲存
    model.save(os.path.join(models_dir, "final_model"))
    train_env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    
    train_env.close()
    eval_env.close()
    del model


def main():
    parser = argparse.ArgumentParser(
        description='RL Algorithm Generalization (Exp 3): PPO/SAC/TD3 × Dynamic/Static',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--algo', type=str, default=None,
                        choices=['PPO', 'SAC', 'TD3'],
                        help='RL 算法')
    parser.add_argument('--reward', type=str, default=None,
                        choices=['dynamic', 'static'],
                        help='Reward 類型')
    parser.add_argument('--run_both_rewards', action='store_true',
                        help='對指定算法同時跑 dynamic 和 static')
    parser.add_argument('--run_all', action='store_true',
                        help='跑全部 3 algos × 2 rewards')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Seed 列表 (default: 0 1 2 3 4)')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='每個配置的訓練步數 (default: 2000000)')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='PPO 平行環境數 (SAC/TD3 強制為 1)')
    args = parser.parse_args()
    
    # 決定要跑的配置
    configs = []
    
    if args.run_all:
        for algo in ['PPO', 'SAC', 'TD3']:
            for reward in ['dynamic', 'static']:
                configs.append((algo, reward))
    elif args.run_both_rewards and args.algo:
        configs.append((args.algo, 'dynamic'))
        configs.append((args.algo, 'static'))
    elif args.algo and args.reward:
        configs.append((args.algo, args.reward))
    else:
        parser.print_help()
        print("\n範例:")
        print("  python train_algo_generalization.py --algo SAC --reward dynamic --seeds 0 1 2")
        print("  python train_algo_generalization.py --algo TD3 --run_both_rewards --seeds 0 1 2 3 4")
        print("  python train_algo_generalization.py --run_all --seeds 0 1 2 3 4")
        return
    
    total_runs = len(configs) * len(args.seeds)
    print(f"\n{'='*65}")
    print(f"  Exp 3: RL Algorithm Generalization")
    print(f"  配置: {[(a, r) for a, r in configs]}")
    print(f"  Seeds: {args.seeds}")
    print(f"  總 runs: {total_runs}")
    print(f"{'='*65}\n")
    
    completed = 0
    for algo, reward in configs:
        for seed in args.seeds:
            completed += 1
            print(f"\n[{completed}/{total_runs}]")
            train_single(
                algo, reward, seed,
                timesteps=args.timesteps,
                n_envs=args.n_envs,
            )
    
    print(f"\n{'='*65}")
    print(f"  全部完成！共 {total_runs} 個 runs")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()