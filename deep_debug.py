import time
import numpy as np

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

def main():
    model_path = 'models/PPO_InfiniteGate_seed4/best_model.zip'
    env = InfiniteGateAviary(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2
    )

    try:
        model = PPO.load(model_path)
    except:
        print("無法載入模型，進行隨機動作偵錯")
        model = None

    obs, info = env.reset()
    for step in range(2000):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        if len(action.shape) == 1:
            action = np.array([action])
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # We want to trace gate passing logic
        state = env._getDroneStateVector(0)
        drone_pos = state[0:3]
        target_gate_pos = env.gate_poses[env.current_target_idx]
        
        if terminated or truncated:
            print(f"FAILED AT STEP {step}")
            print(f"reason: terminated={terminated}, truncated={truncated}")
            print(f"drone_pos: {drone_pos}")
            print(f"target_idx: {env.current_target_idx}")
            print(f"target_gate_pos: {target_gate_pos}")
            dist_to_gate = np.linalg.norm(drone_pos - target_gate_pos)
            print(f"dist to target gate: {dist_to_gate}")
            print(f"Info stats: gates passed = {info.get('gates_passed')}")
            
            # Check what caused truncated
            if truncated:
                if env.step_counter/env.PYB_FREQ > env.EPISODE_LEN_SEC:
                    print("TRUNCATION: Timeout!")
                if dist_to_gate > 10.0:
                    print("TRUNCATION: Too far from gate!")
            if terminated:
                if env._checkCollision():
                    print("TERMINATED: Collision Detected!")
            break

if __name__ == '__main__':
    for _ in range(3):
        main()
        print("-" * 50)
