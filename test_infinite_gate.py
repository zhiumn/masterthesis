import time
import numpy as np
from gym_pybullet_drones.envs.InfiniteGateAviary import InfiniteGateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics
import pybullet as p

def main():
    print("Testing InfiniteGateAviary (2 lookahead gates)")
    env = InfiniteGateAviary(
        gui=True, # Set to False for automated test, switch to True for visual
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        num_lookahead_gates=2
    )

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape) # Should be (1, 12 + 2*6 + actions buffer...)
    print("Expected obs shape base: 12 + 2*6 = 24")

    # Run for a few steps to check mechanics
    start_time = time.time()
    for i in range(100):
        # Action is RPMs [-1, 1], let's command slight forward thrust
        # For CF2X, RPMs are mapped directly. We'll just give random but biased forward commands.
        # Format of action for 1 drone: shape (1, 4)
        action = np.array([[ 0.5, 0.5, 0.5, 0.5 ]]) # Hover+
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended at step {i}. Terminated: {terminated}, Truncated: {truncated}")
            break
            
    print("Final Info:", info)
    print("Test completed successfully.")
    env.close()

if __name__ == "__main__":
    main()
