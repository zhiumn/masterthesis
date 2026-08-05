import os
import numpy as np
import pybullet as p
from PIL import Image

from gym_pybullet_drones.envs.InfiniteGateAviaryAblation import InfiniteGateAviaryAblation
from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
from gym_pybullet_drones.envs.RacingTrackAblation import RacingTrackAblation
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

def capture_fpv_screenshot(env, filename='fpv_screenshot.png', width=1920, height=1080):
    """
    從無人機第一人稱視角 (FPV) 擷取畫面。
    """
    pos = env.pos[0]
    quat = env.quat[0]
    
    # 計算無人機正前方與正上方的向量
    target_pos, _ = p.multiplyTransforms(pos, quat, [5.0, 0.0, 0.0], [0, 0, 0, 1], physicsClientId=env.CLIENT)
    up_pos, _ = p.multiplyTransforms(pos, quat, [0.0, 0.0, 1.0], [0, 0, 0, 1], physicsClientId=env.CLIENT)
    up_vector = [up_pos[0] - pos[0], up_pos[1] - pos[1], up_pos[2] - pos[2]]
    
    # 相機位置稍微往前推一點，避免拍到機身內部穿模
    camera_eye, _ = p.multiplyTransforms(pos, quat, [0.15, 0.0, 0.05], [0, 0, 0, 1], physicsClientId=env.CLIENT)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vector,
        physicsClientId=env.CLIENT
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=90.0,  # FPV 通常具有較廣的視角
        aspect=width / height,
        nearVal=0.1,
        farVal=100.0,
        physicsClientId=env.CLIENT
    )

    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[1, 1, 2],
        lightAmbientCoeff=0.6,
        lightDiffuseCoeff=0.8,
        physicsClientId=env.CLIENT
    )

    img = np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))
    img_rgb = img[:, :, :3]
    
    result = Image.fromarray(img_rgb, 'RGB')
    result.save(filename, quality=95)
    print(f"  已儲存: {filename}")


def main():
    out_dir = 'sample_efficiency_results'
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 50)
    print("擷取場景 A: Standard Corridor (FPV)")
    env_a = InfiniteGateAviaryAblation(gui=False, record=False, obs=ObservationType.KIN, act=ActionType.RPM)
    env_a.reset(seed=42)
    # 稍微模擬幾步讓姿態穩定，或讓無人機靠近門框
    capture_fpv_screenshot(env_a, filename=os.path.join(out_dir, 'scenario_standard_corridor_fpv.png'))
    env_a.close()

    print("\n" + "=" * 50)
    print("擷取場景 B: Sharp Turn Corridor (FPV)")
    env_b = SharpTurnCorridor(gui=False, record=False, obs=ObservationType.KIN, act=ActionType.RPM)
    env_b.reset(seed=42)
    capture_fpv_screenshot(env_b, filename=os.path.join(out_dir, 'scenario_sharp_turn_fpv.png'))
    env_b.close()

    print("\n" + "=" * 50)
    print("擷取場景 C: Racing Track (FPV)")
    env_c = RacingTrackAblation(gui=False, record=False, obs=ObservationType.KIN, act=ActionType.RPM)
    env_c.reset(seed=42)
    capture_fpv_screenshot(env_c, filename=os.path.join(out_dir, 'scenario_racing_track_fpv.png'))
    env_c.close()


if __name__ == '__main__':
    main()
