import os
import numpy as np
import pybullet as p
from PIL import Image

from gym_pybullet_drones.envs.InfiniteGateAviaryAblation import InfiniteGateAviaryAblation
from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
from gym_pybullet_drones.envs.RacingTrackAblation import RacingTrackAblation
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def capture_screenshot(env, filename='screenshot.png', view_type='corridor', width=1920, height=1080):
    """
    從上帝視角 (Isometric/Birds-eye) 擷取 PyBullet 原始環境影像 (保留預設棋盤格地板與紅色門框)。
    """
    # 取得無人機當前位置與朝向
    pos = env.pos[0]
    quat = env.quat[0]
    euler = p.getEulerFromQuaternion(quat)
    drone_yaw = euler[2]
    
    forward_x = np.cos(drone_yaw)
    forward_y = np.sin(drone_yaw)

    if view_type == 'corridor' or view_type == 'sharp':
        # 稍微拉低相機，取得約 45 度的完美等角透視 (Isometric) 平衡點
        # 既能清楚看到門框沿著飛行方向延伸，又不會因為太低而露出過多天空邊界
        camera_eye = [
            pos[0] - forward_x * 4,
            pos[1] - forward_y * 4,
            pos[2] + 7.5
        ]
        camera_target = [
            pos[0] + forward_x * 7,
            pos[1] + forward_y * 7,
            pos[2]
        ]
        
        # 動態載入 PyBullet 內建的「寺廟背景」 (samurai.urdf)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=env.CLIENT)
        temple_pos = [
            pos[0] + forward_x * 30.0,
            pos[1] + forward_y * 30.0,
            -0.5  # 輕微下沉，確保階梯與地平面貼齊
        ]
        # 讓寺廟正面轉向朝著無人機的方向
        temple_yaw = drone_yaw - (np.pi / 2)
        temple_ori = p.getQuaternionFromEuler([0, 0, temple_yaw])
        p.loadURDF("samurai.urdf", temple_pos, temple_ori, globalScaling=2.5, physicsClientId=env.CLIENT)

    else:  # racing
        # Racing Track：維持 90 度橫向選轉，並將相機稍微降低，
        # 給予一點立體深度感，不至於像純上帝視角那樣平。
        camera_target = [0.0, 0.0, 0.0]
        camera_eye = [-16.0, 0.0, 8.0]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1],
        physicsClientId=env.CLIENT
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=width / height,
        nearVal=0.1,
        farVal=100.0
    )

    # 原始渲染 (無漸層背景後處理)
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

    np.random.seed(42)

    # ================================================================
    # 場景 A: Standard Corridor (基準走廊)
    # ================================================================
    print("=" * 50)
    print("擷取場景 A: Standard Corridor")
    print("=" * 50)

    env_a = InfiniteGateAviaryAblation(
        gui=False,       # 不開視窗，用 DIRECT mode 渲染
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    env_a.reset(seed=42)
    
    capture_screenshot(
        env_a,
        filename=os.path.join(out_dir, 'scenario_standard_corridor.png'),
        view_type='corridor'
    )
    env_a.close()


    # ================================================================
    # 場景 B: Sharp Turn Corridor (急彎走廊)
    # ================================================================
    print("\n" + "=" * 50)
    print("擷取場景 B: Sharp Turn Corridor")
    print("=" * 50)

    env_b = SharpTurnCorridor(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    env_b.reset(seed=42)

    capture_screenshot(
        env_b,
        filename=os.path.join(out_dir, 'scenario_sharp_turn.png'),
        view_type='sharp'
    )
    env_b.close()


    # ================================================================
    # 場景 C: Racing Track (環狀賽道)
    # ================================================================
    print("\n" + "=" * 50)
    print("擷取場景 C: Racing Track")
    print("=" * 50)

    env_c = RacingTrackAblation(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    env_c.reset(seed=42)

    capture_screenshot(
        env_c,
        filename=os.path.join(out_dir, 'scenario_racing_track.png'),
        view_type='racing'
    )
    env_c.close()


if __name__ == '__main__':
    main()