import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# 1. 以無 GUI 模式啟動
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 2. **不要**載入地板 (plane.urdf)，只載入單一門框
gate = p.loadURDF("gym_pybullet_drones/assets/hollow_gate.urdf", [0, 0, 0])

# 3. 設定攝影機視角與取得影像 (RGbD 矩陣)
width, height = 1024, 1024
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 0.5],
    distance=2.5, yaw=45, pitch=-20, roll=0, upAxisIndex=2)
proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)

import numpy as np
# 取得影像資料
_, _, rgbImg, _, segImg = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
rgbImg = np.reshape(rgbImg, (height, width, 4)).astype(np.uint8)
segImg = np.reshape(segImg, (height, width))

# 利用 Segmentation mask 將背景的 Alpha 設為 0
rgbImg[segImg == -1, 3] = 0

# 4. 儲存為帶有 Alpha 通道 (透明背景) 的 PNG
plt.imsave("perfect_gate.png", rgbImg)
p.disconnect()