import pybullet as p
import pybullet_data
from PIL import Image
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("samurai.urdf")
view = p.computeViewMatrixFromYawPitchRoll([0,0,0], 40, 0, -30, 0, 2)
proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
_,_,rgb,_,_ = p.getCameraImage(400, 400, viewMatrix=view, projectionMatrix=proj)
img = np.reshape(np.array(rgb, dtype=np.uint8), (400,400,4))[:,:,:3]
Image.fromarray(img).save("tmp_samurai.png")
