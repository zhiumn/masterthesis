import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class InfiniteGateAviary(BaseRLAviary):
    """Infinite Gate Flying RL Environment for dynamic-weighted reward testing."""
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 num_lookahead_gates: int = 1,
                 use_dynamic_reward: bool = True
                 ):
        
        self.use_dynamic_reward = use_dynamic_reward
        
        self.num_lookahead_gates = num_lookahead_gates
        self.MAX_VISIBLE_GATES = 5 # Pool size for smooth transition and recycling
        self.gate_ids = []
        self.gate_poses = []
        self.gate_quats = []
        self.current_target_idx = 0
        
        # Performance Tracking
        self.passed_gates = 0
        self.total_gates_encountered = 0
        self.collisions = 0
        self.episode_steps = 0
        self.total_jerk = 0.0
        self.pass_distances = []
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)
        self.continuous_passing_broken = False
        self.gate_missed = False
        
        # Reward Function Parameters
        self.w_lat = 2.0
        self.w_z = 2.0
        self.w_heading = 1.0
        self.T_temp = 2.0
        self.angle_threshold = 0.5 # radians
        self.V_max = 5.0 # Max speed limit
        self.lambda_v = 10.0 # Speed penalty weight
        self.lambda_jerk = 0.005 # Jerk penalty weight
        
        self.EPISODE_LEN_SEC = 60.0 # Max seconds

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _addObstacles(self):
        """Add virtual hollow gates to the pybullet environment."""
        self.gate_ids = []
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'hollow_gate.urdf')
        
        for i in range(self.MAX_VISIBLE_GATES):
            gate_id = p.loadURDF(urdf_path, [0, 0, -100], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.CLIENT)
            self.gate_ids.append(gate_id)
            
        self._generate_initial_gates()

    def _generate_initial_gates(self):
        self.gate_poses = []
        self.gate_quats = []
        
        # Ensure initial offset matches initial_xyzs if provided, else [0,0,1]
        cur_pos = self.INIT_XYZS[0].copy() if len(self.INIT_XYZS) > 0 else np.array([0., 0., 1.0])
        cur_yaw = 0.0
        
        for i in range(self.MAX_VISIBLE_GATES):
            dist = np.random.uniform(2.5, 4.0)
            yaw_shift = np.random.uniform(-0.35, 0.35)
            z_shift = np.random.uniform(-0.5, 0.5)
            
            cur_yaw += yaw_shift
            dx = dist * np.cos(cur_yaw)
            dy = dist * np.sin(cur_yaw)
            cur_pos = cur_pos + np.array([dx, dy, z_shift])
            cur_pos[2] = max(0.6, cur_pos[2]) # Prevent clipping into ground
            
            self.gate_poses.append(cur_pos.copy())
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, cur_yaw]))
            
            p.resetBasePositionAndOrientation(self.gate_ids[i], cur_pos, self.gate_quats[-1], physicsClientId=self.CLIENT)
            
        self.current_target_idx = 0

    def _generate_next_gate(self, replace_idx):
        last_idx = (self.current_target_idx - 1) % self.MAX_VISIBLE_GATES
        last_pos = self.gate_poses[last_idx]
        last_quat = self.gate_quats[last_idx]
        last_yaw = p.getEulerFromQuaternion(last_quat)[2]
        
        dist = np.random.uniform(2.5, 4.0)
        yaw_shift = np.random.uniform(-0.35, 0.35)
        z_shift = np.random.uniform(-0.5, 0.5)
        
        new_yaw = last_yaw + yaw_shift
        dx = dist * np.cos(new_yaw)
        dy = dist * np.sin(new_yaw)
        new_pos = last_pos + np.array([dx, dy, z_shift])
        new_pos[2] = max(0.6, new_pos[2])
        new_quat = p.getQuaternionFromEuler([0, 0, new_yaw])
        
        self.gate_poses[replace_idx] = new_pos
        self.gate_quats[replace_idx] = new_quat
        p.resetBasePositionAndOrientation(self.gate_ids[replace_idx], new_pos, new_quat, physicsClientId=self.CLIENT)

    def _observationSpace(self):
        obs_dim = 12 + self.num_lookahead_gates * 6
        lo = -np.inf
        hi = np.inf
        
        obs_lower_bound = np.array([[lo for _ in range(obs_dim)] for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi for _ in range(obs_dim)] for _ in range(self.NUM_DRONES)])
        
        for i in range(self.ACTION_BUFFER_SIZE):
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[-1]*4 for _ in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[+1]*4 for _ in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE==ActionType.PID:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[-1]*3 for _ in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[+1]*3 for _ in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[-1] for _ in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[+1] for _ in range(self.NUM_DRONES)])])
                
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        obs_dim = 12 + self.num_lookahead_gates * 6
        obs_full = np.zeros((self.NUM_DRONES, obs_dim))
        
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            
            drone_pos = state[0:3]
            drone_rpy = state[7:10]
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            
            # 旋轉不變性修正：將絕對速度與角速度投影回無人機本身的 Body Frame
            body_vel = R_drone.T @ state[10:13]
            body_ang_vel = R_drone.T @ state[13:16]
            
            # 平移與水平旋轉不變性修正：捨去絕對 X, Y, 以及絕對 Yaw
            # 但保留高度 Z (避免撞地) 以及 Roll, Pitch (感知重力方向)
            obs_pos = np.array([0.0, 0.0, state[2]])
            obs_rpy = np.array([state[7], state[8], 0.0])
            
            obs_kin = np.hstack([obs_pos, obs_rpy, body_vel, body_ang_vel])
            
            next_gates_info = []
            for j in range(self.num_lookahead_gates):
                idx = (self.current_target_idx + j) % self.MAX_VISIBLE_GATES
                g_pos = self.gate_poses[idx]
                g_quat = self.gate_quats[idx]
                
                rel_pos = R_drone.T @ (g_pos - drone_pos)
                
                R_target = Rotation.from_quat(g_quat).as_matrix()
                R_rel = R_drone.T @ R_target
                rel_rpy = Rotation.from_matrix(R_rel).as_euler('xyz')
                
                next_gates_info.extend(rel_pos.tolist())
                next_gates_info.extend(rel_rpy.tolist())
                
            obs_full[i, :] = np.hstack([obs_kin, np.array(next_gates_info)])

        ret = np.array([obs_full[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        return ret
        
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]

        gate_pos = self.gate_poses[self.current_target_idx]
        gate_quat = self.gate_quats[self.current_target_idx]

        R_gate = Rotation.from_quat(gate_quat).as_matrix()

        # 將無人機位置與速度轉到 gate frame
        p_G = R_gate.T @ (drone_pos - gate_pos)
        v_G = R_gate.T @ drone_vel

        # gate frame 下的幾何誤差
        r_approach = v_G[0]
        dist_lat = abs(p_G[1])
        dist_height = abs(p_G[2])

        # heading error
        gate_normal = R_gate @ np.array([1, 0, 0])
        speed = np.linalg.norm(drone_vel)

        if speed > 0.5:
            drone_dir = drone_vel / speed
        else:
            drone_rpy = state[7:10]
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            drone_dir = R_drone @ np.array([1, 0, 0])

        cos_theta = np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0)
        theta_err = np.arccos(cos_theta)

        # 超速懲罰
        speed_penalty = self.lambda_v * (max(0.0, speed - self.V_max) ** 2)

        R_acquire = r_approach - self.w_heading * theta_err
        R_align = r_approach - (self.w_lat * (dist_lat ** 2) + self.w_z * (dist_height ** 2))

        # -----------------------------
        # Static baseline vs Dynamic reward
        # -----------------------------
        if self.use_dynamic_reward:
            # 將最一開始基於「角度誤差的 Sigmoid 突變切換」，
            # 改良為基於「門框距離的平滑空間漏斗切換」。
            d_plane = abs(p_G[0])
            proximity = np.exp(-0.5 * d_plane) # 靠近門框為 1.0，遠處趨近 0.0

            # 逼近門框時，極度重視空間位置對準 (w_align 高達 0.8)
            # 遠離門框時，極度重視機首航向提前對準 (w_acquire 趨近 0.8)
            # 兩者皆設有保底 (0.2)，確保無人機永遠都會兼顧兩個目標，徹底阻絕 Reward Hacking。
            w_align = 0.2 + 0.6 * proximity
            w_acquire = 1.0 - w_align
        else:
            # Static baseline：完全公平分配，不做任何動態切換
            alpha = 0.5
            w_acquire = alpha
            w_align = 1.0 - alpha

        R_total = (w_acquire * R_acquire) + (w_align * R_align) - speed_penalty

        # -----------------------------
        # 穿越 gate 檢查
        # -----------------------------
        if p_G[0] > 0.0:
            self.total_gates_encountered += 1

            # 記錄穿越 gate 平面時與 gate 幾何中心的偏移
            dist_to_center = np.sqrt(dist_lat ** 2 + dist_height ** 2)
            self.pass_distances.append(dist_to_center)

            # 只有真的穿過 gate 中間鏤空區域才算成功
            if dist_lat < 0.5 and dist_height < 0.5:
                if not getattr(self, 'continuous_passing_broken', False):
                    self.passed_gates += 1
                R_total += 50.0
            else:
                self.continuous_passing_broken = True
                self.gate_missed = True
                R_total -= 10.0

            self._generate_next_gate(self.current_target_idx)
            self.current_target_idx = (self.current_target_idx + 1) % self.MAX_VISIBLE_GATES

        return R_total

    def _computeInfo(self):
        """Computes the current info dict(s).
        
        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        # 回傳一個包含空字典的 info，這是最簡單、最不會出錯的寫法
        return {}

    def _checkCollision(self):
        drone_id = self.DRONE_IDS[0]
        # Gate borders collision
        for gate_id in self.gate_ids:
            pts = p.getContactPoints(bodyA=drone_id, bodyB=gate_id, physicsClientId=self.CLIENT)
            if len(pts) > 0:
                return True
        # Ground collision
        if self._getDroneStateVector(0)[2] < 0.1:
            return True
        return False

    def _computeTerminated(self):
        if getattr(self, 'gate_missed', False):
            return True
        if self._checkCollision():
            self.collisions += 1
            return True
        return False

    def _computeTruncated(self):
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        state = self._getDroneStateVector(0)
        gate_pos = self.gate_poses[self.current_target_idx]
        if np.linalg.norm(state[0:3] - gate_pos) > 10.0:
             return True
        return False

    def step(self, action):
        dt = self.CTRL_TIMESTEP
        cur_vel = self._getDroneStateVector(0)[10:13]
        cur_acc = (cur_vel - self.prev_vel) / dt
        
        jerk = 0.0
        if self.episode_steps > 0:
            jerk = np.linalg.norm((cur_acc - self.prev_acc) / dt)
            self.total_jerk += jerk
            
        self.episode_steps += 1
        self.prev_vel = cur_vel
        self.prev_acc = cur_acc
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 將 jerk 納入 reward 懲罰，強迫模型學習平滑飛行
        reward -= self.lambda_jerk * jerk
        
        info["collision_rate"] = self.collisions / max(1, self.episode_steps)
        info["success_rate"] = self.passed_gates / max(1, self.total_gates_encountered)
        info["avg_jerk"] = self.total_jerk / max(1, self.episode_steps)
        info["gates_passed"] = self.passed_gates
        info["avg_pass_dist"] = np.mean(self.pass_distances) if len(self.pass_distances) > 0 else 0.0
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.passed_gates = 0
        self.total_gates_encountered = 0
        self.collisions = 0
        self.episode_steps = 0
        self.total_jerk = 0.0
        self.pass_distances = []
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)
        self.continuous_passing_broken = False
        self.gate_missed = False
        
        return super().reset(seed=seed, options=options)
