import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class InfiniteGateAviary(BaseRLAviary):
    """Infinite Gate Flying RL Environment — V3 Fair Comparison.
    
    公平對照設計原則：
    ┌─────────────────────────────────────────────────────────┐
    │  唯一自變量 (Independent Variable):                      │
    │    bonus 權重是否受 proximity 動態調制                     │
    │                                                         │
    │  控制變因 (Controlled Variables):                         │
    │    - R_base 完全相同 (50/50 fixed)                       │
    │    - 三個 bonus term 結構完全相同                         │
    │    - bonus 的總 budget (期望值) 近似相等                   │
    │    - gate pass/miss 獎懲完全相同                          │
    │    - termination 條件完全相同                             │
    │    - observation space 完全相同                           │
    │    - PPO 超參數完全相同                                   │
    └─────────────────────────────────────────────────────────┘
    """
    
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
        self.MAX_VISIBLE_GATES = 5
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
        self.consecutive_misses = 0
        
        # ============================================================
        # Reward 參數 (Static 和 Dynamic 共用)
        # ============================================================
        self.w_lat = 2.0
        self.w_z = 2.0
        self.w_heading = 1.0
        self.V_max = 5.0
        self.lambda_v = 10.0
        self.lambda_jerk = 0.005
        
        # 三個 Bonus 共用的 scale 參數 (兩組方法完全相同)
        self.centering_bonus_scale = 1.5
        self.heading_bonus_scale = 0.5
        self.smooth_bonus_scale = 0.3
        
        # Dynamic Reward 專用：proximity 衰減銳利度
        self.proximity_sharpness = 0.8
        
        # Static Reward 專用：固定常數權重
        # ============================================================
        # 公平性關鍵設計：
        #
        # Dynamic 的 centering bonus = scale × proximity × quality
        # Dynamic 的 heading bonus  = scale × (1 - proximity) × quality
        #
        # proximity = exp(-0.8 × d_plane)，在典型飛行軌跡中：
        #   d_plane ∈ [0, 4]，proximity 的期望值約為 0.45
        #
        # 為了讓兩組方法的 bonus 總 budget 期望值近似相等，
        # Static 的固定權重設為 proximity 的期望值 ≈ 0.45
        # ============================================================
        self.static_proximity_weight = 0.5  # ≈ E[exp(-0.8 × d_plane)]
        
        # Termination
        self.max_consecutive_misses = 2
        
        self.EPISODE_LEN_SEC = 60.0

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

    # ================================================================
    # 環境建構 (兩組方法完全相同，不做任何修改)
    # ================================================================

    def _addObstacles(self):
        self.gate_ids = []
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'hollow_gate.urdf')
        for i in range(self.MAX_VISIBLE_GATES):
            gate_id = p.loadURDF(urdf_path, [0, 0, -100], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.CLIENT)
            self.gate_ids.append(gate_id)
        self._generate_initial_gates()

    def _generate_initial_gates(self):
        self.gate_poses = []
        self.gate_quats = []
        cur_pos = self.INIT_XYZS[0].copy() if len(self.INIT_XYZS) > 0 else np.array([0., 0., 1.0])
        cur_yaw = 0.0
        for i in range(self.MAX_VISIBLE_GATES):
            dist = self.np_random.uniform(2.5, 4.0)
            yaw_shift = self.np_random.uniform(-0.35, 0.35)
            z_shift = self.np_random.uniform(-0.5, 0.5)
            cur_yaw += yaw_shift
            dx = dist * np.cos(cur_yaw)
            dy = dist * np.sin(cur_yaw)
            cur_pos = cur_pos + np.array([dx, dy, z_shift])
            cur_pos[2] = max(0.6, cur_pos[2])
            self.gate_poses.append(cur_pos.copy())
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, cur_yaw]))
            p.resetBasePositionAndOrientation(self.gate_ids[i], cur_pos, self.gate_quats[-1], physicsClientId=self.CLIENT)
        self.current_target_idx = 0

    def _generate_next_gate(self, replace_idx):
        last_idx = (self.current_target_idx - 1) % self.MAX_VISIBLE_GATES
        last_pos = self.gate_poses[last_idx]
        last_quat = self.gate_quats[last_idx]
        last_yaw = p.getEulerFromQuaternion(last_quat)[2]
        dist = self.np_random.uniform(2.5, 4.0)
        yaw_shift = self.np_random.uniform(-0.35, 0.35)
        z_shift = self.np_random.uniform(-0.5, 0.5)
        new_yaw = last_yaw + yaw_shift
        dx = dist * np.cos(new_yaw)
        dy = dist * np.sin(new_yaw)
        new_pos = last_pos + np.array([dx, dy, z_shift])
        new_pos[2] = max(0.6, new_pos[2])
        new_quat = p.getQuaternionFromEuler([0, 0, new_yaw])
        self.gate_poses[replace_idx] = new_pos
        self.gate_quats[replace_idx] = new_quat
        p.resetBasePositionAndOrientation(self.gate_ids[replace_idx], new_pos, new_quat, physicsClientId=self.CLIENT)

    # ================================================================
    # Observation (兩組方法完全相同)
    # ================================================================

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
            body_vel = R_drone.T @ state[10:13]
            body_ang_vel = R_drone.T @ state[13:16]
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

    # ================================================================
    # 核心：公平比較的 Reward Function
    # ================================================================
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]

        gate_pos = self.gate_poses[self.current_target_idx]
        gate_quat = self.gate_quats[self.current_target_idx]
        R_gate = Rotation.from_quat(gate_quat).as_matrix()

        # ---- Gate frame 投影 (兩組完全相同) ----
        p_G = R_gate.T @ (drone_pos - gate_pos)
        v_G = R_gate.T @ drone_vel

        r_approach = v_G[0]
        dist_lat = abs(p_G[1])
        dist_height = abs(p_G[2])
        d_plane = abs(p_G[0])

        # ---- Heading error (兩組完全相同) ----
        gate_normal = R_gate @ np.array([1, 0, 0])
        speed = np.linalg.norm(drone_vel)
        if speed > 0.5:
            drone_dir = drone_vel / speed
        else:
            drone_rpy = state[7:10]
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            drone_dir = R_drone @ np.array([1, 0, 0])

        # [修正] heading_quality 改為朝向「下一道門」(與 Ablation 環境對齊)
        next_idx = (self.current_target_idx + 1) % self.MAX_VISIBLE_GATES
        next_gate_pos = self.gate_poses[next_idx]
        next_vec = next_gate_pos - drone_pos
        next_dir = next_vec / (np.linalg.norm(next_vec) + 1e-6)

        cos_theta = np.clip(np.dot(drone_dir, next_dir), -1.0, 1.0)
        theta_err = np.arccos(np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0))

        # ---- Speed penalty (兩組完全相同) ----
        speed_penalty = self.lambda_v * (max(0.0, speed - self.V_max) ** 2)

        # ---- Base reward (兩組完全相同) ----
        R_acquire = r_approach - self.w_heading * theta_err
        R_align = r_approach - (self.w_lat * (dist_lat ** 2) + self.w_z * (dist_height ** 2))
        R_base = 0.5 * R_acquire + 0.5 * R_align

        # ---- 三個 bonus 的 quality 計算 (兩組完全相同) ----
        centering_quality = np.exp(-(dist_lat**2 + dist_height**2) / 0.25)
        heading_quality = max(0.0, cos_theta)
        approach_speed = np.clip(v_G[0], 0.0, 3.0)
        smooth_quality = approach_speed * (1.0 - 0.5 * (dist_lat**2 + dist_height**2))

        # ============================================================
        # 唯一差異：bonus 的權重是動態的還是固定的
        # ============================================================
        if self.use_dynamic_reward:
            # ---- Proposed: Proximity-Conditioned Weights ----
            proximity = np.exp(-self.proximity_sharpness * d_plane)
            
            w_centering = proximity              # 近門高，遠門低
            w_heading = 1.0 - proximity           # 遠門高，近門低
            w_smooth = 1.0                        # 全程固定 (兩組相同)
        else:
            # ---- Baseline: Fixed Constant Weights ----
            # 使用 proximity 的期望值作為固定權重
            # 確保 bonus 的總 budget (期望值) 近似相等
            w_centering = self.static_proximity_weight       # ≈ 0.45
            w_heading = 1.0 - self.static_proximity_weight   # ≈ 0.55
            w_smooth = 1.0                                   # 全程固定 (兩組相同)

        # ---- 組合 (結構完全相同，只有 w 值不同) ----
        R_centering = self.centering_bonus_scale * w_centering * centering_quality
        R_heading = self.heading_bonus_scale * w_heading * heading_quality
        R_smooth = self.smooth_bonus_scale * w_smooth * smooth_quality

        R_total = R_base + R_centering + R_heading + R_smooth - speed_penalty

        # ---- 穿越 gate 檢查 (兩組完全相同) ----
        if p_G[0] > 0.0:
            self.total_gates_encountered += 1
            dist_to_center = np.sqrt(dist_lat ** 2 + dist_height ** 2)
            self.pass_distances.append(dist_to_center)

            if dist_lat < 0.5 and dist_height < 0.5:
                self.passed_gates += 1
                self.consecutive_misses = 0
                precision_ratio = 1.0 - (dist_to_center / 0.707)
                gate_reward = 30.0 + 50.0 * max(0.0, precision_ratio)
                R_total += gate_reward
            else:
                self.consecutive_misses += 1
                miss_severity = min(dist_to_center / 1.0, 1.0)
                R_total -= 5.0 + 10.0 * miss_severity

            self._generate_next_gate(self.current_target_idx)
            self.current_target_idx = (self.current_target_idx + 1) % self.MAX_VISIBLE_GATES

        return R_total

    # ================================================================
    # Termination / Truncation / Step / Reset (兩組完全相同)
    # ================================================================

    def _computeInfo(self):
        return {}

    def _checkCollision(self):
        drone_id = self.DRONE_IDS[0]
        for gate_id in self.gate_ids:
            pts = p.getContactPoints(bodyA=drone_id, bodyB=gate_id, physicsClientId=self.CLIENT)
            if len(pts) > 0:
                return True
        if self._getDroneStateVector(0)[2] < 0.1:
            return True
        return False

    def _computeTerminated(self):
        if self.consecutive_misses >= self.max_consecutive_misses:
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
        self.consecutive_misses = 0
        
        return super().reset(seed=seed, options=options)