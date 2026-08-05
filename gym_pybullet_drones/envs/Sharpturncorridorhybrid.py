"""
Sharp-Turn Corridor Hybrid 環境

結合 Additive 的 R_base 與 PBRS 形式的 Bonus。

設計理念：
  - R_base: 保留原有的 R_acquire + R_align（與 Additive 完全相同）
  - Bonus:  使用 PBRS 形式 γ·Φ(s') - Φ(s)（與純 PBRS 相同）

這樣的設計確保公平比較：
  - Additive: R_base + bonus(s)
  - Hybrid:   R_base + [γ·Φ(s') - Φ(s)]
  - 唯一差異是 bonus 的計算方式

Reward 結構：
  R_total = R_base 
          + PBRS_centering 
          + PBRS_heading 
          + PBRS_smooth 
          + R_pass
"""

import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class SharpTurnCorridorHybrid(BaseRLAviary):
    """Hybrid 環境：Additive R_base + PBRS Bonus。

    R_base 與 SharpTurnCorridor 完全相同：
      R_base = 0.5 * R_acquire + 0.5 * R_align

    Bonus 使用 PBRS 形式：
      PBRS_centering = γ·Φ_c(s') - Φ_c(s)
      PBRS_heading   = γ·Φ_h(s') - Φ_h(s)
      PBRS_smooth    = γ·Φ_s(s') - Φ_s(s)
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 num_lookahead_gates: int = 2,
                 # PBRS Bonus 權重（與純 PBRS 相同）
                 pbrs_centering_weight: float = 2.0,
                 pbrs_heading_weight: float = 1.5,
                 pbrs_smooth_weight: float = 1.0,
                 pbrs_gamma: float = 0.995,
                 # 其他參數
                 gate_half_width: float = 0.5,
                 ):

        # PBRS Bonus 參數
        self.pbrs_centering_weight = pbrs_centering_weight
        self.pbrs_heading_weight = pbrs_heading_weight
        self.pbrs_smooth_weight = pbrs_smooth_weight
        self.pbrs_gamma = pbrs_gamma

        # Previous potentials for PBRS
        self.prev_phi_centering = None
        self.prev_phi_heading = None
        self.prev_phi_smooth = None

        self.gate_half_width = gate_half_width
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

        # R_base 參數（與 SharpTurnCorridor 完全相同）
        self.w_lat = 2.0
        self.w_z = 2.0
        self.w_heading = 1.0
        self.V_max = 5.0
        self.lambda_v = 10.0
        self.lambda_jerk = 0.005

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
    # Gate 生成（與 SharpTurnCorridor 完全相同）
    # ================================================================

    def _addObstacles(self):
        self.gate_ids = []
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'assets', 'hollow_gate.urdf')
        for i in range(self.MAX_VISIBLE_GATES):
            gate_id = p.loadURDF(urdf_path, [0, 0, -100],
                                 p.getQuaternionFromEuler([0, 0, 0]),
                                 physicsClientId=self.CLIENT)
            self.gate_ids.append(gate_id)
        self._generate_initial_gates()

    def _generate_initial_gates(self):
        """生成初始門框 — 急彎版本"""
        self.gate_poses = []
        self.gate_quats = []
        cur_pos = self.INIT_XYZS[0].copy() if len(self.INIT_XYZS) > 0 else np.array([0., 0., 1.0])
        cur_yaw = 0.0

        for i in range(self.MAX_VISIBLE_GATES):
            dist = self.np_random.uniform(1.8, 5.0)
            yaw_shift = self.np_random.uniform(-0.6, 0.6)
            z_shift = self.np_random.uniform(-0.8, 0.8)

            cur_yaw += yaw_shift
            dx = dist * np.cos(cur_yaw)
            dy = dist * np.sin(cur_yaw)
            cur_pos = cur_pos + np.array([dx, dy, z_shift])
            cur_pos[2] = max(0.6, cur_pos[2])

            self.gate_poses.append(cur_pos.copy())
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, cur_yaw]))
            p.resetBasePositionAndOrientation(
                self.gate_ids[i], cur_pos, self.gate_quats[-1],
                physicsClientId=self.CLIENT)

        self.current_target_idx = 0

    def _generate_next_gate(self, replace_idx):
        """生成下一扇門框"""
        last_idx = (self.current_target_idx - 1) % self.MAX_VISIBLE_GATES
        last_pos = self.gate_poses[last_idx]
        last_quat = self.gate_quats[last_idx]
        last_yaw = p.getEulerFromQuaternion(last_quat)[2]

        dist = self.np_random.uniform(1.8, 5.0)
        yaw_shift = self.np_random.uniform(-0.6, 0.6)
        z_shift = self.np_random.uniform(-0.8, 0.8)

        new_yaw = last_yaw + yaw_shift
        dx = dist * np.cos(new_yaw)
        dy = dist * np.sin(new_yaw)
        new_pos = last_pos + np.array([dx, dy, z_shift])
        new_pos[2] = max(0.6, new_pos[2])
        new_quat = p.getQuaternionFromEuler([0, 0, new_yaw])

        self.gate_poses[replace_idx] = new_pos
        self.gate_quats[replace_idx] = new_quat
        p.resetBasePositionAndOrientation(
            self.gate_ids[replace_idx], new_pos, new_quat,
            physicsClientId=self.CLIENT)

    # ================================================================
    # Observation（與 SharpTurnCorridor 完全相同）
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
            elif self.ACT_TYPE == ActionType.PID:
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
    # PBRS Potential Functions
    # ================================================================

    def _compute_potentials(self):
        """計算三個 bonus 的 potential values。

        Returns:
            tuple: (Φ_centering, Φ_heading, Φ_smooth)
        """
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        drone_rpy = state[7:10]

        gate_pos = self.gate_poses[self.current_target_idx]
        gate_quat = self.gate_quats[self.current_target_idx]
        R_gate = Rotation.from_quat(gate_quat).as_matrix()

        # Gate-relative position and velocity
        p_G = R_gate.T @ (drone_pos - gate_pos)
        v_G = R_gate.T @ drone_vel

        # === Φ_centering: 越對齊中心越好 ===
        dist_lateral = np.sqrt(p_G[1]**2 + p_G[2]**2)
        Phi_centering = -dist_lateral  # 範圍: (-∞, 0]

        # === Φ_heading: 越對準 gate 方向越好 ===
        # 使用下一個門作為 heading 目標（與 Additive 一致）
        next_idx = (self.current_target_idx + 1) % self.MAX_VISIBLE_GATES
        next_gate_pos = self.gate_poses[next_idx]
        next_vec = next_gate_pos - drone_pos
        next_dist = np.linalg.norm(next_vec)
        
        if next_dist > 0.1:
            next_dir = next_vec / next_dist
        else:
            next_dir = R_gate @ np.array([1, 0, 0])

        speed = np.linalg.norm(drone_vel)
        if speed > 0.5:
            drone_dir = drone_vel / speed
        else:
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            drone_dir = R_drone @ np.array([1, 0, 0])

        cos_theta = np.clip(np.dot(drone_dir, next_dir), -1.0, 1.0)
        Phi_heading = cos_theta  # 範圍: [-1, 1]

        # === Φ_smooth: 接近速度越好 ===
        approach_speed = v_G[0]  # 沿 gate 法向量的速度分量
        alignment = max(0.0, cos_theta)
        Phi_smooth = approach_speed * alignment / 3.0  # 正規化

        return Phi_centering, Phi_heading, Phi_smooth

    # ================================================================
    # Reward Function (Hybrid: Additive R_base + PBRS Bonus)
    # ================================================================

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]

        gate_pos = self.gate_poses[self.current_target_idx]
        gate_quat = self.gate_quats[self.current_target_idx]
        R_gate = Rotation.from_quat(gate_quat).as_matrix()

        p_G = R_gate.T @ (drone_pos - gate_pos)
        v_G = R_gate.T @ drone_vel
        dist_lat = abs(p_G[1])
        dist_height = abs(p_G[2])

        # ================================================================
        # R_base（與 SharpTurnCorridor 完全相同）
        # ================================================================
        r_approach = v_G[0]
        gate_normal = R_gate @ np.array([1, 0, 0])
        speed = np.linalg.norm(drone_vel)
        if speed > 0.5:
            drone_dir = drone_vel / speed
        else:
            drone_rpy = state[7:10]
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            drone_dir = R_drone @ np.array([1, 0, 0])
            
        next_idx = (self.current_target_idx + 1) % self.MAX_VISIBLE_GATES
        next_gate_pos = self.gate_poses[next_idx]
        next_vec = next_gate_pos - drone_pos
        next_dir = next_vec / (np.linalg.norm(next_vec) + 1e-6)
        
        theta_err = np.arccos(np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0))
        speed_penalty = self.lambda_v * (max(0.0, speed - self.V_max) ** 2)

        R_acquire = r_approach - self.w_heading * theta_err
        R_align = r_approach - (self.w_lat * (dist_lat ** 2) + self.w_z * (dist_height ** 2))
        R_base = 0.5 * R_acquire + 0.5 * R_align - speed_penalty

        # ================================================================
        # PBRS Bonus
        # ================================================================

        # 計算當前 potentials
        curr_phi_c, curr_phi_h, curr_phi_s = self._compute_potentials()

        # PBRS shaping rewards
        if self.prev_phi_centering is not None:
            pbrs_centering = self.pbrs_centering_weight * (
                self.pbrs_gamma * curr_phi_c - self.prev_phi_centering
            )
            pbrs_heading = self.pbrs_heading_weight * (
                self.pbrs_gamma * curr_phi_h - self.prev_phi_heading
            )
            pbrs_smooth = self.pbrs_smooth_weight * (
                self.pbrs_gamma * curr_phi_s - self.prev_phi_smooth
            )
        else:
            pbrs_centering = 0.0
            pbrs_heading = 0.0
            pbrs_smooth = 0.0

        # 更新 previous potentials
        self.prev_phi_centering = curr_phi_c
        self.prev_phi_heading = curr_phi_h
        self.prev_phi_smooth = curr_phi_s

        # ================================================================
        # 總 Reward
        # ================================================================

        R_total = R_base + pbrs_centering + pbrs_heading + pbrs_smooth

        # ================================================================
        # Gate 穿越檢查
        # ================================================================

        if p_G[0] > 0.0:
            self.total_gates_encountered += 1
            dist_to_center = np.sqrt(dist_lat ** 2 + dist_height ** 2)
            self.pass_distances.append(dist_to_center)

            if dist_lat < self.gate_half_width and dist_height < self.gate_half_width:
                self.passed_gates += 1
                self.consecutive_misses = 0
                precision_ratio = 1.0 - (dist_to_center / 0.707)
                gate_reward = 30.0 + 50.0 * max(0.0, precision_ratio)
                R_total += gate_reward
            else:
                self.consecutive_misses += 1
                miss_severity = min(dist_to_center / 1.0, 1.0)
                R_total -= 5.0 + 10.0 * miss_severity

            # 重置 potentials（新 gate 的 potential 與舊 gate 不可比）
            self.prev_phi_centering = None
            self.prev_phi_heading = None
            self.prev_phi_smooth = None

            self._generate_next_gate(self.current_target_idx)
            self.current_target_idx = (self.current_target_idx + 1) % self.MAX_VISIBLE_GATES

        return R_total

    # ================================================================
    # Termination / Step / Reset
    # ================================================================

    def _computeInfo(self):
        return {}

    def _checkCollision(self):
        drone_id = self.DRONE_IDS[0]
        for gate_id in self.gate_ids:
            pts = p.getContactPoints(bodyA=drone_id, bodyB=gate_id,
                                     physicsClientId=self.CLIENT)
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
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
        # 重置 PBRS potentials
        self.prev_phi_centering = None
        self.prev_phi_heading = None
        self.prev_phi_smooth = None

        # 重置 performance tracking
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