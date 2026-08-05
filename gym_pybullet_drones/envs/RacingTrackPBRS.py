"""
Racing Track PBRS 環境

使用 Potential-Based Reward Shaping (PBRS) 的封閉賽道環境。

PBRS 理論基礎 (Ng et al., 1999):
  Shaping Reward F(s, a, s') = γ·Φ(s') - Φ(s)
  
  性質：
  - 保證不改變最優策略
  - Telescoping: 累積 shaping reward 只取決於起終點
  - 路徑無關

Racing Track 特殊考量：
  - 封閉迴路需要在完成一圈時正確處理 potential 重置
  - U-Turn 區段的 heading potential 需要特別注意
"""

import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class RacingTrackPBRS(BaseRLAviary):
    """使用 PBRS 的 Racing Track 環境。

    Potential Function 設計:
      Φ(s) = Φ_distance + Φ_centering + Φ_heading
      
      - Φ_distance:  -w_d · |d_plane|     (越接近 gate 平面越好)
      - Φ_centering: -w_c · d_lateral     (越對齊中心越好)
      - Φ_heading:   +w_h · cos(θ)        (越對準 gate 方向越好)
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
                 # Track 參數
                 straight_length: float = 8.0,
                 turn_radius: float = 3.0,
                 n_gates_straight: int = 3,
                 n_gates_turn: int = 3,
                 gate_height: float = 1.0,
                 height_variation: float = 0.3,
                 # PBRS 權重
                 pbrs_distance_weight: float = 1.0,
                 pbrs_centering_weight: float = 2.0,
                 pbrs_heading_weight: float = 1.5,
                 pbrs_gamma: float = 0.995,
                 # 其他參數
                 gate_half_width: float = 0.5,
                 gate_half_height: float = 0.5,
                 ):

        # PBRS 參數
        self.pbrs_distance_weight = pbrs_distance_weight
        self.pbrs_centering_weight = pbrs_centering_weight
        self.pbrs_heading_weight = pbrs_heading_weight
        self.pbrs_gamma = pbrs_gamma
        self.prev_potential = None

        # Track 幾何參數
        self.straight_length = straight_length
        self.turn_radius = turn_radius
        self.n_gates_straight = n_gates_straight
        self.n_gates_turn = n_gates_turn
        self.gate_height = gate_height
        self.height_variation = height_variation
        self.TOTAL_GATES_PER_LAP = 2 * n_gates_straight + 2 * n_gates_turn

        self.gate_half_width = gate_half_width
        self.gate_half_height = gate_half_height
        self.num_lookahead_gates = num_lookahead_gates

        self.gate_ids = []
        self.gate_poses = []
        self.gate_quats = []
        self.current_target_idx = 0
        self.current_lap = 0

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

        # 基礎 Reward 參數
        self.V_max = 5.0
        self.lambda_v = 10.0
        self.lambda_jerk = 0.005

        # Termination
        self.max_consecutive_misses = 2
        self.EPISODE_LEN_SEC = 90.0

        # 設定初始位置
        if initial_xyzs is None:
            initial_xyzs = np.array([[self.straight_length / 2, self.turn_radius, self.gate_height]])

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
    # Track 生成 (與 RacingTrack 相同)
    # ================================================================

    def _addObstacles(self):
        self.gate_ids = []
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'assets', 'hollow_gate.urdf')

        max_gates = self.TOTAL_GATES_PER_LAP + self.num_lookahead_gates
        for i in range(max_gates):
            gate_id = p.loadURDF(urdf_path, [0, 0, -100],
                                 p.getQuaternionFromEuler([0, 0, 0]),
                                 physicsClientId=self.CLIENT)
            self.gate_ids.append(gate_id)

        self._generate_track_gates()

    def _generate_track_gates(self):
        """生成膠囊形軌道的所有 gates。"""
        self.gate_poses = []
        self.gate_quats = []

        L = self.straight_length
        R = self.turn_radius
        H = self.gate_height
        h_var = self.height_variation

        # 北側直道 (向西飛行, yaw = π)
        for i in range(self.n_gates_straight):
            t = (i + 1) / (self.n_gates_straight + 1)
            x = L / 2 - L * t
            y = R
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = np.pi

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))

        # 西側 U-Turn (180° 連續左轉)
        for i in range(self.n_gates_turn):
            theta = np.pi * (i + 1) / (self.n_gates_turn + 1)
            x = -L / 2 - R * np.sin(theta)
            y = R * np.cos(theta)
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = np.pi + theta

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))

        # 南側直道 (向東飛行, yaw = 0)
        for i in range(self.n_gates_straight):
            t = (i + 1) / (self.n_gates_straight + 1)
            x = -L / 2 + L * t
            y = -R
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = 0

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))

        # 東側 U-Turn (180° 連續左轉回到起點)
        for i in range(self.n_gates_turn):
            theta = np.pi * (i + 1) / (self.n_gates_turn + 1)
            x = L / 2 + R * np.sin(theta)
            y = -R * np.cos(theta)
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = theta

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))

        # 設置 gate 位置
        for i, (pos, quat) in enumerate(zip(self.gate_poses, self.gate_quats)):
            if i < len(self.gate_ids):
                p.resetBasePositionAndOrientation(
                    self.gate_ids[i], pos, quat,
                    physicsClientId=self.CLIENT)

        self.current_target_idx = 0
        self.current_lap = 0

    def _get_segment_name(self, lap_idx):
        n_s = self.n_gates_straight
        n_t = self.n_gates_turn

        if lap_idx < n_s:
            return "north_straight"
        elif lap_idx < n_s + n_t:
            return "west_uturn"
        elif lap_idx < 2 * n_s + n_t:
            return "south_straight"
        else:
            return "east_uturn"

    # ================================================================
    # Observation (與 RacingTrack 相同)
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
                idx = (self.current_target_idx + j) % self.TOTAL_GATES_PER_LAP
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
    # PBRS: Potential Function
    # ================================================================

    def _compute_potential(self):
        """計算當前狀態的 Potential Value。

        Φ(s) = Φ_distance + Φ_centering + Φ_heading
        """
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]

        gate_pos = self.gate_poses[self.current_target_idx]
        gate_quat = self.gate_quats[self.current_target_idx]
        R_gate = Rotation.from_quat(gate_quat).as_matrix()

        # Gate-relative position
        p_G = R_gate.T @ (drone_pos - gate_pos)

        # === Φ_distance: 越接近 gate 平面越好 ===
        dist_to_plane = abs(p_G[0])
        Phi_distance = -self.pbrs_distance_weight * dist_to_plane

        # === Φ_centering: 越對齊中心越好 ===
        dist_lateral = np.sqrt(p_G[1]**2 + p_G[2]**2)
        Phi_centering = -self.pbrs_centering_weight * dist_lateral

        # === Φ_heading: 越對準 gate 方向越好 ===
        gate_normal = R_gate @ np.array([1, 0, 0])
        speed = np.linalg.norm(drone_vel)
        if speed > 0.5:
            drone_dir = drone_vel / speed
            cos_theta = np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0)
        else:
            drone_rpy = state[7:10]
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            drone_dir = R_drone @ np.array([1, 0, 0])
            cos_theta = np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0)

        Phi_heading = self.pbrs_heading_weight * cos_theta

        # === 總 Potential ===
        Phi_total = Phi_distance + Phi_centering + Phi_heading

        return Phi_total

    # ================================================================
    # Reward Function (PBRS)
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

        # === 速度懲罰 ===
        speed = np.linalg.norm(drone_vel)
        speed_penalty = self.lambda_v * (max(0.0, speed - self.V_max) ** 2)

        # === PBRS Shaping Reward ===
        current_potential = self._compute_potential()

        if self.prev_potential is not None:
            shaping_reward = self.pbrs_gamma * current_potential - self.prev_potential
        else:
            shaping_reward = 0.0

        self.prev_potential = current_potential

        # === 基礎 Reward ===
        r_approach = v_G[0]
        R_base = r_approach - speed_penalty

        # === 總 Reward ===
        R_total = R_base + shaping_reward

        # === Gate 穿越檢查 ===
        if p_G[0] > 0.0:
            self.total_gates_encountered += 1
            dist_to_center = np.sqrt(dist_lat ** 2 + dist_height ** 2)
            self.pass_distances.append(dist_to_center)

            if dist_lat < self.gate_half_width and dist_height < self.gate_half_height:
                self.passed_gates += 1
                self.consecutive_misses = 0
                precision_ratio = 1.0 - (dist_to_center / 0.707)
                gate_reward = 30.0 + 50.0 * max(0.0, precision_ratio)
                R_total += gate_reward
            else:
                self.consecutive_misses += 1
                miss_severity = min(dist_to_center / 1.0, 1.0)
                R_total -= 5.0 + 10.0 * miss_severity

            # 重置 potential（新 gate 的 potential 與舊 gate 不可比）
            self.prev_potential = None

            # 前進到下一個 gate
            self.current_target_idx = (self.current_target_idx + 1) % self.TOTAL_GATES_PER_LAP
            if self.current_target_idx == 0:
                self.current_lap += 1

        return R_total

    # ================================================================
    # Termination / Step / Reset
    # ================================================================

    def _computeInfo(self):
        return {
            "current_lap": self.current_lap,
            "segment": self._get_segment_name(self.current_target_idx),
        }

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
        if np.linalg.norm(state[0:3] - gate_pos) > 15.0:
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
        info["laps_completed"] = self.current_lap
        info["avg_pass_dist"] = np.mean(self.pass_distances) if len(self.pass_distances) > 0 else 0.0

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # 重置 PBRS 狀態
        self.prev_potential = None

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
        self.current_lap = 0

        obs, info = super().reset(seed=seed, options=options)

        # 重新生成軌道
        self._generate_track_gates()

        return obs, info