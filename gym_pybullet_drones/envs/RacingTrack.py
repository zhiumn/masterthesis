"""
場景 C: Racing Track (封閉賽道)

與 Sharp-Turn Corridor 的結構性差異：
┌───────────────────────────────────────────────────────────┐
│  特性            Sharp-Turn           Racing Track        │
│  ──────────────────────────────────────────────────────── │
│  軌道拓撲        開放直線             封閉迴路 (膠囊形)     │
│  轉向模式        間歇大角度           連續曲線 (U-Turn)     │
│  Yaw 變化        離散 ±34°            連續 0°→180°→0°     │
│  Gate 間距       均勻隨機             曲率依賴 (彎道更密)   │
│  累積 Yaw/圈     ~240°                720° (兩個 180° 轉)  │
│  ──────────────────────────────────────────────────────── │
│  Reward Function: 完全相同 (確保公平比較)                  │
│  Observation:     完全相同                                │
│  PPO 超參數:      完全相同                                │
└───────────────────────────────────────────────────────────┘

設計理由：
1. 連續曲率變化 vs 離散轉折 → 測試 Smooth bonus 的價值
2. 封閉迴路 → 測試長期軌跡追蹤能力
3. U-Turn 需要連續大角度轉向 → Heading bonus 的不同使用模式
"""

import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class RacingTrack(BaseRLAviary):
    """場景 C: Racing Track (封閉賽道)

    膠囊形軌道結構：
                        ┌─────────────────────┐
                       /    U-Turn North       \
                      │  G3 ← G2 ← G1 ← START  │
                      │                        │
            S-Curve   │ G4                 G12 │   S-Curve
             Entry    │ ↓                   ↑  │    Exit
                      │ G5                 G11 │
                      │                        │
                      │  G6 → G7 → G8 → G9 → G10
                       \    U-Turn South       /
                        └─────────────────────┘

    參數：
      - 直道長度: 8m (可調)
      - U-Turn 半徑: 3m (CrazyFlie 可執行的最小合理半徑)
      - 每段直道 Gates: 3
      - 每個 U-Turn Gates: 3
      - 總 Gates: 12 (單圈)
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
                 use_dynamic_reward: bool = True,
                 # Track 參數
                 straight_length: float = 8.0,
                 turn_radius: float = 3.0,
                 n_gates_straight: int = 3,
                 n_gates_turn: int = 3,
                 gate_height: float = 1.0,
                 height_variation: float = 0.3,
                 # Reward 參數
                 gate_half_width: float = 0.5,
                 gate_half_height: float = 0.5,
                 heading_bonus_scale: float = 0.5,
                 centering_bonus_scale: float = 1.5,
                 ):

        self.use_dynamic_reward = use_dynamic_reward

        # Track 幾何參數
        self.straight_length = straight_length
        self.turn_radius = turn_radius
        self.n_gates_straight = n_gates_straight
        self.n_gates_turn = n_gates_turn
        self.gate_height = gate_height
        self.height_variation = height_variation
        self.gate_half_width = gate_half_width
        self.gate_half_height = gate_half_height
        self.TOTAL_GATES_PER_LAP = 2 * n_gates_straight + 2 * n_gates_turn

        self.num_lookahead_gates = num_lookahead_gates

        self.gate_ids = []
        self.gate_poses = []
        self.gate_quats = []
        self.current_target_idx = 0
        self.current_lap = 0
        self.total_gates_in_buffer = 0

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
        # Reward 參數 (與 SharpTurnCorridor 完全相同)
        # ============================================================
        self.w_lat = 2.0
        self.w_z = 2.0
        self.w_heading = 1.0
        self.V_max = 5.0
        self.lambda_v = 10.0
        self.lambda_jerk = 0.005

        self.centering_bonus_scale = centering_bonus_scale
        self.heading_bonus_scale = heading_bonus_scale
        self.smooth_bonus_scale = 0.3

        self.proximity_sharpness = 0.8
        self.static_proximity_weight = 0.5

        self.max_consecutive_misses = 2
        self.EPISODE_LEN_SEC = 90.0  # 比 corridor 長，因為要跑圈

        # 設定初始位置 (北側直道起點)
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
    # 環境建構 — Racing Track 專用
    # ================================================================

    def _addObstacles(self):
        """載入所有 gate URDF 並生成軌道。"""
        self.gate_ids = []
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'assets', 'hollow_gate.urdf')

        # 預載足夠的 gates (單圈 + lookahead buffer)
        max_gates = self.TOTAL_GATES_PER_LAP + self.num_lookahead_gates
        for i in range(max_gates):
            gate_id = p.loadURDF(urdf_path, [0, 0, -100],
                                 p.getQuaternionFromEuler([0, 0, 0]),
                                 physicsClientId=self.CLIENT)
            self.gate_ids.append(gate_id)

        self._generate_track_gates()

    def _generate_track_gates(self):
        """生成膠囊形軌道的所有 gates。

        軌道結構 (俯視圖):
          - 北側直道: 從東向西飛 (yaw = π)
          - 西側 U-Turn: 180° 左轉
          - 南側直道: 從西向東飛 (yaw = 0)
          - 東側 U-Turn: 180° 左轉回到起點
        """
        self.gate_poses = []
        self.gate_quats = []

        L = self.straight_length
        R = self.turn_radius
        H = self.gate_height
        h_var = self.height_variation

        gate_idx = 0

        # ============================================================
        # 北側直道 (G1-G3): 向西飛行, yaw = π
        # ============================================================
        for i in range(self.n_gates_straight):
            t = (i + 1) / (self.n_gates_straight + 1)
            x = L / 2 - L * t  # 從 +L/2 到 -L/2
            y = R
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = np.pi  # 面向西

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))
            gate_idx += 1

        # ============================================================
        # 西側 U-Turn (G4-G6): 180° 連續左轉
        # ============================================================
        for i in range(self.n_gates_turn):
            theta = np.pi * (i + 1) / (self.n_gates_turn + 1)  # 0 → π
            x = -L / 2 - R * np.sin(theta)
            y = R * np.cos(theta)  # R → 0 → -R
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = np.pi + theta  # π → 2π (切線方向)

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))
            gate_idx += 1

        # ============================================================
        # 南側直道 (G7-G9): 向東飛行, yaw = 0
        # ============================================================
        for i in range(self.n_gates_straight):
            t = (i + 1) / (self.n_gates_straight + 1)
            x = -L / 2 + L * t  # 從 -L/2 到 +L/2
            y = -R
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = 0  # 面向東

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))
            gate_idx += 1

        # ============================================================
        # 東側 U-Turn (G10-G12): 180° 連續左轉回到起點
        # ============================================================
        for i in range(self.n_gates_turn):
            theta = np.pi * (i + 1) / (self.n_gates_turn + 1)  # 0 → π
            x = L / 2 + R * np.sin(theta)
            y = -R * np.cos(theta)  # -R → 0 → R
            z = H + self.np_random.uniform(-h_var, h_var)
            z = max(0.6, z)
            yaw = theta  # 0 → π (切線方向)

            self.gate_poses.append(np.array([x, y, z]))
            self.gate_quats.append(p.getQuaternionFromEuler([0, 0, yaw]))
            gate_idx += 1

        # 設置 gate 位置
        for i, (pos, quat) in enumerate(zip(self.gate_poses, self.gate_quats)):
            if i < len(self.gate_ids):
                p.resetBasePositionAndOrientation(
                    self.gate_ids[i], pos, quat,
                    physicsClientId=self.CLIENT)

        self.current_target_idx = 0
        self.current_lap = 0
        self.total_gates_in_buffer = len(self.gate_poses)

    def _get_gate_index_in_lap(self, global_idx):
        """將全局 gate index 轉換為圈內 index。"""
        return global_idx % self.TOTAL_GATES_PER_LAP

    def _get_segment_name(self, lap_idx):
        """根據圈內 index 返回所在段落名稱。"""
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
    # Observation (與 SharpTurnCorridor 完全相同)
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
                # 使用 modulo 處理封閉軌道
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
    # Reward Function (與 SharpTurnCorridor 完全相同)
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

        r_approach = v_G[0]
        dist_lat = abs(p_G[1])
        dist_height = abs(p_G[2])
        d_plane = abs(p_G[0])

        gate_normal = R_gate @ np.array([1, 0, 0])
        speed = np.linalg.norm(drone_vel)
        if speed > 0.5:
            drone_dir = drone_vel / speed
        else:
            drone_rpy = state[7:10]
            R_drone = Rotation.from_euler('xyz', drone_rpy).as_matrix()
            drone_dir = R_drone @ np.array([1, 0, 0])

        # 下一個門的方向 (封閉軌道用 modulo)
        next_idx = (self.current_target_idx + 1) % self.TOTAL_GATES_PER_LAP
        next_gate_pos = self.gate_poses[next_idx]
        next_vec = next_gate_pos - drone_pos
        next_dir = next_vec / (np.linalg.norm(next_vec) + 1e-6)

        cos_theta = np.clip(np.dot(drone_dir, next_dir), -1.0, 1.0)
        theta_err = np.arccos(np.clip(np.dot(drone_dir, gate_normal), -1.0, 1.0))

        speed_penalty = self.lambda_v * (max(0.0, speed - self.V_max) ** 2)

        R_acquire = r_approach - self.w_heading * theta_err
        R_align = r_approach - (self.w_lat * (dist_lat ** 2) + self.w_z * (dist_height ** 2))
        R_base = 0.5 * R_acquire + 0.5 * R_align

        approach_speed = np.clip(v_G[0], 0.0, 3.0)
        centering_quality = np.exp(-(dist_lat**2 + dist_height**2) / 0.25) * (approach_speed / 3.0)
        heading_quality = max(0.0, cos_theta) * (approach_speed / 3.0)
        smooth_quality = approach_speed * (1.0 - 0.5 * (dist_lat**2 + dist_height**2))

        if self.use_dynamic_reward:
            proximity = np.exp(-self.proximity_sharpness * d_plane)
            w_centering = proximity
            w_heading = 1.0 - proximity
            w_smooth = 1.0
        else:
            w_centering = self.static_proximity_weight
            w_heading = 1.0 - self.static_proximity_weight
            w_smooth = 1.0

        R_centering = self.centering_bonus_scale * w_centering * centering_quality
        R_heading = self.heading_bonus_scale * w_heading * heading_quality
        R_smooth = self.smooth_bonus_scale * w_smooth * smooth_quality

        R_total = R_base + R_centering + R_heading + R_smooth - speed_penalty

        # 穿越 gate 檢查
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

            # 前進到下一個 gate (封閉軌道用 modulo)
            self.current_target_idx = (self.current_target_idx + 1) % self.TOTAL_GATES_PER_LAP

            # 計算完成圈數
            if self.current_target_idx == 0:
                self.current_lap += 1

        return R_total

    # ================================================================
    # Termination / Truncation / Step / Reset
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
        if np.linalg.norm(state[0:3] - gate_pos) > 15.0:  # 稍微放寬，因為軌道較大
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

        # 重新生成軌道 (含高度隨機化)
        self._generate_track_gates()

        return obs, info