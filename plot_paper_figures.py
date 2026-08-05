"""
Publication-Quality Trajectory Visualization (Revised)
======================================================

Figure A: CLGC 2x2 3D trajectories (all 4 methods)
  - Thicker, more visible gate frames with fill
  - Gate index labels at key positions (G1, G5, G10, G15...)

Figure B: HCGS local turning segment, Fixed vs Step side-by-side
  - Zoomed into high-curvature segment (not full trajectory)
  - Left subplot = Fixed, Right subplot = Step
  - Visible quiver velocity arrows
  - Only label key gate indices

Figure C: dy-dz crossing scatter (2x2) with corrected statistics
  - N = crossing events (clarified in title)
  - Statistics: mean|d|, median|d|, 90th percentile, inside rate
  - Density contours added

Figure D: Fixed vs Step overlay — density contour version
Figure E: Radial error CDF — Fixed vs Step
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.SharpTurnCorridor import SharpTurnCorridor
from gym_pybullet_drones.envs.RacingTrackAblation import RacingTrackAblation
from gym_pybullet_drones.envs.Sharpturncorridorhybrid import SharpTurnCorridorHybrid
from gym_pybullet_drones.envs.Racingtrackhybrid import RacingTrackHybrid
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

OUT_DIR = 'trajectory_plots'
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = {
    'Base': {
        'sharp_model': 'models/sharp_ablation_A0_seed0/best_model.zip',
        'racing_model': 'models/racing_ablation_A0_seed0/best_model.zip',
        'color': '#7f8c8d', 'label': 'Base Only', 'ls': '--', 'marker': '^',
        'sharp_env': SharpTurnCorridor, 'racing_env': RacingTrackAblation,
    },
    'PBRS': {
        'sharp_model': 'models/sharp_hybrid_seed0/best_model.zip',
        'racing_model': 'models/racing_hybrid_seed0/best_model.zip',
        'color': '#e67e22', 'label': 'PBRS', 'ls': '-.', 'marker': 'v',
        'sharp_env': SharpTurnCorridorHybrid, 'racing_env': RacingTrackHybrid,
    },
    'Fixed': {
        'sharp_model': 'models/sharp_ablation_A7-S_seed0/best_model.zip',
        'racing_model': 'models/racing_track_static_seed0/best_model.zip',
        'color': '#27ae60', 'label': 'Additive (Fixed)', 'ls': '-', 'marker': 'o',
        'sharp_env': SharpTurnCorridor, 'racing_env': RacingTrackAblation,
    },
    'Step': {
        'sharp_model': 'models/sharp_ablation_P5_seed0/best_model.zip',
        'racing_model': 'models/racing_ablation_P5_seed0/best_model.zip',
        'color': '#8e44ad', 'label': 'Additive (Step)', 'ls': '-', 'marker': 's',
        'sharp_env': SharpTurnCorridor, 'racing_env': RacingTrackAblation,
    },
}


# ─── Helper ───────────────────────────────────────────────────────────
def run_episode(env_class, model_path, seed, override_gates=None):
    """Run one episode; return trajectory, velocities, gates, crossings."""
    env = env_class(gui=False, num_lookahead_gates=2,
                    obs=ObservationType.KIN, act=ActionType.RPM)
    model = PPO.load(model_path)
    obs, info = env.reset(seed=seed)

    # Force identical gate layouts if override_gates is provided
    if override_gates is not None:
        env.gate_poses = [g['pos'] for g in override_gates]
        env.gate_quats = [g['quat'] for g in override_gates]
        import pybullet as p_mod
        for i, g in enumerate(override_gates):
            if i < len(env.gate_ids):
                p_mod.resetBasePositionAndOrientation(
                    env.gate_ids[i], g['pos'], g['quat'], physicsClientId=env.CLIENT)
        obs = env._computeObs()

    traj, vels = [], []
    gates = []

    def snapshot_gates():
        for i in range(len(env.gate_poses)):
            pos = env.gate_poses[i]
            quat = env.gate_quats[i]
            if not any(np.linalg.norm(g['pos'] - pos) < 0.1 for g in gates):
                gates.append({'pos': pos.copy(), 'quat': quat})

    snapshot_gates()

    n_slots = 50
    prev_fwd = [None] * n_slots
    prev_y = [None] * n_slots
    prev_z = [None] * n_slots
    crossings = []

    while True:
        state = env._getDroneStateVector(0)
        pos, vel = state[0:3], state[10:13]
        traj.append(pos.copy())
        vels.append(vel.copy())

        for i, g in enumerate(gates):
            if i >= n_slots:
                break
            R_g = Rotation.from_quat(g['quat']).as_matrix()
            p_G = R_g.T @ (pos - g['pos'])
            if prev_fwd[i] is not None and prev_fwd[i] <= 0 and p_G[0] > 0:
                t = -prev_fwd[i] / (p_G[0] - prev_fwd[i] + 1e-9)
                cy = prev_y[i] + t * (p_G[1] - prev_y[i])
                cz = prev_z[i] + t * (p_G[2] - prev_z[i])
                crossings.append((i, cy, cz))
            prev_fwd[i] = p_G[0]
            prev_y[i] = p_G[1]
            prev_z[i] = p_G[2]

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        snapshot_gates()
        if terminated or truncated:
            break

    env.close()
    return (np.array(traj), np.array(vels), gates, crossings,
            info.get('gates_passed', 0))


def draw_gate_3d(ax, gate, color='#2c3e50', alpha=0.25, lw=2.5,
                 fill_color='#aed6f1'):
    """Draw a visible 3D gate rectangle with fill."""
    gp = gate['pos']
    yaw = Rotation.from_quat(gate['quat']).as_euler('xyz')[2]
    hw, hh = 0.5, 0.5
    dy = np.array([-np.sin(yaw), np.cos(yaw), 0]) * hw
    dz = np.array([0, 0, 1]) * hh
    c = [gp - dy - dz, gp + dy - dz, gp + dy + dz, gp - dy + dz]
    # fill
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [list(map(tuple, c))]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=fill_color,
                            edgecolor=color, linewidth=lw)
    ax.add_collection3d(poly)


# =====================================================================
# FIGURE A  --  CLGC 2x2 3D trajectories
# =====================================================================
def figure_A(seed=42):
    print(f"\n{'='*60}")
    print(f"Figure A: CLGC 3D Trajectory Comparison  (seed={seed})")
    print(f"{'='*60}")

    results = {}
    shared_gates = None
    for name, m in METHODS.items():
        print(f"  Running {name}...")
        traj, vels, gates, crossings, gp = run_episode(
            m['racing_env'], m['racing_model'], seed, override_gates=shared_gates)
        results[name] = dict(traj=traj, vels=vels, gates=gates,
                             crossings=crossings, gp=gp)
        if shared_gates is None:
            shared_gates = gates

    # uniform coordinate range
    all_pts = np.vstack([r['traj'] for r in results.values()])
    xmin, xmax = all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1
    ymin, ymax = all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1
    zmin, zmax = 0, max(2.5, all_pts[:, 2].max() + 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                              subplot_kw={'projection': '3d'})
    method_order = ['Base', 'PBRS', 'Fixed', 'Step']
    titles = ['(a) Base Only', '(b) PBRS',
              '(c) Additive (Fixed)', '(d) Additive (Step)']

    # Gate index labels: show every 4th gate
    label_every = 4

    for idx, (mkey, title) in enumerate(zip(method_order, titles)):
        ax = axes[idx // 2][idx % 2]
        r = results[mkey]
        traj = r['traj']
        own_gates = r['gates']
        c = METHODS[mkey]['color']

        # draw gates with fill + gate index labels
        for gi, g in enumerate(own_gates):
            draw_gate_3d(ax, g, color='#1a5276', alpha=0.3,
                         lw=2.5, fill_color='#aed6f1')
            # label key gates
            if gi % label_every == 0 or gi == len(own_gates) - 1:
                gp = g['pos']
                ax.text(gp[0], gp[1], gp[2] + 0.6,
                        f'G{gi+1}', fontsize=7, ha='center',
                        color='#1a5276', fontweight='bold')

        # trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=c, linewidth=2.0, alpha=0.85)

        # start / end
        ax.scatter(*traj[0], color='lime', s=70, marker='o',
                   edgecolors='black', zorder=10, label='Start')
        ax.scatter(*traj[-1], color='red', s=70, marker='X',
                   edgecolors='black', zorder=10, label='End')

        # crossing markers (use own gates)
        for ci, cy, cz in r['crossings']:
            if ci < len(own_gates):
                gpos = own_gates[ci]['pos']
                R_g = Rotation.from_quat(own_gates[ci]['quat']).as_matrix()
                world_pt = gpos + R_g @ np.array([0, cy, cz])
                ax.scatter(*world_pt, color=c, s=40, marker='.',
                           alpha=0.8, zorder=8)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Adjust aspect ratio: True physical ratio is (Dx, Dy, Dz). 
        # We multiply Dz by 1.8 to keep it readable, but much less stretched than default.
        ax.set_box_aspect((xmax - xmin, ymax - ymin, (zmax - zmin) * 1.8))
        
        ax.view_init(elev=35, azim=-60)
        ax.tick_params(labelsize=7)

    fig.suptitle('Figure A: CLGC Scenario -- 3D Flight Trajectory Comparison',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, f'fig_A_CLGC_3d_seed{seed}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {path}")


# =====================================================================
# FIGURE B  --  HCGS local turning segment, Fixed vs Step side-by-side
# =====================================================================
def figure_B(seed=42, gate_range=(5, 14)):
    """Zoomed-in comparison of a high-curvature segment."""
    g_start, g_end = gate_range
    print(f"\n{'='*60}")
    print(f"Figure B: HCGS Local Turning Segment G{g_start+1}-G{g_end+1} (seed={seed})")
    print(f"{'='*60}")

    results = {}
    shared_gates = None
    for name, m in METHODS.items():
        print(f"  Running {name}...")
        traj, vels, gates, crossings, gp = run_episode(
            m['sharp_env'], m['sharp_model'], seed, override_gates=shared_gates)
        results[name] = dict(traj=traj, vels=vels, gates=gates,
                             crossings=crossings, gp=gp)
        if shared_gates is None:
            shared_gates = gates

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        m = METHODS[name]
        c = m['color']
        traj = r['traj']
        vels = r['vels']
        own_gates = r['gates']
        
        ref_gates = results['Fixed']['gates']
        n_gates = len(ref_gates)
        # clamp range
        gs = min(g_start, n_gates - 1)
        ge = min(g_end, n_gates - 1)

        segment_gates = ref_gates[gs:ge+1]
        seg_xs = [g['pos'][0] for g in segment_gates]
        seg_ys = [g['pos'][1] for g in segment_gates]
        pad = 2.0
        xlim = (min(seg_xs) - pad, max(seg_xs) + pad)
        ylim = (min(seg_ys) - pad, max(seg_ys) + pad)

        arrow_every = 8

        # centre-line (gray dashed) through segment gates
        cx_list = [own_gates[i]['pos'][0] for i in range(gs, min(ge+1, len(own_gates)))]
        cy_list = [own_gates[i]['pos'][1] for i in range(gs, min(ge+1, len(own_gates)))]
        ax.plot(cx_list, cy_list, 'k--', alpha=0.35, linewidth=1.5,
                label='Centre-line', zorder=1)

        # draw gates in segment
        for gi in range(gs, min(ge+1, len(own_gates))):
            g = own_gates[gi]
            gx, gy = g['pos'][0], g['pos'][1]
            yaw = Rotation.from_quat(g['quat']).as_euler('xyz')[2]
            hw, ht = 0.5, 0.06
            corners = np.array([[ht, hw], [-ht, hw], [-ht, -hw], [ht, -hw]])
            R2 = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
            cw = (R2 @ corners.T).T + np.array([gx, gy])
            poly = patches.Polygon(cw, closed=True,
                                    facecolor='#d5e8f0', edgecolor='#1a5276',
                                    linewidth=2.0, alpha=0.6, zorder=2)
            ax.add_patch(poly)

            # label key gates only
            local_i = gi - gs
            if local_i % 3 == 0 or gi == gs or gi == ge:
                ax.text(gx, gy + 0.3, f'G{gi+1}', fontsize=9, ha='center',
                        color='#1a5276', fontweight='bold', zorder=9)

        # trajectory line
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2.5,
                alpha=0.85, label=METHODS[name]['label'], zorder=4)

        # quiver velocity arrows
        for k in range(0, len(traj), arrow_every):
            vx, vy = vels[k, 0], vels[k, 1]
            spd = np.sqrt(vx**2 + vy**2)
            if spd < 0.3:
                continue
            if not (xlim[0] <= traj[k, 0] <= xlim[1] and
                    ylim[0] <= traj[k, 1] <= ylim[1]):
                continue
            scale = 0.18
            ax.quiver(traj[k, 0], traj[k, 1], vx * scale, vy * scale,
                      angles='xy', scale_units='xy', scale=1,
                      color=c, alpha=0.7, width=0.008, headwidth=4,
                      headlength=5, zorder=5)

        # start / end markers
        ax.scatter(traj[0, 0], traj[0, 1], color='lime', s=100,
                   marker='o', edgecolors='black', zorder=10, label='Start')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=100,
                   marker='X', edgecolors='black', zorder=10, label='End')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        title = f"{name}"
        if r['crossings'] is not None:
            title += f" (Passed {r['gp']} gates)"
        ax.set_title(title, fontsize=14, fontweight='bold', color=c)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Figure B: HCGS Anticipatory Turning (Seed {seed})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(OUT_DIR, f'fig_B_HCGS_topdown_seed{seed}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_path}")


# =====================================================================
# FIGURE C  --  dy-dz crossing scatter, corrected statistics
# =====================================================================
def figure_C(n_episodes=50):
    print(f"\n{'='*60}")
    print(f"Figure C: Gate Crossing Precision -- {n_episodes} episodes")
    print(f"{'='*60}")

    all_crossings = {}
    for name, m in METHODS.items():
        pool = []
        print(f"  Collecting {name} ({n_episodes} eps)...")
        for ep in range(n_episodes):
            seed = 1000 + ep
            _, _, _, crossings, _ = run_episode(
                m['sharp_env'], m['sharp_model'], seed)
            pool.extend(crossings)
        all_crossings[name] = pool
        print(f"    -> {len(pool)} crossing events")

    gate_hw, gate_hh = 0.5, 0.5

    # ── 2x2 subplots ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    method_order = ['Base', 'PBRS', 'Fixed', 'Step']
    titles = ['(a) Base Only', '(b) PBRS',
              '(c) Additive (Fixed)', '(d) Additive (Step)']

    for idx, (mkey, title) in enumerate(zip(method_order, titles)):
        ax = axes[idx // 2][idx % 2]
        c = METHODS[mkey]['color']
        cx = all_crossings[mkey]

        # gate frame
        rect = patches.Rectangle((-gate_hw, -gate_hh), 2*gate_hw, 2*gate_hh,
                                  linewidth=2.5, edgecolor='#2c3e50',
                                  facecolor='#eaf2f8', alpha=0.5, zorder=1)
        ax.add_patch(rect)
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
        ax.axvline(0, color='gray', ls='--', alpha=0.4)

        if len(cx) > 0:
            dy_arr = np.array([pt[1] for pt in cx])
            dz_arr = np.array([pt[2] for pt in cx])
            r_arr = np.sqrt(dy_arr**2 + dz_arr**2)

            # scatter
            ax.scatter(dy_arr, dz_arr, color=c, s=20, alpha=0.4,
                       edgecolors='none', zorder=5)

            # density contours
            try:
                kde = gaussian_kde(np.vstack([dy_arr, dz_arr]))
                yi = np.linspace(-gate_hw - 0.2, gate_hw + 0.2, 100)
                zi = np.linspace(-gate_hh - 0.2, gate_hh + 0.2, 100)
                YI, ZI = np.meshgrid(yi, zi)
                density = kde(np.vstack([YI.ravel(), ZI.ravel()])).reshape(YI.shape)
                ax.contour(YI, ZI, density, levels=4, colors=c,
                           linewidths=1.2, alpha=0.7, zorder=6)
            except Exception:
                pass

            # mean marker
            mean_dy, mean_dz = np.mean(dy_arr), np.mean(dz_arr)
            ax.scatter(mean_dy, mean_dz, color=c, s=200, marker='+',
                       linewidths=3, zorder=8)

            # corrected statistics
            mean_r = np.mean(r_arr)
            median_r = np.median(r_arr)
            p90_r = np.percentile(r_arr, 90)
            inside = np.sum((np.abs(dy_arr) < gate_hw) &
                            (np.abs(dz_arr) < gate_hh))
            inside_rate = inside / len(r_arr) * 100

            ax.text(0.03, 0.97,
                    f'N = {len(cx)} crossings\n'
                    f'mean |d| = {mean_r:.3f} m\n'
                    f'median |d| = {median_r:.3f} m\n'
                    f'P90 |d| = {p90_r:.3f} m\n'
                    f'inside rate = {inside_rate:.1f}%',
                    transform=ax.transAxes, fontsize=8.5,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.85))

        ax.set_xlim(-gate_hw - 0.3, gate_hw + 0.3)
        ax.set_ylim(-gate_hh - 0.3, gate_hh + 0.3)
        ax.set_xlabel('Lateral deviation $d_y$ (m)', fontsize=10)
        ax.set_ylabel('Vertical deviation $d_z$ (m)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, ls=':', alpha=0.3)

    fig.suptitle(
        'Figure C: Gate Crossing Precision -- $d_y$-$d_z$ Distribution\n'
        f'Successful gate crossings accumulated over {n_episodes} test '
        'episodes per method (HCGS scenario)',
        fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, 'fig_C_crossing_scatter.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {path}")

    # ── Figure D: density contour overlay (Fixed vs Step) ──
    fig_d, ax_d = plt.subplots(figsize=(8, 7))
    rect_d = patches.Rectangle((-gate_hw, -gate_hh), 2*gate_hw, 2*gate_hh,
                                linewidth=2.5, edgecolor='#2c3e50',
                                facecolor='#eaf2f8', alpha=0.5, zorder=1)
    ax_d.add_patch(rect_d)
    ax_d.axhline(0, color='gray', ls='--', alpha=0.4)
    ax_d.axvline(0, color='gray', ls='--', alpha=0.4)

    contour_levels_pct = [0.50, 0.75, 0.90]
    for mkey in ['Fixed', 'Step']:
        c_color = METHODS[mkey]['color']
        cx = all_crossings[mkey]
        if len(cx) < 10:
            continue
        dy_arr = np.array([pt[1] for pt in cx])
        dz_arr = np.array([pt[2] for pt in cx])

        try:
            kde = gaussian_kde(np.vstack([dy_arr, dz_arr]))
            yi = np.linspace(-gate_hw - 0.2, gate_hw + 0.2, 150)
            zi = np.linspace(-gate_hh - 0.2, gate_hh + 0.2, 150)
            YI, ZI = np.meshgrid(yi, zi)
            density = kde(np.vstack([YI.ravel(), ZI.ravel()])).reshape(YI.shape)

            # compute contour levels from percentiles
            flat = density.ravel()
            sorted_d = np.sort(flat)[::-1]
            cum = np.cumsum(sorted_d) / np.sum(sorted_d)
            levels = []
            for pct in contour_levels_pct:
                idx_p = np.searchsorted(cum, pct)
                if idx_p < len(sorted_d):
                    levels.append(sorted_d[idx_p])
            levels = sorted(levels)

            cs = ax_d.contour(YI, ZI, density, levels=levels,
                              colors=c_color, linewidths=[1.0, 1.5, 2.0],
                              linestyles=['-', '-', '-'], zorder=6)
            ax_d.contourf(YI, ZI, density, levels=[levels[-1], density.max()],
                          colors=[c_color], alpha=0.15, zorder=3)

            # mean marker
            ax_d.scatter(np.mean(dy_arr), np.mean(dz_arr), color=c_color,
                         s=200, marker='+', linewidths=3, zorder=8,
                         label=f'{METHODS[mkey]["label"]} (mean)')
        except Exception:
            pass

    ax_d.set_xlim(-gate_hw - 0.3, gate_hw + 0.3)
    ax_d.set_ylim(-gate_hh - 0.3, gate_hh + 0.3)
    ax_d.set_xlabel('Lateral deviation $d_y$ (m)', fontsize=12)
    ax_d.set_ylabel('Vertical deviation $d_z$ (m)', fontsize=12)
    ax_d.set_title('Figure D: Crossing Density Contours -- Fixed vs Step\n'
                   '50%, 75%, 90% density contours',
                   fontsize=13, fontweight='bold')
    ax_d.legend(fontsize=11, loc='upper right')
    ax_d.set_aspect('equal')
    ax_d.grid(True, ls=':', alpha=0.3)
    path_d = os.path.join(OUT_DIR, 'fig_D_density_contour.png')
    plt.savefig(path_d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {path_d}")

    # ── Figure E: Radial error CDF ──
    fig_e, ax_e = plt.subplots(figsize=(8, 6))
    for mkey in ['Base', 'PBRS', 'Fixed', 'Step']:
        cx = all_crossings[mkey]
        if len(cx) == 0:
            continue
        dy_arr = np.array([pt[1] for pt in cx])
        dz_arr = np.array([pt[2] for pt in cx])
        r_arr = np.sort(np.sqrt(dy_arr**2 + dz_arr**2))
        cdf = np.arange(1, len(r_arr) + 1) / len(r_arr)
        ax_e.plot(r_arr, cdf, color=METHODS[mkey]['color'],
                  linewidth=2.5, label=METHODS[mkey]['label'],
                  linestyle=METHODS[mkey]['ls'])

    ax_e.axvline(0.5, color='gray', ls=':', alpha=0.5,
                 label='Gate half-width (0.5 m)')
    ax_e.set_xlabel('Radial distance from gate center $|d|$ (m)', fontsize=12)
    ax_e.set_ylabel('Cumulative proportion', fontsize=12)
    ax_e.set_title('Figure E: Radial Error CDF -- All Methods\n'
                   f'{n_episodes} episodes per method (HCGS scenario)',
                   fontsize=13, fontweight='bold')
    ax_e.legend(fontsize=10, loc='lower right')
    ax_e.set_xlim(0, 1.0)
    ax_e.set_ylim(0, 1.05)
    ax_e.grid(True, ls=':', alpha=0.4)
    path_e = os.path.join(OUT_DIR, 'fig_E_radial_cdf.png')
    plt.savefig(path_e, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {path_e}")


# =====================================================================
if __name__ == '__main__':
    figure_A(seed=42)
    figure_B(seed=42)
    figure_C(n_episodes=50)
    print("\n[DONE] All paper figures saved in", OUT_DIR)
