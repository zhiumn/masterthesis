import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ================================================================
# 全局風格設定 (學術論文風格)
# ================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLOR_GATE = '#1565C0'
COLOR_DRONE = '#E53935'
COLOR_VEC_H = '#8E24AA'
COLOR_VEC_V = '#00897B'
COLOR_PROJ = '#757575'
COLOR_BG = '#FAFAFA'

def draw_drone(ax, x, y, radius=0.15):
    """Draw a simple quadrotor icon."""
    circle = patches.Circle((x, y), radius, facecolor='white', edgecolor=COLOR_DRONE, lw=2, zorder=5)
    ax.add_patch(circle)
    # rotors
    r_rot = radius * 0.5
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        cx = x + dx * radius * 0.8
        cy = y + dy * radius * 0.8
        rotor = patches.Circle((cx, cy), r_rot, facecolor=COLOR_DRONE, alpha=0.6, zorder=4)
        ax.add_patch(rotor)
    ax.plot([x-radius, x+radius], [y-radius, y+radius], color=COLOR_DRONE, lw=1.5, zorder=4)
    ax.plot([x-radius, x+radius], [y+radius, y-radius], color=COLOR_DRONE, lw=1.5, zorder=4)

def plot_top_down_view(ax):
    """(a) Top-Down View (X-Y plane) showing heading, velocity, approach and lateral error."""
    ax.set_facecolor(COLOR_BG)
    
    # Gate representation (on the Y axis, looking down Z)
    gate_width = 2.0
    ax.plot([0, 0], [-gate_width/2, gate_width/2], color=COLOR_GATE, lw=5, solid_capstyle='round', zorder=2)
    ax.plot([0], [0], 'o', color=COLOR_GATE, markersize=8, zorder=3)
    ax.text(0.1, 0.2, 'Gate Center', color=COLOR_GATE, fontsize=11, fontweight='bold')
    
    # Gate normal
    ax.annotate('', xy=(1.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_GATE, lw=2, ls='--'))
    ax.text(1.3, 0.1, 'Gate Normal $\mathbf{n}$', color=COLOR_GATE, fontsize=12, fontweight='bold')
    
    # Drone position
    px, py = -2.5, 1.2
    draw_drone(ax, px, py)
    
    # Lateral distance d_lat
    ax.plot([px, 0], [py, py], '--', color=COLOR_PROJ, lw=1.5)
    ax.plot([0, 0], [0, py], '--', color=COLOR_PROJ, lw=1.5)
    ax.annotate('', xy=(0, py), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PROJ, lw=1.5))
    ax.text(0.1, py/2, '$d_{lat}$', color='black', fontsize=14, fontweight='bold', va='center')
    
    # Distance to gate plane
    ax.annotate('', xy=(0, py), xytext=(px, py),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PROJ, lw=1.5))
    ax.text(px/2, py+0.1, '$p_x$', color='black', fontsize=14, fontweight='bold', ha='center')
    
    # Heading vector (h)
    heading_angle = -np.radians(20)
    hx = np.cos(heading_angle)
    hy = np.sin(heading_angle)
    ax.annotate('', xy=(px + hx, py + hy), xytext=(px, py),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_VEC_H, lw=2.5))
    ax.text(px + hx + 0.1, py + hy + 0.1, 'Heading $\mathbf{h}$', color=COLOR_VEC_H, fontsize=12, fontweight='bold')
    
    # Velocity vector (v)
    vel_angle = -np.radians(40)
    v_mag = 1.8
    vx = v_mag * np.cos(vel_angle)
    vy = v_mag * np.sin(vel_angle)
    ax.annotate('', xy=(px + vx, py + vy), xytext=(px, py),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_VEC_V, lw=2.5))
    ax.text(px + vx - 0.2, py + vy - 0.3, 'Velocity $\mathbf{v}$', color=COLOR_VEC_V, fontsize=12, fontweight='bold')
    
    # Approach velocity (r_approach)
    ax.plot([px, px + vx], [py + vy, py + vy], ':', color=COLOR_VEC_V, lw=1.5)
    ax.plot([px + vx, px + vx], [py, py + vy], ':', color=COLOR_VEC_V, lw=1.5)
    ax.annotate('', xy=(px + vx, py), xytext=(px, py),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_VEC_V, lw=2))
    ax.text(px + vx/2, py - 0.25, '$r_{approach}$', color=COLOR_VEC_V, fontsize=13, fontweight='bold', ha='center')
    
    # Theta error arc
    arc_radius = 0.5
    theta_vals = np.linspace(0, heading_angle, 20)
    arc_x = px + arc_radius * np.cos(theta_vals)
    arc_y = py + arc_radius * np.sin(theta_vals)
    ax.plot(arc_x, arc_y, color=COLOR_VEC_H, lw=1.5)
    ax.text(px + 0.6, py - 0.15, r'$\theta_{err}$', color=COLOR_VEC_H, fontsize=14, fontweight='bold')
    
    # Reference line for Theta err (parallel to normal)
    ax.plot([px, px + 1.0], [py, py], '--', color=COLOR_GATE, lw=1.5, alpha=0.5)

    ax.set_title('(a) Top-Down View (X-Y Plane)', fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.set_xlim(-3.5, 2.0)
    ax.set_ylim(-1.5, 2.0)
    ax.axis('off')

def plot_front_view(ax):
    """(b) Front View (Y-Z plane) showing lateral and vertical error."""
    ax.set_facecolor(COLOR_BG)
    
    # Gate rectangle
    gw, gh = 1.0, 1.0 # full width/height = 2*hw
    hw, hh = gw/2, gh/2
    gate_rect = patches.Rectangle((-hw, -hh), gw, gh, linewidth=4, edgecolor=COLOR_GATE, facecolor='none', zorder=2)
    ax.add_patch(gate_rect)
    ax.plot([0], [0], '+', color=COLOR_GATE, markersize=12, markeredgewidth=2, zorder=3)
    ax.text(0, -hh - 0.2, 'Gate Inner Clearance', color=COLOR_GATE, fontsize=11, fontweight='bold', ha='center')
    
    # Drone projection
    dy, dz = 0.35, 0.25
    draw_drone(ax, dy, dz, radius=0.08)
    
    # Distances
    ax.plot([dy, 0], [dz, dz], '--', color=COLOR_PROJ, lw=1.5)
    ax.plot([0, 0], [0, dz], '--', color=COLOR_PROJ, lw=1.5)
    
    ax.annotate('', xy=(dy, dz), xytext=(0, dz),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PROJ, lw=1.5))
    ax.text(dy/2, dz + 0.05, '$d_{lat}$', color='black', fontsize=14, fontweight='bold', ha='center', va='bottom')
    
    ax.annotate('', xy=(0, dz), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PROJ, lw=1.5))
    ax.text(-0.05, dz/2, '$d_z$', color='black', fontsize=14, fontweight='bold', ha='right', va='center')
    
    # Total distance to center
    ax.plot([0, dy], [0, dz], '-', color='#FF8F00', lw=2)
    ax.text(dy/2 + 0.1, dz/2 - 0.05, r'$\sqrt{d_{lat}^2 + d_z^2}$', color='#FF8F00', fontsize=13, fontweight='bold')
    
    # Boundaries
    ax.plot([-hw, hw], [0, 0], 'k:', alpha=0.3)
    ax.plot([0, 0], [-hh, hh], 'k:', alpha=0.3)

    ax.set_title('(b) Front View (Y-Z Plane Projection)', fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.axis('off')

def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_top_down_view(axes[0])
    plot_front_view(axes[1])
    
    plt.tight_layout()
    
    out_dir = 'trajectory_plots'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gate_crossing_schematic.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved schematic to {out_path}")
    plt.close()

if __name__ == '__main__':
    main()
