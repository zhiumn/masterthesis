"""
三場景示意圖生成腳本 (適用於學術論文)

生成三個場景的俯視圖 (Top-down view):
  (a) Standard Corridor — 溫和彎道，隨機走廊
  (b) Sharp Turn Corridor — 急彎走廊，高轉折角度
  (c) Racing Track — 膠囊形封閉賽道

Usage:
  python plot_scenario_schematics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


# ================================================================
# 全局風格設定 (學術論文風格)
# ================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# 配色方案
COLOR_GATE = '#1565C0'          # 門框 (深藍)
COLOR_DRONE = '#E53935'         # 無人機 (紅色)
COLOR_PATH = '#66BB6A'          # 飛行路徑 (綠色)
COLOR_ANNOTATION = '#5D4037'    # 註解 (棕色)
COLOR_BG = '#FAFAFA'            # 背景色


def draw_gate(ax, x, y, yaw, width=0.5, color=COLOR_GATE, linewidth=3, alpha=1.0):
    """繪製門框 (以短線段表示)。"""
    tx = -np.sin(yaw)
    ty = np.cos(yaw)
    x1 = x + tx * width / 2
    y1 = y + ty * width / 2
    x2 = x - tx * width / 2
    y2 = y - ty * width / 2
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha,
            solid_capstyle='round')


def draw_drone(ax, x, y, yaw, size=0.2, color=COLOR_DRONE):
    """繪製無人機起始位置標記。"""
    ax.plot(x, y, 'o', color=color, markersize=8, zorder=10, markeredgecolor='white',
            markeredgewidth=1.5)
    dx = np.cos(yaw) * size * 2
    dy = np.sin(yaw) * size * 2
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))


def draw_path(ax, positions, color=COLOR_PATH, style='--', alpha=0.5, lw=1.5):
    """繪製飛行路線 (虛線)。"""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(xs, ys, style, color=color, alpha=alpha, linewidth=lw, zorder=1)


def add_dimension_arrow(ax, x1, y1, x2, y2, text, color=COLOR_ANNOTATION,
                        offset_x=0, offset_y=0.4, fontsize=9):
    """加入尺寸標註箭頭。"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    mx, my = (x1 + x2) / 2 + offset_x, (y1 + y2) / 2 + offset_y
    ax.text(mx, my, text, ha='center', va='bottom', fontsize=fontsize,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.9))


# ================================================================
# 場景 A: Standard Corridor (基準走廊)
# ================================================================
def generate_standard_corridor(ax, seed=42):
    rng = np.random.RandomState(seed)

    n_gates = 10
    cur_pos = np.array([0., 0.])
    cur_yaw = 0.0
    positions = [cur_pos.copy()]
    yaws = []

    for i in range(n_gates):
        dist = rng.uniform(2.5, 4.0)
        yaw_shift = rng.uniform(-0.35, 0.35)
        cur_yaw += yaw_shift
        dx = dist * np.cos(cur_yaw)
        dy = dist * np.sin(cur_yaw)
        cur_pos = cur_pos + np.array([dx, dy])
        positions.append(cur_pos.copy())
        yaws.append(cur_yaw)

    draw_path(ax, positions)

    for i, (pos, yaw) in enumerate(zip(positions[1:], yaws)):
        draw_gate(ax, pos[0], pos[1], yaw, width=1.0)
        # 交錯上下以避免重疊
        offset = 0.7 if i % 2 == 0 else -0.9
        ax.text(pos[0], pos[1] + offset, f'$G_{{{i+1}}}$', ha='center',
                fontsize=8, color=COLOR_GATE, fontweight='bold')

    draw_drone(ax, positions[0][0], positions[0][1], yaws[0])

    # 參數標註框
    props = dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#A5D6A7', alpha=0.9)
    param_text = ('$d \sim \mathcal{U}(2.5,\, 4.0)$ m\n'
                  '$\Delta\\psi \sim \mathcal{U}(-0.35,\, 0.35)$ rad\n'
                  '$\\approx \pm 20°$')
    ax.text(0.03, 0.97, param_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.set_title('(a) Standard Corridor', fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':')


# ================================================================
# 場景 B: Sharp Turn Corridor (急彎走廊)
# ================================================================
def generate_sharp_turn(ax, seed=42):
    rng = np.random.RandomState(seed)

    n_gates = 10
    cur_pos = np.array([0., 0.])
    cur_yaw = 0.0
    positions = [cur_pos.copy()]
    yaws = []

    for i in range(n_gates):
        dist = rng.uniform(1.8, 5.0)
        yaw_shift = rng.uniform(-0.6, 0.6)
        cur_yaw += yaw_shift
        dx = dist * np.cos(cur_yaw)
        dy = dist * np.sin(cur_yaw)
        cur_pos = cur_pos + np.array([dx, dy])
        positions.append(cur_pos.copy())
        yaws.append(cur_yaw)

    draw_path(ax, positions)

    for i, (pos, yaw) in enumerate(zip(positions[1:], yaws)):
        draw_gate(ax, pos[0], pos[1], yaw, width=1.0)
        offset = 0.8 if i % 2 == 0 else -1.0
        ax.text(pos[0], pos[1] + offset, f'$G_{{{i+1}}}$', ha='center',
                fontsize=8, color=COLOR_GATE, fontweight='bold')

    draw_drone(ax, positions[0][0], positions[0][1], yaws[0])

    # 標出最大轉折
    max_turn_idx = 0
    max_turn = 0
    for i in range(1, len(yaws)):
        turn = abs(yaws[i] - yaws[i-1])
        if turn > max_turn:
            max_turn = turn
            max_turn_idx = i

    if max_turn_idx > 0:
        p_curr = positions[max_turn_idx + 1]
        ax.annotate(f'Sharp turn ≈ {np.degrees(max_turn):.0f}°',
                    xy=(p_curr[0], p_curr[1]),
                    xytext=(p_curr[0] + 2.0, p_curr[1] + 2.0),
                    fontsize=9, color='#C62828', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5,
                                    connectionstyle='arc3,rad=0.2'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.95))

    # 參數標註框
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor='#FFCC80', alpha=0.9)
    param_text = ('$d \sim \mathcal{U}(1.8,\, 5.0)$ m\n'
                  '$\Delta\\psi \sim \mathcal{U}(-0.6,\, 0.6)$ rad\n'
                  '$\\approx \pm 34°$')
    ax.text(0.03, 0.97, param_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.set_title('(b) Sharp Turn Corridor', fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':')


# ================================================================
# 場景 C: Racing Track (膠囊形封閉賽道)
# ================================================================
def generate_racing_track(ax):
    L = 8.0
    R = 3.0
    n_s = 3
    n_t = 3

    gate_positions = []
    gate_yaws = []
    segment_names = []

    # 北側直道: 向西 (yaw=π)
    for i in range(n_s):
        t = (i + 1) / (n_s + 1)
        x = L/2 - L * t
        y = R
        gate_positions.append([x, y])
        gate_yaws.append(np.pi)
        segment_names.append('N')

    # 西側 U-Turn
    for i in range(n_t):
        theta = np.pi * (i + 1) / (n_t + 1)
        x = -L/2 - R * np.sin(theta)
        y = R * np.cos(theta)
        gate_positions.append([x, y])
        gate_yaws.append(np.pi + theta)
        segment_names.append('W')

    # 南側直道: 向東 (yaw=0)
    for i in range(n_s):
        t = (i + 1) / (n_s + 1)
        x = -L/2 + L * t
        y = -R
        gate_positions.append([x, y])
        gate_yaws.append(0)
        segment_names.append('S')

    # 東側 U-Turn
    for i in range(n_t):
        theta = np.pi * (i + 1) / (n_t + 1)
        x = L/2 + R * np.sin(theta)
        y = -R * np.cos(theta)
        gate_positions.append([x, y])
        gate_yaws.append(theta)
        segment_names.append('E')

    # 膠囊形軌道曲線
    n_smooth = 300
    track_x, track_y = [], []
    for i in range(n_smooth):
        frac = i / n_smooth
        if frac < 0.25:
            t = frac / 0.25
            track_x.append(L/2 - L * t)
            track_y.append(R)
        elif frac < 0.5:
            t = (frac - 0.25) / 0.25
            theta = np.pi * t
            track_x.append(-L/2 - R * np.sin(theta))
            track_y.append(R * np.cos(theta))
        elif frac < 0.75:
            t = (frac - 0.5) / 0.25
            track_x.append(-L/2 + L * t)
            track_y.append(-R)
        else:
            t = (frac - 0.75) / 0.25
            theta = np.pi * t
            track_x.append(L/2 + R * np.sin(theta))
            track_y.append(-R * np.cos(theta))
    track_x.append(track_x[0])
    track_y.append(track_y[0])
    ax.plot(track_x, track_y, '--', color=COLOR_PATH, alpha=0.4, lw=1.5)

    # 門框與標籤
    for i, (pos, yaw) in enumerate(zip(gate_positions, gate_yaws)):
        draw_gate(ax, pos[0], pos[1], yaw, width=1.0)
        # 根據段落選擇標籤偏移
        seg = segment_names[i]
        if seg == 'N':
            ox, oy = 0, 0.8
        elif seg == 'S':
            ox, oy = 0, -1.0
        elif seg == 'W':
            ox, oy = -0.9, 0
        else:
            ox, oy = 0.9, 0

        ax.text(pos[0] + ox, pos[1] + oy, f'$G_{{{i+1}}}$', ha='center', va='center',
                fontsize=7, color=COLOR_GATE, fontweight='bold')

    # 無人機
    start_x = L / 2 + 0.5
    start_y = R
    draw_drone(ax, start_x, start_y, np.pi, size=0.3)
    ax.text(start_x + 0.3, start_y + 0.6, 'Start', fontsize=8, color=COLOR_DRONE,
            fontweight='bold')

    # 飛行方向大箭頭
    arrow_cfg = dict(arrowstyle='->', color='#9E9E9E', lw=2.5, alpha=0.5)
    # 北：向西
    ax.annotate('', xy=(-1, R + 1.6), xytext=(1.5, R + 1.6), arrowprops=arrow_cfg)
    # 南：向東
    ax.annotate('', xy=(1, -R - 1.6), xytext=(-1.5, -R - 1.6), arrowprops=arrow_cfg)
    # 西：向下
    ax.annotate('', xy=(-L/2 - R - 1.3, -0.8), xytext=(-L/2 - R - 1.3, 0.8), arrowprops=arrow_cfg)
    # 東：向上
    ax.annotate('', xy=(L/2 + R + 1.3, 0.8), xytext=(L/2 + R + 1.3, -0.8), arrowprops=arrow_cfg)

    # 段落標籤
    ax.text(0, R + 2.2, 'North Straight', ha='center', fontsize=9,
            color='#424242', style='italic')
    ax.text(0, -R - 2.2, 'South Straight', ha='center', fontsize=9,
            color='#424242', style='italic')
    ax.text(-L/2 - R - 2.5, 0, 'West\nU-Turn', ha='center', fontsize=9,
            color='#424242', style='italic')
    ax.text(L/2 + R + 2.5, 0, 'East\nU-Turn', ha='center', fontsize=9,
            color='#424242', style='italic')

    # 尺寸標註
    add_dimension_arrow(ax, -L/2, -R - 3.5, L/2, -R - 3.5,
                        f'L = {L:.0f} m', offset_y=0.35)
    add_dimension_arrow(ax, L/2 + R + 3.5, -R, L/2 + R + 3.5, R,
                        f'2R = {2*R:.0f} m', offset_x=0.5, offset_y=0)

    # 參數標註框
    props = dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.9)
    param_text = (f'{n_s*2 + n_t*2} gates/lap\n'
                  f'Closed circuit\n'
                  f'L={L:.0f}m, R={R:.0f}m')
    ax.text(0.03, 0.97, param_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.set_title('(c) Racing Track', fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':')


# ================================================================
# 主函數
# ================================================================
def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax in axes:
        ax.set_facecolor(COLOR_BG)

    generate_standard_corridor(axes[0])
    generate_sharp_turn(axes[1])
    generate_racing_track(axes[2])

    # 共用圖例
    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_GATE, linewidth=3.5, label='Gate'),
        plt.Line2D([0], [0], color=COLOR_PATH, linewidth=1.5, linestyle='--',
                    alpha=0.6, label='Flight path'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DRONE,
                    markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                    label='Drone (start)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=11,
               edgecolor='#BDBDBD')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = 'sample_efficiency_results/scenario_schematics.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已儲存示意圖: {out_path}")

    out_pdf = 'sample_efficiency_results/scenario_schematics.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"已儲存 PDF 版本: {out_pdf}")

    plt.close()
    print("完成！")


if __name__ == '__main__':
    main()
