import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 參數設定
d_plane = np.linspace(0, 5, 500) # 距離範圍 0 到 5 公尺

# 2. 定義四種 proximity 函數 (根據論文公式 72-75)
prox_exp = np.exp(-0.8 * d_plane)
prox_lin = np.maximum(0, 1 - d_plane / 4.0)
prox_step = np.where(d_plane < 1.5, 1.0, 0.0)
prox_sig = 1.0 / (1.0 + np.exp(3.0 * (d_plane - 1.5)))

# 3. 繪圖設定 (學術風格)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

fig, ax = plt.subplots(figsize=(8, 5.5))

# 4. 繪製四條曲線 (使用不同線條樣式與顏色以利黑白列印辨識)
ax.plot(d_plane, prox_exp, label=r'$\rho_{exp} (d_{plane})$', color='#1f77b4', linestyle='-', linewidth=2.5)
ax.plot(d_plane, prox_lin, label=r'$\rho_{lin} (d_{plane})$', color='#2ca02c', linestyle='-', linewidth=2.5)
ax.plot(d_plane, prox_sig, label=r'$\rho_{sig} (d_{plane})$', color='#ff7f0e', linestyle='-', linewidth=2.5)
ax.plot(d_plane, prox_step, label=r'$\rho_{step} (d_{plane})$', color='#d62728', linestyle='-', linewidth=2.5)

# 4.5 繪製對稱曲線 (w_heading = 1 - proximity) 與 W_smooth
ax.plot(d_plane, 1 - prox_exp, color='#1f77b4', linestyle='--', linewidth=2.5, alpha=0.4)
ax.plot(d_plane, 1 - prox_lin, color='#2ca02c', linestyle='--', linewidth=2.5, alpha=0.4)
ax.plot(d_plane, 1 - prox_sig, color='#ff7f0e', linestyle='--', linewidth=2.5, alpha=0.4)
ax.plot(d_plane, 1 - prox_step, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.4)

ax.plot(d_plane, np.ones_like(d_plane), color='gray', linestyle=':', linewidth=2.5)

# 5. 標註切換中心 (d_plane = 1.5)
#ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
#ax.text(1.55, 0.85, 'Switching Center\n($d_{plane} = 1.5$)', color='gray', fontsize=11, verticalalignment='center')

# 7. 軸與格線設定
ax.set_xlabel(r'Distance to Gate Plane $d_{plane}$ (m)')
ax.set_ylabel(r'Normalized Weight')
#ax.set_title(r'Dynamic Weight Scheduling', pad=15)
ax.set_xlim(0, 5)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, linestyle='--', alpha=0.6)
leg1 = ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.82), frameon=True, edgecolor='lightgray', framealpha=0.9)
ax.add_artist(leg1)

# 6. 標註權重分配邏輯 (Weight Allocation Logic)
custom_lines = [
    Line2D([0], [0], color='dimgray', lw=2, linestyle='-'),
    Line2D([0], [0], color='dimgray', lw=2, linestyle='--'),
    Line2D([0], [0], color='gray', lw=2, linestyle=':')
]
custom_labels = [
    r'$w_{centering} = \rho(d_{plane})$',
    r'$w_{heading} = 1 - \rho(d_{plane})$',
    r'$w_{smooth} = 1.0$'
]
ax.legend(custom_lines, custom_labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
          ncol=3, frameon=False, framealpha=0.9, handlelength=2.5,
         )

# 8. 調整佈局並儲存為高解析度圖片
plt.tight_layout()
plt.savefig('dynamic_weight_scheduling.png', dpi=300, bbox_inches='tight')
plt.savefig('dynamic_weight_scheduling.pdf', bbox_inches='tight') # 建議論文使用 PDF 向量圖
plt.show()