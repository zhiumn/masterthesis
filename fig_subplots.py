import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 參數設定
d_plane = np.linspace(0, 5, 500) # 距離範圍 0 到 5 公尺

# 2. 定義四種 proximity 函數
prox_exp = np.exp(-0.8 * d_plane)
prox_lin = np.maximum(0, 1 - d_plane / 4.0)
prox_step = np.where(d_plane < 1.5, 1.0, 0.0)
prox_sig = 1.0 / (1.0 + np.exp(3.0 * (d_plane - 1.5)))

# 3. 繪圖設定 (學術風格)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'legend.fontsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# 4. 建立 2x2 的子圖
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

titles = [
    r'(a) Exponential: $\rho_{exp}(d_{plane})$',
    r'(b) Linear: $\rho_{lin}(d_{plane})$',
    r'(c) Sigmoid: $\rho_{sig}(d_{plane})$',
    r'(d) Step: $\rho_{step}(d_{plane})$'
]
functions = [prox_exp, prox_lin, prox_sig, prox_step]

# 顏色與線條樣式設定
c_center = '#1f77b4' # 藍色實線 (W_centering)
c_head = '#ff7f0e'   # 橘色虛線 (W_heading)
c_smooth = '#2ca02c' # 綠色點虛線 (W_smooth)

# 5. 繪製各子圖
for i, ax in enumerate(axs.flat):
    # Plot W_centering (實線)
    ax.plot(d_plane, functions[i], color=c_center, linestyle='-', linewidth=2.5)
    
    # Plot W_heading (虛線)
    ax.plot(d_plane, 1 - functions[i], color=c_head, linestyle='--', linewidth=2.5)
    
    # Plot W_smooth (點虛線)
    ax.plot(d_plane, np.ones_like(d_plane), color=c_smooth, linestyle=':', linewidth=2.5)
    
    # 設定子圖標題
    ax.set_title(titles[i], fontweight='bold', pad=10)
    
    # 每張圖都獨立標示 X Y 軸
    ax.set_xlabel(r'Distance to Gate Plane $d_{plane}$ (m)')
    ax.set_ylabel(r'Normalized Weight')
    
    # 軸與格線設定
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.05, 1.1)

# 6. 加入全局的 Legend (圖片正上方)
custom_lines = [
    Line2D([0], [0], color=c_center, lw=2.5, linestyle='-'),
    Line2D([0], [0], color=c_head, lw=2.5, linestyle='--'),
    Line2D([0], [0], color=c_smooth, lw=2.5, linestyle=':')
]
custom_labels = [
    r'$w_{centering} = \rho(d_{plane})$',
    r'$w_{heading} = 1 - \rho(d_{plane})$',
    r'$w_{smooth} = 1$'
]
# bbox_to_anchor 設定在整張 figure 內部頂部 (y=1.0)
fig.legend(custom_lines, custom_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=3, frameon=False, handlelength=2.5)

# 7. 調整佈局並儲存
plt.tight_layout(rect=[0, 0, 1, 0.90]) # 留出頂部 10% 空間給 Legend
plt.savefig('dynamic_weight_subplots.png', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig('dynamic_weight_subplots.pdf', transparent=True, bbox_inches='tight')
plt.show()
