import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

scenes = ['sharp_turn', 'racing_track']
widths = [0.8, 0.6, 0.5]
filters = ['sharp_ablation_P5', 'sharp_ablation_A7-S', 'racing_ablation_P5', 'racing_track_static']

# Map environment filters
scene_filters = {
    'sharp_turn': ['sharp_ablation_P5', 'sharp_ablation_A7-S'],
    'racing_track': ['racing_ablation_P5', 'racing_track_static']
}

summary_data = []

for scene in scenes:
    for width in widths:
        print(f"\n=======================================================")
        print(f"Running evaluation for {scene} with gate width {width}m")
        print(f"=======================================================")
        
        cmd = [
            r"C:\Users\miat\miniconda3\envs\drone_rl\python.exe", "eval_width_param.py",
            "--scene", scene,
            "--gate_width", str(width),
            "--compare_all",
            "--filter"
        ] + scene_filters[scene]
        
        # Run the evaluation
        subprocess.run(cmd, check=True)
        
        # Find the summary CSV
        # Folder is evaluation_results/{scene}_W{width}_...
        search_dir = "evaluation_results"
        target_folder = ""
        for d in os.listdir(search_dir):
            if d.startswith(f"{scene}_W{width}_"):
                target_folder = os.path.join(search_dir, d)
                break
        
        if target_folder:
            summary_csv = os.path.join(target_folder, "Summary_Table.csv")
            if os.path.exists(summary_csv):
                df = pd.read_csv(summary_csv)
                for index, row in df.iterrows():
                    summary_data.append({
                        'Scene': scene,
                        'Gate_Width': width,
                        'Method': row['Method'],
                        'Gates_Mean': row['Gates_Mean'],
                        'Gates_SEM': row['Gates_SEM']
                    })

# Save aggregated summary
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("evaluation_results/robustness_width_summary.csv", index=False)
print("\nAggregated summary saved to evaluation_results/robustness_width_summary.csv")

# Plotting
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    pass

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Zero-Shot Transfer Robustness (Narrowing Gate Widths)', fontsize=16, fontweight='bold', y=1.05)

colors = {'Additive-Step': '#9B59B6', 'Additive-Fixed': '#4A90E2'}

for i, scene in enumerate(scenes):
    ax = axes[i]
    scene_data = df_summary[df_summary['Scene'] == scene]
    
    methods = scene_data['Method'].unique()
    for method in methods:
        method_data = scene_data[scene_data['Method'] == method]
        method_data = method_data.sort_values(by='Gate_Width', ascending=False)
        
        color = colors.get(method, 'gray')
        ax.errorbar(method_data['Gate_Width'], method_data['Gates_Mean'], yerr=method_data['Gates_SEM'], 
                    label=method, color=color, marker='o', linewidth=2.5, markersize=8, capsize=5)

    ax.set_title(scene.replace('_', ' ').title(), fontsize=14)
    ax.set_xlabel('Gate Width (m)', fontsize=12)
    ax.set_ylabel('Gates Passed', fontsize=12)
    ax.invert_xaxis()  # 1.0 -> 0.8 -> 0.6 -> 0.5 (Width decreasing)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

plt.tight_layout()
plt.savefig("evaluation_results/robustness_width_plot.png", dpi=300, bbox_inches='tight')
print("Plot saved to evaluation_results/robustness_width_plot.png")
