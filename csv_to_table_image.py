import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def render_table_from_csv(csv_path):
    print(f"正在處理: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 建立一個新 DataFrame 用來顯示美化的資料
    display_df = pd.DataFrame()
    display_df['Method Name'] = df['Method']
    display_df['Gates Passed'] = df.apply(lambda row: f"{row['Gates_Mean']:.1f} ± {row['Gates_Std']:.1f}", axis=1)
    display_df['Average Jerk'] = df.apply(lambda row: f"{row['Jerk_Mean']:.2f} ± {row['Jerk_Std']:.2f}", axis=1)
    display_df['Gate Distance'] = df.apply(lambda row: f"{row['Dist_Mean']:.2f} ± {row['Dist_Std']:.2f}", axis=1)

    # 隱藏外框與座標軸
    fig_height = len(df) * 0.5 + 1.2
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    ax.axis('tight')

    # 繪製表格
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # 排版風格 (類似頂級期刊的清爽俐落風格)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.25, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#B0BEC5')  # 邊框顏色
        if row == 0:
            # 標題欄設計：深藍底 + 白字粗體
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white')
        else:
            # 奇偶行漸變設計（利於長時間閱讀不眼花）
            if row % 2 == 0:
                cell.set_facecolor('#F8F9F9')
            else:
                cell.set_facecolor('#FFFFFF')

    # 儲存圖片到與 CSV 同一個資料夾內
    output_path = csv_path.replace(".csv", "_image.png")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  --> 成功產出表格圖片: {output_path}")

def main():
    print("=" * 60)
    print("  CSV 轉 高畫質表格圖片 轉換器")
    print("=" * 60)
    
    csv_files = glob.glob(os.path.join("evaluation_results", "**", "Summary_Table.csv"), recursive=True)
    
    if not csv_files:
        print("沒有找到任何 Summary_Table.csv 檔案。請確保有產生過結果！")
        return
        
    for f in csv_files:
        render_table_from_csv(f)
        
    print("\n所有 CSV 均已處理完畢！可直接拉進 Word 或論文中使用。")

if __name__ == '__main__':
    main()
