import os
import glob
import pandas as pd
import numpy as np

def holm_bonferroni(p_values):
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    corrected = np.zeros(n)
    cummax = 0.0
    for rank, idx in enumerate(sorted_idx):
        adjusted = p_values[idx] * (n - rank)
        cummax = max(cummax, adjusted)
        corrected[idx] = min(cummax, 1.0)
    return corrected

def recompute_directory(eval_dir):
    csv_files = glob.glob(os.path.join(eval_dir, "pairwise_stats_*.csv"))
    
    family_A = [
        ('Base Only', 'Additive-Fixed'),
        ('Base Only', 'sharp_ablation_A7-S'),
        ('Base Only', 'sharp_ablation_A7_S'),
        ('Base Only', 'Additive-Step'),
        ('PBRS (shared)', 'Additive-Fixed'),
        ('PBRS (shared)', 'sharp_ablation_A7-S'),
        ('PBRS (shared)', 'sharp_ablation_A7_S'),
        ('PBRS (shared)', 'Additive-Step'),
        ('Additive-Fixed', 'Additive-Step'),
        ('sharp_ablation_A7-S', 'Additive-Step'),
        ('sharp_ablation_A7_S', 'Additive-Step'),
        ('Additive-Step', 'Additive-Fixed')
    ]
    
    family_B = [
        ('Base Only', '+Centering'),
        ('Base Only', 'Centering'),
        ('Base Only', 'add Centering'),
        ('Base Only', '+Heading'),
        ('Base Only', 'Heading'),
        ('Base Only', 'add Heading'),
        ('Base Only', '+Smooth'),
        ('Base Only', 'Smooth'),
        ('Base Only', 'add Smooth')
    ]
    
    def is_in_family(m1, m2, fam_list):
        for (a, b) in fam_list:
            if (m1 == a and m2 == b) or (m1 == b and m2 == a):
                return True
        return False

    for csv_file in csv_files:
        if '_planned.csv' in csv_file:
            continue
        df = pd.read_csv(csv_file)
        
        if 'Family' in df.columns:
            continue # already updated
            
        new_rows = []
        
        # Family A
        fam_a_rows = []
        for _, row in df.iterrows():
            if is_in_family(row['Method_A'], row['Method_B'], family_A):
                fam_a_rows.append(row.copy())
                
        if fam_a_rows:
            p_raws = [float(r['p_raw']) for r in fam_a_rows]
            pvals_corrected = holm_bonferroni(p_raws)
            for r, p_c in zip(fam_a_rows, pvals_corrected):
                r['Family'] = 'Family A'
                r['p_holm'] = p_c
                r['Significance'] = '**' if p_c < 0.01 else ('*' if p_c < 0.05 else 'n.s.')
                new_rows.append(r)
                
        # Family B
        fam_b_rows = []
        for _, row in df.iterrows():
            if is_in_family(row['Method_A'], row['Method_B'], family_B):
                fam_b_rows.append(row.copy())
                
        if fam_b_rows:
            p_raws = [float(r['p_raw']) for r in fam_b_rows]
            pvals_corrected = holm_bonferroni(p_raws)
            for r, p_c in zip(fam_b_rows, pvals_corrected):
                r['Family'] = 'Family B'
                r['p_holm'] = p_c
                r['Significance'] = '**' if p_c < 0.01 else ('*' if p_c < 0.05 else 'n.s.')
                new_rows.append(r)
                
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # Reorder columns
            cols = ['Family', 'Method_A', 'Method_B', 'N_A', 'N_B', 'Delta', 'CI_95_Low', 'CI_95_High', 'Hedges_g', 'p_raw', 'p_holm', 'Significance']
            new_df = new_df[[c for c in cols if c in new_df.columns]]
            
            out_file = csv_file.replace('.csv', '_planned.csv')
            new_df.to_csv(out_file, index=False)
            print(f"Updated: {out_file}")

def main():
    base_dir = "evaluation_results"
    if not os.path.exists(base_dir):
        return
        
    for d in os.listdir(base_dir):
        full_d = os.path.join(base_dir, d)
        if os.path.isdir(full_d):
            recompute_directory(full_d)

if __name__ == "__main__":
    main()
