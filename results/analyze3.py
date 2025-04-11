import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
PARSED_FILE = os.path.expanduser('~/Desktop/asg1/results/parse2_negative_only.csv')
RETURNS_FILE = os.path.expanduser('~/Desktop/asg1/results/returns.csv')

# === STEP 1: Load Parsed Negative Word Data ===
print("Loading parsed data...")
df = pd.read_csv(PARSED_FILE)

# Extract filing date and zero-pad CIK (if not already)
df['filing_date'] = df['filename'].str.extract(r'(\d{8})')[0]
df['filing_date'] = pd.to_datetime(df['filing_date'], format='%Y%m%d', errors='coerce')

# === STEP 2: Compute Document Frequency (df_i for each word) ===
print("Calculating document frequency per word...")
doc_freq = df.groupby('word')['filename'].nunique()
N = df['filename'].nunique()
df_i = doc_freq.to_dict()

# === STEP 3: Compute Weighted Score (w_ij) ===
print("Computing w_ij (TF-IDF-based weights)...")
def compute_weight(row):
    tf_ij = row['tf_ij']
    a_j = row['a_j']
    word = row['word']
    df_i_value = df_i.get(word, 1)  # avoid div-by-zero
    if tf_ij >= 1 and a_j > 0:
        return ((1 + np.log(tf_ij)) / (1 + np.log(a_j))) * np.log(N / df_i_value)
    return 0

df['w_ij'] = df.apply(compute_weight, axis=1)

# === STEP 4: Aggregate Weight per Document ===
print("Aggregating scores per document...")
doc_score = df.groupby('filename')['w_ij'].sum().reset_index()
doc_score.rename(columns={'w_ij': 'neg_score'}, inplace=True)

# === STEP 5: Merge with Excess Returns ===
print("Merging with returns data...")
returns = pd.read_csv(RETURNS_FILE)
merged = doc_score.merge(returns, on='filename', how='inner')

# === STEP 6: Create Quintiles Based on Negativity Score ===
print("Creating quintiles...")
merged['neg_quintile'] = pd.qcut(merged['neg_score'], 5, labels=False)

# === STEP 7: Calculate Median Return by Quintile ===
plot_df = merged.groupby('neg_quintile')['excess_return'].median().reset_index()
plot_df['neg_quintile'] = plot_df['neg_quintile'].astype(int) + 1  # 1-based

# === STEP 8: Plot ===
plt.figure(figsize=(8, 5))
plt.plot(plot_df['neg_quintile'], plot_df['excess_return'], marker='o', linestyle='-')
plt.xlabel('Negativity Quintile')
plt.ylabel('Median 3-Day Excess Return')
plt.title('Figure 1: Excess Return by Negativity Quintile')
plt.xticks(plot_df['neg_quintile'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plot_path = os.path.expanduser('~/Desktop/asg1/results/figure1_neg_quintiles.png')
plt.savefig(plot_path)
plt.show()

print(f"Plot saved to: {plot_path}")
