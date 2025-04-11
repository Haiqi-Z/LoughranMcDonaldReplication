import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===
LM_PARSED_FILE = os.path.expanduser('~/Desktop/asg1/results/parse2_negative_only.csv')
LM_RETURNS_FILE = os.path.expanduser('~/Desktop/asg1/results/returns.csv')

HARVARD_PARSED_FILE = os.path.expanduser('~/Desktop/asg1/results/parse2_harvard_negative_only.csv')
HARVARD_RETURNS_FILE = os.path.expanduser('~/Desktop/asg1/results/returns_h.csv')

# === TF-IDF Weight Calculation Function ===
def calculate_weight(df):
    N = df['filename'].nunique()
    df['df_i'] = df.groupby('word')['filename'].transform('nunique')
    df['tfidf'] = ((1 + np.log(df['tf_ij'])) / (1 + np.log(df['a_j']))) * (np.log(N / df['df_i']))
    return df

# === Load and process LM data ===
lm_parsed = pd.read_csv(LM_PARSED_FILE)
lm_returns = pd.read_csv(LM_RETURNS_FILE)
lm_parsed = calculate_weight(lm_parsed)
lm_doc_weights = lm_parsed.groupby('filename')['tfidf'].sum().reset_index(name='weight')
lm_combined = pd.merge(lm_doc_weights, lm_returns, on='filename')

# === Load and process Harvard data ===
harvard_parsed = pd.read_csv(HARVARD_PARSED_FILE)
harvard_returns = pd.read_csv(HARVARD_RETURNS_FILE)
harvard_parsed = calculate_weight(harvard_parsed)
harvard_doc_weights = harvard_parsed.groupby('filename')['tfidf'].sum().reset_index(name='weight')
harvard_combined = pd.merge(harvard_doc_weights, harvard_returns, on='filename')

# === Assign quintiles ===
lm_combined['quintile'] = pd.qcut(lm_combined['weight'], 5, labels=False) + 1
harvard_combined['quintile'] = pd.qcut(harvard_combined['weight'], 5, labels=False) + 1

# === Compute median excess return per quintile ===
lm_quintiles = lm_combined.groupby('quintile')['excess_return'].median()
harvard_quintiles = harvard_combined.groupby('quintile')['excess_return'].median()

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(lm_quintiles.index, lm_quintiles.values, marker='o', label='LM Negative', linewidth=2)
plt.plot(harvard_quintiles.index, harvard_quintiles.values, marker='s', label='Harvard Negative', linewidth=2)
plt.xlabel('Negativity Quintile')
plt.ylabel('Median 3-Day Excess Return')
plt.title('Figure 1: Excess Return by Negativity Quintile (LM vs Harvard)')
plt.legend()
plt.grid(True)
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig(os.path.expanduser('~/Desktop/asg1/results/figure1_comparison.png'))
plt.show()