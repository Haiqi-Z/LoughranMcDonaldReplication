import pandas as pd

years = [2020, 2021, 2022, 2023, 2024]
dfs = [pd.read_csv(f'~/Desktop/asg1/results/parse2_negative_only_{y}.csv') for y in years]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('~/Desktop/asg1/results/parse2_negative_only.csv', index=False)
print(f"Combined {len(combined)} rows saved to parse2_negative_only.csv")

