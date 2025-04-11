import pandas as pd
import glob
import os

files = glob.glob(os.path.expanduser('~/Desktop/asg1/results/parse2_harvard_negative_*.csv'))
print(f"Found {len(files)} files:")
for f in files:
    print(" -", os.path.basename(f))

if not files:
    raise ValueError("No files matched the pattern!")

df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_all.to_csv(os.path.expanduser('~/Desktop/asg1/results/parse2_harvard_negative_only.csv'), index=False)

print("Combined files saved as parse2_harvard_negative_only.csv")

