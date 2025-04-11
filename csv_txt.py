import pandas as pd

# Read the CSV file
df = pd.read_csv("sp500_ciks_2024.csv")

# Extract CIK column, pad to 10 digits, and save to TXT
df["CIK"].astype(str).str.zfill(10).to_csv(
    "sp500_2024ciks.txt", 
    index=False, 
    header=False
)