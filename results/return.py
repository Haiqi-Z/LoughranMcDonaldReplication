import pandas as pd
import numpy as np
import glob
import wrds
import os
from datetime import timedelta

# === CONFIGURATION ===
PARSED_FILE = '~/Desktop/asg1/results/parse2_harvard_negative_only.csv' # comment this and uncomment next line if using LM Dictionary
#PARSED_FILE = os.path.expanduser('~/Desktop/asg1/results/parse2_negative_only.csv')
SP500_FILES = os.path.expanduser('~/Desktop/asg1/sp500_ciks_*.csv')
#OUTPUT_FILE = os.path.expanduser('~/Desktop/asg1/results/returns.csv')
OUTPUT_FILE = os.path.expanduser('~/Desktop/asg1/results/returns_h.csv')

# === STEP 1: Load Parsed File and Extract Filing Info ===
print("Loading parsed negative word data...")
parsed_df = pd.read_csv(PARSED_FILE)

filings = parsed_df[['filename']].drop_duplicates()

# Extract filing date from filename
filings['filing_date'] = filings['filename'].str.extract(r'(\d{8})')[0]
filings['filing_date'] = pd.to_datetime(filings['filing_date'], format='%Y%m%d', errors='coerce')

# Extract CIK using regex from filename
filings['CIK'] = filings['filename'].str.extract(r'edgar_data_(\d+)_')[0]
filings['CIK'] = filings['CIK'].astype(str).str.zfill(10)

filings['year'] = filings['filing_date'].dt.year


print(f"Total unique filings found: {len(filings)}")

# === STEP 2: Load S&P500 Constituents by Year ===
print("Loading S&P500 constituents per year...")
sp500_yearly = []

files = glob.glob(SP500_FILES)
print(f"Found {len(files)} S&P500 constituent files")

for file in files:
    basename = os.path.basename(file)
    try:
        year = int(basename.split('_')[2].split('.')[0])  # expects sp500_ciks_2020.csv
    except ValueError:
        print(f"Skipping unexpected file format: {basename}")
        continue

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()  # normalize columns
    df['cik'] = df['cik'].astype(str).str.zfill(10)
    df['ticker'] = df['symbol'].str.upper()
    df = df.drop(columns='symbol')
    df['year'] = year
    sp500_yearly.append(df)

sp500_df = pd.concat(sp500_yearly, ignore_index=True)

# === STEP 3: Merge Filings with Tickers ===
merged = filings.merge(sp500_df, left_on=['CIK', 'year'], right_on=['cik', 'year'], how='left')
print(f"Ticker-matched filings: {merged['ticker'].notnull().sum()} / {len(merged)}")
print("Sample unmatched filings:", merged[merged['ticker'].isnull()][['CIK', 'year']].head())

# === STEP 4: Connect to WRDS and Map Ticker → PERMNO ===
print("Connecting to WRDS...")
conn = wrds.Connection()

perm_map = conn.raw_sql("""
    SELECT permno, ticker, namedt, nameendt
    FROM crsp.msenames
    WHERE nameendt >= '2018-01-01' AND namedt <= '2025-12-31'
""")
perm_map['ticker'] = perm_map['ticker'].str.upper()

merged = merged.merge(perm_map, on='ticker', how='left')
merged = merged[
    (merged['filing_date'] >= merged['namedt']) &
    (merged['filing_date'] <= merged['nameendt'])
]
print(f"PERMNO-matched filings: {merged['permno'].notnull().sum()} / {len(merged)}")
print("Sample unmatched tickers:", merged[merged['permno'].isnull()]['ticker'].dropna().unique()[:10])

merged = merged.drop_duplicates('filename')

# === STEP 5: Pull CRSP Returns ===
permnos = merged['permno'].dropna().unique().tolist()
permno_list = ','.join(map(str, permnos))

print(f"Pulling CRSP returns for {len(permnos)} PERMNOs...")

returns = conn.raw_sql(f"""
    SELECT date, permno, retx
    FROM crsp.dsf
    WHERE date BETWEEN '2018-01-01' AND '2025-01-01'
      AND permno IN ({permno_list})
""")
returns['date'] = pd.to_datetime(returns['date'])

# === STEP 6: Compute 3-Day Post-Filing Excess Returns ===
def get_cum_return(row):
    p = row['permno']
    d = row['filing_date']
    df = returns[(returns['permno'] == p) & (returns['date'] > d) & (returns['date'] <= d + timedelta(days=4))]
    if df.empty:
        return np.nan
    return (df['retx'] + 1).prod() - 1

print("Calculating 3-day post-filing excess returns...")
merged['excess_return'] = merged.apply(get_cum_return, axis=1)

# === STEP 7: Save Output ===
final_df = merged[['filename', 'excess_return']].dropna()
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(final_df)} excess returns to: {OUTPUT_FILE}")
