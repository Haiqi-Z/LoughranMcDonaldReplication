import os
import re
import glob
import time
import pandas as pd
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count

# === CHANGE THIS TO THE YEAR YOU'RE RUNNING ===
YEAR = '2024'

# === CONFIGURATION ===
LM_DICT_FILE = os.path.expanduser('~/Desktop/asg1/LoughranMcDonald_MasterDictionary_2024.csv')
TARGET_FILES = os.path.expanduser(f'~/Desktop/asg1/data/{YEAR}/**/*.txt')
OUTPUT_FILE = os.path.expanduser(f'~/Desktop/asg1/results/parse2_negative_only_{YEAR}.csv')

# === Load Only Negative Words ===
def load_negative_words(path):
    df = pd.read_csv(path)
    df = df[df['Negative'] > 0]
    df = df[df['Word'].notnull()]
    df['Word'] = df['Word'].str.upper()
    return set(df['Word'])

# === Tokenizer ===
def tokenize(text):
    return re.findall(r'\b[A-Z]{2,}\b', text.upper())

# === Parser ===
def process_file(args):
    filepath, neg_words = args
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        tokens = tokenize(text)
        filtered = [word for word in tokens if word in neg_words]
        a_j = len(filtered)
        if a_j == 0:
            return []

        tf_counts = Counter(filtered)
        filename = os.path.basename(filepath)
        cik = filename.split('_')[0] if '_' in filename else 'unknown'

        return [[filename, cik, word, tf_ij, a_j] for word, tf_ij in tf_counts.items()]

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

# === Main ===
def main():
    print(f"Parsing filings for year: {YEAR}")
    files = glob.glob(TARGET_FILES, recursive=True)
    print(f"Found {len(files)} .txt files in data/{YEAR}/")

    neg_words = load_negative_words(LM_DICT_FILE)
    print(f"Loaded {len(neg_words)} negative words from LM dictionary.")

    args_list = [(file, neg_words) for file in files]
    all_results = []

    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, args_list), total=len(files), desc=f"Parsing {YEAR}"):
            if result:
                all_results.extend(result)

    print(f"Parsed {len(all_results)} word-document rows for {YEAR}.")

    if all_results:
        df = pd.DataFrame(all_results, columns=['filename', 'CIK', 'word', 'tf_ij', 'a_j'])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Output saved to: {OUTPUT_FILE}")
    else:
        print("No data extracted. Check your input files and LM dictionary.")

    print(f"Done in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
