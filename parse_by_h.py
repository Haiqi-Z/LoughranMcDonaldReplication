import os
import re
import glob
import time
import pandas as pd
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from load_harvard_negative import load_harvard_neg_words

# === CONFIGURATION ===
HARVARD_DICT_FILE = os.path.expanduser('~/Desktop/asg1/inquirerbasic.csv')
YEAR = '2024'  # Change this for each year
TARGET_FILES = os.path.expanduser(f'~/Desktop/asg1/data/{YEAR}/**/*.txt')
OUTPUT_FILE = os.path.expanduser(f'~/Desktop/asg1/results/parse2_harvard_negative_{YEAR}.csv')

# === Tokenize and Filter Text ===
def tokenize(text):
    return re.findall(r'\b[A-Z]{2,}\b', text.upper())

# === Core Parsing Logic (Worker Function) ===
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

# === Main Function ===
def main():
    print(f"\n Parsing Harvard negative words for year {YEAR}...")
    files = glob.glob(TARGET_FILES, recursive=True)
    print(f"Found {len(files)} .txt files to parse.")

    neg_words = load_harvard_neg_words(HARVARD_DICT_FILE)
    print(f"Loaded {len(neg_words)} Harvard negative words.")

    args_list = [(file, neg_words) for file in files]
    all_results = []

    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, args_list), total=len(files), desc="Parsing"):
            if result:
                all_results.extend(result)

    print(f"Parsed {len(all_results)} total rows.")

    if all_results:
        df = pd.DataFrame(all_results, columns=['filename', 'CIK', 'word', 'tf_ij', 'a_j'])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Output saved to: {OUTPUT_FILE}")
    else:
        print("No data was extracted. Check dictionary and file content.")

    print(f"Completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
