import os
import re
import glob
import time
import pandas as pd
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count

# === CONFIGURATION ===
LM_DICT_FILE = os.path.expanduser('~/Desktop/asg1/LoughranMcDonald_MasterDictionary_2024.csv')
TARGET_FILES = os.path.expanduser('~/Desktop/asg1/data/**/*.txt')
OUTPUT_FILE = os.path.expanduser('~/Desktop/asg1/results/parse_output2.csv')

# === Load LM Words from Dictionary File ===
def load_lm_words(path):
    df = pd.read_csv(path)
    df = df[df['Word'].notnull()]            # Remove NaNs
    df['Word'] = df['Word'].str.upper()
    return set(df['Word'])

# === Tokenize and Filter Text ===
def tokenize(text):
    return re.findall(r'\b[A-Z]{2,}\b', text.upper())

# === Core Parsing Logic (Worker Function) ===
def process_file(args):
    filepath, lm_words = args
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        tokens = tokenize(text)
        filtered = [word for word in tokens if word in lm_words]
        a_j = len(filtered)
        if a_j == 0:
            print("a_j is 0, causing empty return")
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
    print("Starting parse2.py...")
    files = glob.glob(TARGET_FILES, recursive=True)
    print(f"Found {len(files)} .txt files to parse.")

    lm_words = load_lm_words(LM_DICT_FILE)
    print(f"Loaded {len(lm_words)} LM dictionary words.")

    args_list = [(file, lm_words) for file in files]
    all_results = []

    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, args_list), total=len(files), desc="Parsing"):
            if result:
                all_results.extend(result)

    print(f"Parsed {len(all_results)} total word rows.")

    if all_results:
        df = pd.DataFrame(all_results, columns=['filename', 'CIK', 'word', 'tf_ij', 'a_j'])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Output saved to: {OUTPUT_FILE}")
    else:
        print("No data was extracted. Check LM dictionary and file content.")

    print(f"Completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
