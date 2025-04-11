import pandas as pd

def load_harvard_neg_words(path):
    """
    Load Harvard-IV-4 Psychosociological Dictionary words tagged as negative.

    Parameters:
        path (str): Path to the Harvard dictionary CSV file

    Returns:
        Set[str]: Set of uppercase words tagged as negative
    """
    df = pd.read_csv(path)
    if 'Entry' not in df.columns or 'Negativ' not in df.columns:
        raise ValueError("CSV must contain 'Entry' and 'Negativ' columns")

    df['Entry'] = df['Entry'].str.upper()
    negative_words = df[df['Negativ'].notnull()]['Entry'].unique()
    return set(negative_words)

# Example usage (uncomment to test directly):
# harvard_neg = load_harvard_neg_words('~/Desktop/asg1/harvard_dict.csv')
# print(f"Loaded {len(harvard_neg)} Harvard negative words")

