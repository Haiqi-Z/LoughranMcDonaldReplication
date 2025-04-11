import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def extract_sp500_ciks(year):
    # Fetch archived Wikipedia page
    url = f"https://web.archive.org/web/{year}1231/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the constituents table
    table = soup.find("table", {"id": "constituents"})
    if not table:
        print(f"No constituents table found for {year}.")
        return None

    # Extract headers to find CIK column index
    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    try:
        cik_col_idx = headers.index("cik")
    except ValueError:
        print(f"No 'CIK' column found for {year}.")
        return None

    # Extract rows
    rows = table.find_all("tr")
    data = []
    for row in rows[1:]:  # Skip header row
        cols = row.find_all("td")
        if not cols:
            continue

        # Extract symbol
        symbol = cols[0].get_text(strip=True).upper().replace(".", "-")
        
        # Extract CIK (from <a> tag if present)
        cik_cell = cols[cik_col_idx]
        cik = ""
        a_tag = cik_cell.find("a")
        if a_tag:
            href = a_tag.get("href", "")
            # Extract CIK from URL (e.g., /cgi-bin/browse-edgar?CIK=0000320193)
            match = re.search(r"cik=(\d+)", href, re.IGNORECASE)
            if match:
                cik = match.group(1).zfill(10)
        else:
            cik = cik_cell.get_text(strip=True).zfill(10)

        data.append({"Symbol": symbol, "CIK": cik})

    # Save to DataFrame
    df = pd.DataFrame(data)
    df.to_csv(f"sp500_ciks_{year}.csv", index=False)
    print(f"Saved {len(df)} CIKs for {year}.")
    return df

# Run for years 2018–2022
years = [2020, 2021, 2022, 2023, 2024]
for year in years:
    extract_sp500_ciks(year)