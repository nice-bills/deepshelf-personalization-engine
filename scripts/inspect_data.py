import pandas as pd
from pathlib import Path

# Config
DATA_PATH = Path("data/synthetic/user_sequences.parquet")

def inspect():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    try:
        print(f"Reading {DATA_PATH}...")
        df = pd.read_parquet(DATA_PATH)
        
        print("\n--- Schema ---")
        print(df.info())
        
        print("\n--- First 5 Rows ---")
        print(df.head().to_string())
        
        print("\n--- Sample User History ---")
        # Show the full history of the first user
        first_user = df.iloc[0]
        print(f"User ID: {first_user.get('user_id', 'N/A')}")
        print(f"History: {first_user.get('book_history', 'N/A')}")

    except Exception as e:
        print(f"Failed to read parquet: {e}")

if __name__ == "__main__":
    inspect()

