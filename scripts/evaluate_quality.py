import pandas as pd
import requests
import random
import argparse
import sys
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.personalization.config import settings

# Config
CATALOG_PATH = Path("data/catalog/books_catalog.csv")
NUM_SAMPLES = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--samples", type=int, default=100, help="Number of evaluation queries")
    args = parser.parse_args()
    
    api_url = f"http://{args.host}:{args.port}/personalize/recommend"
    
    print("Loading catalog for ground truth...")
    if not CATALOG_PATH.exists():
        print("Catalog not found!")
        return
        
    df = pd.read_csv(CATALOG_PATH)
    
    # Filter authors with at least 5 books
    author_counts = df['authors'].value_counts()
    valid_authors = author_counts[author_counts >= 5].index.tolist()
    
    print(f"Found {len(valid_authors)} authors with 5+ books.")
    
    hits = 0
    genre_matches = 0
    total_recs = 0
    
    print(f"Running {args.samples} evaluation queries against {api_url}...")
    
    for _ in tqdm(range(args.samples)):
        # 1. Pick a random author
        author = random.choice(valid_authors)
        books = df[df['authors'] == author]
        
        if len(books) < 5:
            continue
            
        # 2. Split: History (3 books) -> Target (1 book)
        sample = books.sample(n=4, replace=False)
        history = sample.iloc[:3]['title'].tolist()
        target_book = sample.iloc[3]
        target_title = target_book['title']
        
        # 3. Call API
        try:
            payload = {"user_history": history, "top_k": 10}
            resp = requests.post(api_url, json=payload)
            
            if resp.status_code != 200:
                continue
                
            recs = resp.json()
            rec_titles = [r['title'] for r in recs]
            
            # Metrics
            if target_title in rec_titles:
                hits += 1
            
            # Author Match
            rec_authors = df[df['title'].isin(rec_titles)]['authors'].tolist()
            if author in rec_authors:
                matches = rec_authors.count(author)
                genre_matches += matches
                
            total_recs += len(recs)
            
        except Exception as e:
            print(f"Connection Error: {e}")
            break
            
    if total_recs > 0:
        print("\n--- Evaluation Results ---")
        print(f"Exact Target Hit Rate @ 10: {hits / args.samples:.2%}")
        print(f"Same Author Relevance: {genre_matches / total_recs:.2%} (Approx)")
    else:
        print("No results obtained. Check API connection.")

if __name__ == "__main__":
    main()