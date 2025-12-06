import pandas as pd
import numpy as np
import faiss
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Evaluator")

# Paths
DATA_DIR = Path("data")
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic" / "user_sequences.parquet"
CATALOG_PATH = DATA_DIR / "catalog" / "books_catalog.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_cache.npy"
INDEX_PATH = DATA_DIR / "index" / "optimized.index"

def evaluate_hit_rate(top_k=10, sample_size=1000):
    """
    Evaluates the recommender using a Leave-One-Out strategy.
    metric: Hit Rate @ k
    """
    
    # 1. Load Resources
    logger.info("Loading Catalog and Embeddings...")
    if not CATALOG_PATH.exists() or not EMBEDDINGS_PATH.exists():
        logger.error("Missing Data! Run download scripts first.")
        return

    # Load Titles for mapping
    df_catalog = pd.read_csv(CATALOG_PATH)
    titles = df_catalog['title'].tolist()
    # Create Title -> Index map (normalized)
    title_to_idx = {t.lower().strip(): i for i, t in enumerate(titles)}
    
    # Load Embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    
    # Load Index
    logger.info("Loading FAISS Index...")
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        index.nprobe = 10
    else:
        logger.info("Optimized index not found, building flat index on the fly...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    # 2. Load Synthetic Users
    logger.info(f"Loading Synthetic Data from {SYNTHETIC_DATA_PATH}...")
    df_users = pd.read_parquet(SYNTHETIC_DATA_PATH)
    
    # Sample users if dataset is too large
    if len(df_users) > sample_size:
        df_users = df_users.sample(sample_size, random_state=42)
        
    logger.info(f"Evaluating on {len(df_users)} users...")
    
    hits = 0
    processed_users = 0
    
    for _, row in tqdm(df_users.iterrows(), total=len(df_users)):
        history = row['book_sequence']
        
        # Need at least 2 books (1 for history, 1 for test)
        if len(history) < 2:
            continue
            
        # Leave-One-Out Split
        target_book = history[-1]
        context_books = history[:-1]
        
        # 3. Convert Context to Vector
        valid_indices = []
        for book in context_books:
            norm_title = book.lower().strip()
            if norm_title in title_to_idx:
                valid_indices.append(title_to_idx[norm_title])
                
        if not valid_indices:
            continue
            
        # Get vectors and average (Time Decay Simulation)
        context_vectors = embeddings[valid_indices]
        
        # Simple Time Decay
        n = len(valid_indices)
        decay_factor = 0.9
        weights = np.array([decay_factor ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        
        user_vector = np.average(context_vectors, axis=0, weights=weights).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(user_vector)
        
        # 4. Search
        # We search for top_k + len(context) because the model might return books the user already read
        search_k = top_k + len(valid_indices) + 5 
        scores, indices = index.search(user_vector, search_k)
        
        # Filter results
        recommended_titles = []
        seen_indices = set(valid_indices) # Don't recommend what they just read
        
        for idx in indices[0]:
            if idx in seen_indices:
                continue
            
            rec_title = titles[idx]
            recommended_titles.append(rec_title)
            
            if len(recommended_titles) >= top_k:
                break
                
        # 5. Check Hit
        # We check if the TARGET book title is in the recommended list
        # Using loose matching (substring or exact) can be generous, but strict is better for ML metrics
        # We'll stick to exact string match (normalized)
        
        target_norm = target_book.lower().strip()
        rec_norm = [t.lower().strip() for t in recommended_titles]
        
        if target_norm in rec_norm:
            hits += 1
            
        processed_users += 1

    # 6. Report
    if processed_users == 0:
        print("No valid users found for evaluation.")
        return

    hit_rate = hits / processed_users
    print("\n" + "="*40)
    print(f"EVALUATION REPORT (Sample: {processed_users} users)")
    print("="*40)
    print(f"Metric: Hit Rate @ {top_k}")
    print(f"Score:  {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print("-" * 40)
    print("Interpretation:")
    print(f"In {hit_rate*100:.1f}% of cases, the model successfully predicted")
    print("the exact next book the user would read.")
    print("="*40)

if __name__ == "__main__":
    evaluate_hit_rate()
