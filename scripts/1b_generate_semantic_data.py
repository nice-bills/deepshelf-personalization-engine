import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import torch
from sentence_transformers import SentenceTransformer
import random
import faiss

NUM_USERS = 10000
MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 50
DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "catalog" / "books_catalog.csv"
OUTPUT_DIR = DATA_DIR / "synthetic"
MODEL_NAME = "all-MiniLM-L6-v2" 

def main():
    print("Loading catalog...")
    df = pd.read_csv(CATALOG_PATH)
    
    df['rich_content'] = (
        "Title: " + df['title'].fillna("") + 
        "; Author: " + df['authors'].fillna("Unknown") +
        "; Genres: " + df['genres'].fillna("") + 
        "; Description: " + df['description'].fillna("").astype(str).str.slice(0, 300)
    )
    
    titles = df['title'].tolist()
    content_to_encode = df['rich_content'].tolist()
    
    EMBEDDINGS_CACHE = DATA_DIR / "embeddings_cache.npy"
        
    if EMBEDDINGS_CACHE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        emb_np = np.load(EMBEDDINGS_CACHE)
        print("Embeddings loaded.")
    else:
        print(f"Loading Teacher Model ({MODEL_NAME})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        
        print("Encoding books (Title + Author + Genre + Desc)...")
        embeddings = model.encode(content_to_encode, show_progress_bar=True, convert_to_tensor=True)
        emb_np = embeddings.cpu().numpy()
        
        print(f"Saving embeddings to {EMBEDDINGS_CACHE}...")
        np.save(EMBEDDINGS_CACHE, emb_np)
    
    print(f"Generating {NUM_USERS} semantic user journeys...")
    
    cpu_index = faiss.IndexFlatIP(emb_np.shape[1])
    faiss.normalize_L2(emb_np)
    cpu_index.add(emb_np)
    
    users = []
    
    for user_id in tqdm(range(NUM_USERS)):
        sequence = []
        
        num_interests = random.choice([1, 1, 2, 3])
        
        for _ in range(num_interests):
            anchor_idx = random.randint(0, len(titles) - 1)
            
            k_neighbors = 50
            q = emb_np[anchor_idx].reshape(1, -1)
            _, indices = cpu_index.search(q, k_neighbors)
            neighbors_indices = indices[0]
            
            num_to_read = random.randint(5, 15)
            
            read_indices = np.random.choice(neighbors_indices, size=min(len(neighbors_indices), num_to_read), replace=False)
            
            for idx in read_indices:
                sequence.append(titles[idx])
                
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
            
        if len(sequence) >= MIN_SEQUENCE_LENGTH:
             users.append({
                'user_id': user_id,
                'book_sequence': sequence,
                'sequence_length': len(sequence),
                'persona': 'semantic_explorer',
                'metadata': {'generated': True} 
            })
            
    users_df = pd.DataFrame(users)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "user_sequences.parquet"
    users_df.to_parquet(output_path, index=False)
    
    stats = {
        'num_users': len(users_df),
        'avg_sequence_length': float(users_df['sequence_length'].mean()),
        'generated_via': "semantic_clustering"
    }
    
    with open(OUTPUT_DIR / "user_metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\n Generated {len(users_df)} semantic users")
    print(f"   Output: {output_path}")

if __name__ == "__main__":
    main()