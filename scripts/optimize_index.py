import numpy as np
import faiss
from pathlib import Path
import time
import sys

# Config
DATA_DIR = Path("data")
EMBEDDINGS_PATH = DATA_DIR / "embeddings_cache.npy"
OUTPUT_PATH = DATA_DIR / "index" / "optimized.index"

def main():
    if not EMBEDDINGS_PATH.exists():
        print("No embeddings found. Run scripts/1b... first.")
        sys.exit(1)
        
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    d = embeddings.shape[1]
    nb = embeddings.shape[0]
    
    print(f"Dataset: {nb} items, {d} dimensions.")
    
    nlist = 100  
    m = 32       
    nbits = 8   
    
    print(f"Training IVF{nlist}, PQ{m} index...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    
    start_t = time.time()
    index.train(embeddings)
    print(f"Training time: {time.time() - start_t:.2f}s")
    
    print("Adding vectors to index...")
    index.add(embeddings)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT_PATH))
    
    print(f"Optimized index saved to {OUTPUT_PATH}")
    print(f"Original Size: {nb * d * 4 / 1024 / 1024:.2f} MB")
    print(f"Optimized Size: {nb * m / 1024 / 1024:.2f} MB (Approx)")

if __name__ == "__main__":
    main()
