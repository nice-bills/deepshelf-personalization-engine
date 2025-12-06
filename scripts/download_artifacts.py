import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

HF_REPO_ID = "nice-bill/book-recommender-artifacts" 
REPO_TYPE = "dataset"

# Local Paths
DATA_DIR = Path("data")
CATALOG_DIR = DATA_DIR / "catalog"
INDEX_DIR = DATA_DIR / "index"

FILES_TO_DOWNLOAD = {
    "books_catalog.csv": CATALOG_DIR,
    "embeddings_cache.npy": DATA_DIR,
    "optimized.index": INDEX_DIR
}

def main():
    print(f"--- Checking artifacts from {HF_REPO_ID} ---")
    
    # Ensure directories exist
    for dir_path in [DATA_DIR, CATALOG_DIR, INDEX_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    for filename, dest_dir in FILES_TO_DOWNLOAD.items():
        dest_path = dest_dir / filename
        
        if dest_path.exists():
            print(f"Found {filename}")
            continue
            
        print(f"Downloading {filename}...")
        try:
            # Download to local cache
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE
            )
            
            # Copy from cache to our project structure
            shutil.copy(cached_path, dest_path)
            print(f"   Saved to {dest_path}")
            
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print("   (Did you create the HF repo and upload the files?)")

    print("\nArtifact setup complete.")

if __name__ == "__main__":
    main()
