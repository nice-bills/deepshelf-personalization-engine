import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualizer")

# Paths
DATA_DIR = Path("data")
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic" / "user_sequences.parquet"
CATALOG_PATH = DATA_DIR / "catalog" / "books_catalog.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_cache.npy"
OUTPUT_DIR = Path("docs")
OUTPUT_IMAGE = OUTPUT_DIR / "user_clusters_tsne.png"

def visualize_clusters(sample_size=2000):
    """
    Generates a 2D t-SNE projection of user vectors, colored by Persona.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE
    except ImportError as e:
        logger.error("Missing visualization libraries!")
        logger.error("Please run: uv pip install matplotlib seaborn")
        return

    # 1. Load Resources
    logger.info("Loading Data...")
    if not CATALOG_PATH.exists() or not EMBEDDINGS_PATH.exists():
        logger.error("Missing Data! Run download scripts first.")
        return

    # Load Titles for mapping
    df_catalog = pd.read_csv(CATALOG_PATH)
    titles = df_catalog['title'].tolist()
    title_to_idx = {t.lower().strip(): i for i, t in enumerate(titles)}
    
    # Load Embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    
    # Load Users
    df_users = pd.read_parquet(SYNTHETIC_DATA_PATH)
    
    # Sample
    if len(df_users) > sample_size:
        df_users = df_users.sample(sample_size, random_state=42)
    
    logger.info(f"Processing {len(df_users)} users...")
    
    user_vectors = []
    user_personas = []
    
    # 2. Calculate User Vectors
    valid_users = 0
    for _, row in tqdm(df_users.iterrows(), total=len(df_users)):
        history = row['book_sequence']
        persona = row['persona']
        
        valid_indices = []
        for book in history:
            norm_title = book.lower().strip()
            if norm_title in title_to_idx:
                valid_indices.append(title_to_idx[norm_title])
        
        if not valid_indices:
            continue
            
        # Average Embeddings
        vectors = embeddings[valid_indices]
        user_vec = np.mean(vectors, axis=0)
        
        user_vectors.append(user_vec)
        user_personas.append(persona)
        valid_users += 1
        
    X = np.array(user_vectors)
    
    # 3. t-SNE Reduction
    logger.info("Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter_without_progress=300, init='pca')
    X_embedded = tsne.fit_transform(X)
    
    # 4. Plotting
    logger.info("Generating Plot...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=user_personas,
        palette="viridis",
        alpha=0.7,
        s=60
    )
    
    plt.title(f"Semantic User Clusters (t-SNE Projection of {valid_users} Users)", fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Persona")
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    logger.info(f"✅ Visualization saved to {OUTPUT_IMAGE}")
    print(f"Success! Check {OUTPUT_IMAGE} to see your user clusters.")

if __name__ == "__main__":
    visualize_clusters()
