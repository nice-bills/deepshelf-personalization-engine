from contextlib import asynccontextmanager
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from typing import List
import pandas as pd
import faiss
import numpy as np
import time

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "catalog" / "books_catalog.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_cache.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# Global State
state = {
    "titles": [],
    "title_to_idx": {},
    "index": None,
    "embeddings": None,
    "ratings": [],
    "genres": [],
    "model": None, 
    "popular_indices": [] 
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading resources...")
    start_time = time.time()
    
    if not CATALOG_PATH.exists() or not EMBEDDINGS_PATH.exists():
        logger.error("Missing catalog or embeddings! Run scripts/1b... first.")
        # We continue but service might be degraded
    else:
        try:
            df = pd.read_csv(CATALOG_PATH)
            state["titles"] = df['title'].tolist()
            state["genres"] = df['genres'].fillna("").tolist()
            
            raw_ratings = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0)
            max_rating = raw_ratings.max()
            state["ratings"] = (raw_ratings / max_rating).tolist() if max_rating > 0 else [0.5] * len(df)
            
            # Use normalized keys for robust lookup
            state["title_to_idx"] = {t.lower().strip(): i for i, t in enumerate(state["titles"])}
            
            state["popular_indices"] = np.argsort(raw_ratings)[::-1][:50].tolist()
            
            logger.info("Loading embeddings...")
            embeddings = np.load(EMBEDDINGS_PATH)
            state["embeddings"] = embeddings
            
            OPTIMIZED_INDEX_PATH = DATA_DIR / "index" / "optimized.index"
            
            if OPTIMIZED_INDEX_PATH.exists():
                logger.info("Loading OPTIMIZED FAISS index (IVF-PQ)...")
                state["index"] = faiss.read_index(str(OPTIMIZED_INDEX_PATH))
                state["index"].nprobe = 10 
            else:
                logger.info("Building Standard FAISS index (Flat)...")
                d = embeddings.shape[1]
                index = faiss.IndexFlatIP(d)
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                state["index"] = index
            
            logger.info(f"Loading Semantic Model ({MODEL_NAME})...")
            state["model"] = SentenceTransformer(MODEL_NAME)
            
            logger.info(f"Ready! Loaded {len(state['titles'])} books in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            # Consider raising if critical
    
    yield
    
    logger.info("Shutting down...")
    # Clean up resources if needed

app = FastAPI(title="Semantic Book Discovery Engine", lifespan=lifespan)

# Add Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

class RecommendationRequest(BaseModel):
    user_history: List[str]
    top_k: int = 10

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class BookResponse(BaseModel):
    title: str
    score: float
    genres: str

@app.post("/search", response_model=List[BookResponse])
async def search(request: SearchRequest):
    if state["model"] is None or state["index"] is None:
        raise HTTPException(status_code=503, detail="Service loading...")
        
    query_vector = state["model"].encode([request.query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    scores, indices = state["index"].search(query_vector, request.top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append(BookResponse(
            title=state["titles"][idx],
            score=float(score),
            genres=str(state["genres"][idx])
        ))
        
    return results

@app.post("/personalize/recommend", response_model=List[BookResponse])
async def recommend(request: RecommendationRequest):
    if state["index"] is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    valid_indices = []
    for title in request.user_history:
        normalized_title = title.lower().strip()
        if normalized_title in state["title_to_idx"]:
            valid_indices.append(state["title_to_idx"][normalized_title])
            
    if not valid_indices:
        logger.info("Cold start user: returning popular books")
        results = []
        for idx in state["popular_indices"][:request.top_k]:
             results.append(BookResponse(
                title=state["titles"][idx],
                score=state["ratings"][idx],
                genres=str(state["genres"][idx])
            ))
        return results
    
    history_vectors = state["embeddings"][valid_indices]
    
    n = len(valid_indices)
    decay_factor = 0.9
    weights = np.array([decay_factor ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()
    
    user_vector = np.average(history_vectors, axis=0, weights=weights).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(user_vector)
    
    search_k = (request.top_k * 3) + len(valid_indices)
    scores, indices = state["index"].search(user_vector, search_k)
    
    results = []
    seen_indices = set(valid_indices)
    seen_titles = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx in seen_indices: continue
        title = state["titles"][idx]
        if title in seen_titles: continue
        seen_titles.add(title)
        
        final_score = float(score) + (state["ratings"][idx] * 0.1)
            
        results.append(BookResponse(
            title=title,
            score=final_score,
            genres=str(state["genres"][idx])
        ))
        
        if len(results) >= request.top_k:
            break
    
    results.sort(key=lambda x: x.score, reverse=True)
            
    return results