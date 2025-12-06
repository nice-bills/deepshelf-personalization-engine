# Semantic Book Personalization Engine

A high-performance, standalone recommendation service that uses **Semantic Search** to provide personalized book suggestions.

Unlike traditional recommenders that rely on collaborative filtering (which fails without massive user data), this engine uses **Sentence Transformers** to understand the *content* of books (Title + Author + Genre + Description), allowing it to work effectively from Day 1 ("Cold Start").

## Key Features

*   **Semantic Understanding:** Connects "The Haunted School" to "Ghost Beach" based on plot descriptions, not just title keywords.
*   **Hybrid Scoring:** Combines **Semantic Similarity** (85%) with **Book Ratings** (15%) to recommend high-quality matches.
*   **Smart Optimization:** Uses **Product Quantization (IVF-PQ)** to compress the search index by **48x** (146MB -> 3MB) with minimal accuracy loss.
*   **Time-Decay Memory:** Prioritizes a user's *recent* reads over ancient history.
*   **Evaluation:** Achieves **40% Exact Hit Rate @ 10** on held-out author tests.
*   **Standalone API:** Runs as a separate microservice (FastAPI) on Port 8001.

## Architecture

This project uses a **retrieval-based** approach:

1.  **The Brain:** A pre-trained `all-MiniLM-L6-v2` model encodes all book metadata (Title, Author, Genre, Description) into 384-dimensional vectors.
2.  **The Index:** A highly optimized FAISS `IndexIVFPQ` (Inverted File + Product Quantization) index for millisecond retrieval.
3.  **The Engine:**
    *   User history is converted to vectors.
    *   Vectors are aggregated using **Time-Decay Averaging**.
    *   The engine searches the FAISS index for the nearest neighbors.
    *   Results are re-ranked using the book's rating.

## Installation & Setup

### Prerequisites
*   Python 3.10+ (or Docker)
*   `uv` (recommended for fast package management) or `pip`

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd personalise
```

### 2. Setup Environment
```bash
# Using uv (Recommended)
uv venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 3. Data Preparation (Crucial Step)
The system needs the "Brain" (Embeddings) and "Index" to function.

**Option A: Download Pre-computed Artifacts (Fast)**
```bash
# Make sure you are in the root 'personalise' folder
python scripts/download_artifacts.py
```

**Option B: Generate from Scratch (Slow - ~1.5 hours)**
```bash
# 1. Generate Embeddings
python scripts/1b_generate_semantic_data.py

# 2. Optimize Index
python scripts/optimize_index.py
```

## Run the Application

### Option A: Run Locally
```bash
uvicorn src.personalization.api.main:app --reload --port 8001
```
API will be available at `http://localhost:8001`.

### Option B: Run with Docker
The Dockerfile is optimized to cache the model and data layers.

```bash
# 1. Build the image
docker build -t personalise .

# 2. Run the container
docker run -p 8001:8001 personalise
```

## Evaluation & Demo
We have included a synthetic dataset of 10,000 users to validate the model.

**Run the Offline Evaluation:**
This script uses a "Leave-One-Out" strategy to see if the model can predict the next book a user reads.
```bash
python scripts/evaluate_system.py
```

**Visualize User Clusters:**
Generate a 2D t-SNE plot showing how the model groups users by interest (requires `matplotlib` & `seaborn`).
```bash
# First install viz deps
uv pip install matplotlib seaborn

# Run visualization
python scripts/visualize_users.py
```
*Output saved to `docs/user_clusters_tsne.png`*

**Inspect Synthetic Data:**
```bash
python scripts/inspect_data.py
```

## API Usage

#### POST `/personalize/recommend`
Get personalized books based on reading history.
```json
{
  "user_history": ["The Haunted School", "It Came from Beneath the Sink!"],
  "top_k": 5
}
```

#### POST `/search`
Semantic search by plot or vibe.
```json
{
  "query": "detective in space solving crimes",
  "top_k": 5
}
```

## Performance Stats

| Metric | Brute Force (Flat) | Optimized (IVF-PQ) |
| :--- | :--- | :--- |
| **Memory** | ~150 MB | **~3 MB** |
| **Recall @ 10** | 100% | ~95% |
| **Speed** | ~10ms | ~2ms |
| **Hit Rate @ 10** | N/A | **40.0%** |

## Roadmap & Future Improvements
*   **Model Compression (ONNX):** Replace the heavy PyTorch dependency with **ONNX Runtime**. This would reduce the Docker image size from ~3GB to ~500MB and improve CPU inference latency by 2-3x.
*   **Real-Time Learning:** Implement a "Session-Based" Recommender (using RNNs or Transformers) to adapt to user intent within a single session, rather than just long-term history.
*   **A/B Testing Framework:** Add infrastructure to serve different model versions to different user segments to scientifically measure engagement.

## License
MIT
