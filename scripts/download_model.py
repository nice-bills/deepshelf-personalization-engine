from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "all-MiniLM-L6-v2"

def download():
    print(f"Downloading {MODEL_NAME}...")
    # This will download to HF_HOME (set in Dockerfile)
    SentenceTransformer(MODEL_NAME)
    print("Done.")

if __name__ == "__main__":
    download()
