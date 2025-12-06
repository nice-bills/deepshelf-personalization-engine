import requests
import argparse
import sys
from pathlib import Path

# Add src to path to import config
sys.path.append(str(Path(__file__).parent))
from src.personalization.config import settings

def main():
    parser = argparse.ArgumentParser(description="Test the Recommendation API")
    parser.add_argument("--host", type=str, default=settings.HOST, help="API Host")
    parser.add_argument("--port", type=int, default=settings.PORT, help="API Port")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/personalize/recommend"

    payload = {
        "user_history": [
            "The Haunted School", 
            "It Came from Beneath the Sink!"
            "Legion"
        ],
        "top_k": 5
    }

    print(f"Sending request to {url}...")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            results = response.json()
            print("\u2714 Recommendations:")
            for i, book in enumerate(results, 1):
                print(f"{i}. {book['title']} (Score: {book['score']:.4f})")
        else:
            print(f"\u2714 Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"\u2714 Failed to connect: {e}")
        print("Make sure the uvicorn server is running on port 8001!")

if __name__ == "__main__":
    main()