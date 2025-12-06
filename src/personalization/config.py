import os

class Settings:
    # Default to 8001, but allow Env Override
    HOST = os.getenv("API_HOST", "localhost")
    PORT = int(os.getenv("API_PORT", 8001))
    
    @property
    def BASE_URL(self):
        return f"http://{self.HOST}:{self.PORT}"

settings = Settings()
