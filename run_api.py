import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.main import app
from src.config.settings import settings
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)