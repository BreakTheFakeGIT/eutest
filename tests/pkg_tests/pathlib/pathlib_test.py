
import os
from pathlib import Path
from dotenv import load_dotenv

def get_project_root() -> Path:
    return Path(__file__).parent.parent

path = get_project_root()
print(f'Current path: {path}')

env_path = path / ".env"
load_dotenv(env_path)
BASE_URL = os.environ.get("BASE_URL")
print(f'BASE_URL: {BASE_URL}')













