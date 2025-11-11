import os
from pathlib import Path

# Get the directory where this config.py file is located
BASE_DIR = Path(__file__).resolve().parent

# Use relative paths from the app directory
questions_file = str(BASE_DIR / "questions.json")
prompts_file = str(BASE_DIR / "prompts.json")