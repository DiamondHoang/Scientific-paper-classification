import os
import warnings
from . import config

warnings.filterwarnings("ignore")


def create_directories():
    """Create necessary directories"""
    directories = [
        config.CACHE_DIR,
        config.FIGURES_DIR,
        config.REPORTS_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)