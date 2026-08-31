"""
TalentMatrix AI Application Launcher & Environment Verifier.
"""
import sys
import os
import subprocess
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import init_db, seed_initial_data_if_empty
from src.config import CLASSIFIER_PATH
from scripts.train_pipeline import run_pipeline


def verify_and_init():
    print("[*] Initializing TalentMatrix AI Database...")
    init_db()
    seed_initial_data_if_empty()

    if not CLASSIFIER_PATH.exists():
        print("[*] Pre-trained classifier models not found. Running training pipeline...")
        run_pipeline(samples_per_category=70)
    else:
        print("[+] Machine learning models and vectorizers verified.")


def main():
    if "--init-only" in sys.argv:
        verify_and_init()
        return

    print("=" * 70)
    print("  TalentMatrix AI(TM) - Enterprise Candidate Intelligence System")
    print("  Final Year Capstone Project Launcher")
    print("=" * 70)

    verify_and_init()

    print("\n[+] Launching Streamlit Web Application at http://localhost:8501...")
    webbrowser.open("http://localhost:8501")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"], cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
