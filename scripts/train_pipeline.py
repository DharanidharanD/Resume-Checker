"""
End-to-End Training and Artifact Generation Pipeline.
Loads or generates data, trains and evaluates candidate classifiers, and saves the best model.
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.models.trainer import ModelTrainer
from scripts.generate_synthetic_data import generate_dataset, generate_sample_resumes_and_jds


def run_pipeline(samples_per_category: int = 80):
    print("=" * 70)
    print("  RESUME SCREENING & CLASSIFICATION ML PIPELINE")
    print("=" * 70)

    dataset_path = PROCESSED_DATA_DIR / "resumes_dataset.csv"
    if not dataset_path.exists():
        print(f"[*] Dataset not found at {dataset_path}. Generating realistic dataset...")
        df = generate_dataset(samples_per_category=samples_per_category)
        generate_sample_resumes_and_jds()
    else:
        print(f"[+] Loading existing dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)

    print(f"[+] Dataset loaded: {len(df)} total resumes across {df['category'].nunique()} categories.")
    print(f"[*] Category distribution:\n{df['category'].value_counts().to_string()}\n")

    texts = df["resume_text"].tolist()
    labels = df["category"].tolist()

    trainer = ModelTrainer(random_state=42)
    results = trainer.train_and_evaluate(texts, labels, test_size=0.20)

    print("\n" + "=" * 70)
    print("  TRAINING & BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Winning Model: {results['best_model_name']}")
    print(f"Weighted F1 Score: {results['best_f1'] * 100:.2f}%")
    print(f"Accuracy: {results['best_metrics']['accuracy'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline(samples_per_category=80)
