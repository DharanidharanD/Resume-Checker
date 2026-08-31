"""
Configuration settings, constants, and paths for Resume Screening & Classification System.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
SAMPLE_JDS_DIR = DATA_DIR / "sample_jds"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

# Ensure essential directories exist
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RESUMES_DIR, SAMPLE_JDS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Pretrained Model Artifact Paths
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METRICS_REPORT_PATH = MODELS_DIR / "evaluation_metrics.json"

# Supported Resume Formats
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Target Industry Categories
CATEGORIES = [
    "Data Science",
    "Machine Learning / AI",
    "Web Development",
    "Software Engineering",
    "DevOps & Cloud",
    "Cyber Security",
    "Database Administration",
    "Mobile App Development",
    "Human Resources (HR)",
    "Finance & Accounting",
    "Product Management",
    "Operations & QA"
]

# Screening Weight Configuration (Weights sum to 1.0)
DEFAULT_SCREENING_WEIGHTS = {
    "skill_match_weight": 0.50,
    "semantic_tfidf_weight": 0.30,
    "experience_match_weight": 0.20,
}

# NLP Text Cleaner Settings
NLP_SETTINGS = {
    "min_token_length": 2,
    "max_token_length": 30,
    "lowercase": True,
    "remove_numbers": False,
    "remove_stopwords": True,
    "lemmatize": True,
}
