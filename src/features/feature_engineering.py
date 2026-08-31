"""
Feature Engineering Pipeline: TF-IDF Representations, Engineered Metrics, and Label Encodings.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib

from src.preprocessing.text_cleaner import TextCleaner
from src.extractors.skill_extractor import SkillExtractor
from src.extractors.experience_extractor import ExperienceExtractor


class FeaturePipeline:
    """
    Constructs and transforms text into multidimensional feature vectors
    suitable for Machine Learning classifiers.
    """

    def __init__(
        self,
        max_features: int = 4000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True
    ):
        self.text_cleaner = TextCleaner()
        self.skill_extractor = SkillExtractor()
        self.experience_extractor = ExperienceExtractor()

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english"
        )
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_text_statistics(self, raw_text: str) -> Dict[str, float]:
        """
        Computes numerical linguistic and structural features from text.
        """
        if not raw_text:
            return {
                "word_count": 0.0,
                "char_count": 0.0,
                "lexical_diversity": 0.0,
                "skill_count": 0.0,
                "years_experience": 0.0
            }

        words = raw_text.split()
        word_count = len(words)
        char_count = len(raw_text)
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0.0

        skills_data = self.skill_extractor.extract_skills(raw_text)
        skill_count = float(skills_data["skill_count"])

        exp_data = self.experience_extractor.extract_experience_and_education(raw_text)
        years_exp = float(exp_data["estimated_years_experience"])

        return {
            "word_count": float(word_count),
            "char_count": float(char_count),
            "lexical_diversity": round(lexical_diversity, 4),
            "skill_count": skill_count,
            "years_experience": years_exp
        }

    def fit(self, raw_texts: List[str], labels: Optional[List[str]] = None) -> "FeaturePipeline":
        """
        Fits TF-IDF vectorizer, label encoder, and statistical scaler.
        """
        # Clean texts
        cleaned_texts = [self.text_cleaner.clean_text(t) for t in raw_texts]
        
        # Fit vectorizer
        self.vectorizer.fit(cleaned_texts)

        # Fit labels
        if labels is not None:
            self.label_encoder.fit(labels)

        self.is_fitted = True
        return self

    def transform(self, raw_texts: List[str]) -> csr_matrix:
        """
        Transforms raw text strings into TF-IDF sparse feature matrices.
        """
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before transforming data.")

        cleaned_texts = [self.text_cleaner.clean_text(t) for t in raw_texts]
        tfidf_features = self.vectorizer.transform(cleaned_texts)
        return tfidf_features

    def fit_transform(self, raw_texts: List[str], labels: Optional[List[str]] = None):
        """
        Fits and transforms in a single pass.
        """
        self.fit(raw_texts, labels)
        X = self.transform(raw_texts)
        y = self.label_encoder.transform(labels) if labels is not None else None
        return (X, y) if y is not None else X

    def transform_single(self, raw_text: str) -> csr_matrix:
        """Transform a single resume / JD text."""
        return self.transform([raw_text])

    def save(self, filepath_prefix: str) -> None:
        """Saves vectorizer and label encoder to disk."""
        joblib.dump(self.vectorizer, f"{filepath_prefix}_vectorizer.joblib")
        joblib.dump(self.label_encoder, f"{filepath_prefix}_label_encoder.joblib")

    @classmethod
    def load(cls, vectorizer_path: str, label_encoder_path: str) -> "FeaturePipeline":
        """Loads serialized feature pipeline components."""
        pipeline = cls()
        pipeline.vectorizer = joblib.load(vectorizer_path)
        pipeline.label_encoder = joblib.load(label_encoder_path)
        pipeline.is_fitted = True
        return pipeline
