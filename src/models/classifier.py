"""
Resume Classifier Inference Engine.
Predicts job domain, confidence scores, and class probability distributions.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import joblib

from src.preprocessing.text_cleaner import TextCleaner
from src.config import CLASSIFIER_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH


class ResumeClassifier:
    """
    High-performance inference engine for candidate resume classification.
    """

    def __init__(
        self,
        classifier_path: Union[str, Path] = CLASSIFIER_PATH,
        vectorizer_path: Union[str, Path] = VECTORIZER_PATH,
        encoder_path: Union[str, Path] = LABEL_ENCODER_PATH
    ):
        self.classifier_path = Path(classifier_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.encoder_path = Path(encoder_path)
        
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.text_cleaner = TextCleaner()
        self._is_loaded = False

        if self.classifier_path.exists() and self.vectorizer_path.exists() and self.encoder_path.exists():
            self.load()

    def load(self) -> None:
        """Loads serialized model artifacts."""
        self.model = joblib.load(self.classifier_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.encoder_path)
        self._is_loaded = True

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self.model is not None

    def predict(self, raw_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classifies a single candidate resume.
        
        Returns:
            Dict containing:
                - 'predicted_category': Best predicted role domain
                - 'confidence': Probability score for top class (0.0 to 1.0)
                - 'top_k_predictions': List of dicts with role and probability
                - 'all_probabilities': Dict mapping all categories to probabilities
        """
        if not self.is_ready:
            raise RuntimeError("Classifier models not loaded. Train models first or verify artifact paths.")

        if not raw_text or not raw_text.strip():
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "top_k_predictions": [],
                "all_probabilities": {}
            }

        # Clean text & transform
        cleaned = self.text_cleaner.clean_text(raw_text)
        features = self.vectorizer.transform([cleaned])

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(features)[0]
            # Softmax
            exp_df = np.exp(df - np.max(df))
            probs = exp_df / np.sum(exp_df)
        else:
            pred_idx = self.model.predict(features)[0]
            probs = np.zeros(len(self.label_encoder.classes_))
            probs[pred_idx] = 1.0

        classes = self.label_encoder.classes_
        top_indices = np.argsort(probs)[::-1]

        best_idx = top_indices[0]
        best_category = str(classes[best_idx])
        confidence = float(probs[best_idx])

        top_k_preds = [
            {"category": str(classes[idx]), "probability": round(float(probs[idx]), 4)}
            for idx in top_indices[:top_k]
        ]

        all_probs = {
            str(classes[idx]): round(float(probs[idx]), 4)
            for idx in top_indices
        }

        return {
            "predicted_category": best_category,
            "confidence": round(confidence, 4),
            "top_k_predictions": top_k_preds,
            "all_probabilities": all_probs
        }

    def predict_batch(self, raw_texts: List[str]) -> List[Dict[str, Any]]:
        """Performs batch classification for multiple resumes."""
        if not self.is_ready:
            raise RuntimeError("Classifier models not loaded.")

        cleaned_texts = [self.text_cleaner.clean_text(t) for t in raw_texts]
        features = self.vectorizer.transform(cleaned_texts)

        if hasattr(self.model, "predict_proba"):
            probs_matrix = self.model.predict_proba(features)
        else:
            preds = self.model.predict(features)
            probs_matrix = np.zeros((len(raw_texts), len(self.label_encoder.classes_)))
            for i, p in enumerate(preds):
                probs_matrix[i, p] = 1.0

        classes = self.label_encoder.classes_
        results = []

        for probs in probs_matrix:
            top_indices = np.argsort(probs)[::-1]
            best_idx = top_indices[0]
            results.append({
                "predicted_category": str(classes[best_idx]),
                "confidence": round(float(probs[best_idx]), 4),
                "top_3_predictions": [
                    {"category": str(classes[idx]), "probability": round(float(probs[idx]), 4)}
                    for idx in top_indices[:3]
                ]
            })

        return results
