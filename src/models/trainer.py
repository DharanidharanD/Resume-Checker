"""
Multi-Model Classifier Training, Comparison, and Serialization Engine.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

from src.features.feature_engineering import FeaturePipeline
from src.models.evaluator import ModelEvaluator
from src.config import CLASSIFIER_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH, METRICS_REPORT_PATH


class ModelTrainer:
    """
    Trains, benchmarks, and persists candidate classification models.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_pipeline = FeaturePipeline()
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None
        self.comparison_results: List[Dict[str, Any]] = []

    def get_candidate_models(self) -> Dict[str, Any]:
        """Returns dictionary of candidate ML classifiers for benchmarking."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, 
                C=2.0, 
                random_state=self.random_state
            ),
            "Linear SVM (Calibrated)": CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, random_state=self.random_state),
                cv=3
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, 
                max_depth=25, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Multinomial Naive Bayes": MultinomialNB(
                alpha=0.1
            ),
        }

    def train_and_evaluate(
        self, 
        texts: List[str], 
        labels: List[str], 
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Trains and compares multiple classifiers on the dataset.
        Selects the best model and serializes all pipeline artifacts.
        """
        print(f"[*] Extracting features from {len(texts)} resume samples...")
        start_time = time.time()

        # Step 1: Feature Pipeline Transformation
        X, y = self.feature_pipeline.fit_transform(texts, labels)
        target_names = list(self.feature_pipeline.label_encoder.classes_)

        # Step 2: Stratified Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )

        candidates = self.get_candidate_models()
        self.comparison_results = []
        best_score = -1.0

        print(f"[*] Benchmarking {len(candidates)} candidate models across {len(target_names)} classes...")
        for name, model in candidates.items():
            t0 = time.time()
            # Fit
            model.fit(X_train, y_train)
            train_duration = time.time() - t0

            # Predict
            y_pred = model.predict(X_test)
            metrics = ModelEvaluator.evaluate(y_test, y_pred, target_names)

            # 5-fold Stratified Cross-Validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

            entry = {
                "model_name": name,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
                "cv_f1_std": round(float(np.std(cv_scores)), 4),
                "train_time_sec": round(train_duration, 2),
                "metrics": metrics
            }
            self.comparison_results.append(entry)
            print(f"    - {name:25s} | Test Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f} | CV F1: {np.mean(cv_scores):.4f}")

            # Track best model
            if metrics["f1_weighted"] > best_score:
                best_score = metrics["f1_weighted"]
                self.best_model = model
                self.best_model_name = name
                self.best_metrics = metrics

        total_time = time.time() - start_time
        print(f"[+] Best Model: {self.best_model_name} with Weighted F1: {best_score:.4f} (Total Time: {total_time:.2f}s)")

        # Persist winning artifacts
        self.save_artifacts()

        return {
            "best_model_name": self.best_model_name,
            "best_f1": best_score,
            "comparison_table": self.comparison_results,
            "best_metrics": self.best_metrics
        }

    def save_artifacts(
        self,
        classifier_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None
    ) -> None:
        """Serializes best classifier, vectorizer, label encoder, and evaluation report."""
        c_path = classifier_path or CLASSIFIER_PATH
        v_path = vectorizer_path or VECTORIZER_PATH
        e_path = encoder_path or LABEL_ENCODER_PATH
        m_path = metrics_path or METRICS_REPORT_PATH

        c_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, c_path)
        joblib.dump(self.feature_pipeline.vectorizer, v_path)
        joblib.dump(self.feature_pipeline.label_encoder, e_path)
        
        if self.best_metrics:
            save_payload = {
                "best_model_name": self.best_model_name,
                "comparison_results": self.comparison_results,
                "metrics": self.best_metrics
            }
            ModelEvaluator.save_metrics(save_payload, m_path)
        print(f"[+] Saved artifacts to {c_path.parent}")
