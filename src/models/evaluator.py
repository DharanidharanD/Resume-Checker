"""
Model Evaluation and Metrics Reporting Module.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:
    """
    Evaluates multi-class classification models and generates structured diagnostic reports.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        target_names: List[str]
    ) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics.
        """
        acc = accuracy_score(y_true, y_pred)
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        report_dict = classification_report(
            y_true, 
            y_pred, 
            target_names=target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec_weighted), 4),
            "recall_weighted": round(float(rec_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "target_names": target_names,
            "confusion_matrix": cm,
            "classification_report": report_dict
        }

        return metrics

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], filepath: Path) -> None:
        """Saves evaluation report as pretty JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
