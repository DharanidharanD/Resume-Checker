"""
Unit tests for FeaturePipeline and Classifier training.
"""
import pytest
import numpy as np
from src.features.feature_engineering import FeaturePipeline
from src.models.trainer import ModelTrainer


def test_feature_pipeline_fit_transform():
    texts = [
        "Python developer with Django and PostgreSQL experience",
        "Data scientist with machine learning and pandas skills",
        "DevOps engineer experienced in Kubernetes and Docker"
    ]
    labels = ["Web Development", "Data Science", "DevOps & Cloud"]

    pipe = FeaturePipeline()
    X, y = pipe.fit_transform(texts, labels)

    assert X.shape[0] == 3
    assert len(y) == 3
    assert len(pipe.label_encoder.classes_) == 3

    # Transform single
    single_vec = pipe.transform_single("Kubernetes docker cloud")
    assert single_vec.shape[0] == 1


def test_model_training_micro():
    texts = [
        # Data Science
        "Python machine learning deep learning neural networks tensorflow pytorch",
        "Python data scientist pandas numpy scikit-learn statistical analysis",
        "Data science natural language processing nlp bert models predictive analytics",
        "Machine learning engineer model evaluation feature engineering scikit-learn",
        "Deep learning computer vision opencv tensorflow pytorch neural net",
        
        # Web Development
        "React javascript typescript next.js css html frontend web",
        "Full stack node.js express react web developer html css",
        "Frontend developer vue.js javascript angular css tailwind",
        "Backend developer python django postgresql api rest graphql",
        "Web developer modern react redux next.js typescript frontend",
        
        # DevOps & Cloud
        "AWS docker kubernetes terraform devops ci/cd jenkins cloud",
        "Cloud engineer aws azure gcp terraform docker kubernetes",
        "Devops engineer continuous integration prometheus grafana ansible",
        "Site reliability engineer sre kubernetes docker cloud linux",
        "Infrastructure engineer terraform aws cloudformation docker helm"
    ]
    labels = [
        "Data Science", "Data Science", "Data Science", "Data Science", "Data Science",
        "Web Development", "Web Development", "Web Development", "Web Development", "Web Development",
        "DevOps & Cloud", "DevOps & Cloud", "DevOps & Cloud", "DevOps & Cloud", "DevOps & Cloud"
    ]

    trainer = ModelTrainer(random_state=42)
    res = trainer.train_and_evaluate(texts, labels, test_size=0.20)

    assert res["best_model_name"] is not None
    assert res["best_f1"] >= 0.80
