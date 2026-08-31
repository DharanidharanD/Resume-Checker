"""
Integration tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "supported_categories" in data


def test_parse_text_endpoint():
    payload = {
        "text": """
        Sarah Connor
        Senior DevOps Engineer | Seattle, WA
        Email: sarah.connor@cyberdyne.org | Phone: +1 206-555-0199
        Skills: AWS, Docker, Kubernetes, Terraform, Linux, CI/CD, Python
        Experience: 7 years managing cloud infrastructure.
        Education: Bachelor of Science in Computer Science
        """
    }
    response = client.post("/api/v1/parse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["contact_info"]["email"] == "sarah.connor@cyberdyne.org"
    assert "aws" in data["skills"]["all_skills"]
    assert "docker" in data["skills"]["all_skills"]
    assert data["experience_education"]["estimated_years_experience"] >= 7.0


def test_screen_endpoint():
    payload = {
        "resume_text": "Python, Machine Learning, Scikit-Learn, PyTorch, SQL developer with 4 years experience.",
        "jd_text": "Looking for Data Scientist with Python, Machine Learning, SQL, and 3+ years experience.",
        "candidate_name": "Test Candidate"
    }
    response = client.post("/api/v1/screen", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["final_score"] > 50.0
    assert "python" in data["skills"]["matched_skills"]
