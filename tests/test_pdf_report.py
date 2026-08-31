"""
Unit tests for Candidate PDF Assessment Report Generator.
"""
import pytest
from src.reports.pdf_generator import CandidateReportGenerator


def test_generate_pdf_report_bytes():
    screening_data = {
        "candidate_name": "Dr. Alan Turing",
        "final_score": 92.5,
        "status": "Strong Match (Recommended for Interview)",
        "scores": {
            "skill_match_pct": 95.0,
            "tfidf_similarity_pct": 88.0,
            "experience_match_pct": 100.0
        },
        "skills": {
            "matched_skills": ["python", "machine learning", "algorithms", "cryptography"],
            "missing_skills": ["docker"],
            "additional_skills": ["statistical analysis", "deep learning"]
        },
        "candidate_profile": {
            "email": "alan.turing@cambridge.edu",
            "phone": "+44 20 7946 0912",
            "location": "London, UK",
            "highest_degree": "Doctorate / Ph.D.",
            "years_experience": 10.0,
            "seniority_level": "Lead / Staff"
        }
    }

    pdf_bytes = CandidateReportGenerator.generate_pdf_bytes(screening_data, job_title="Lead AI Researcher")

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
