"""
Unit tests for ResumeScreeningMatcher.
"""
import pytest
from src.screening.matcher import ResumeScreeningMatcher


@pytest.fixture
def matcher():
    return ResumeScreeningMatcher(skill_weight=0.50, tfidf_weight=0.30, experience_weight=0.20)


def test_screening_high_match(matcher):
    resume_text = """
    Jane Doe | Senior Data Scientist
    Email: jane.doe@example.com
    5+ years experience in Python, Machine Learning, Deep Learning, Scikit-Learn, PyTorch, SQL, Pandas.
    Master of Science in Data Science.
    """
    
    jd_text = """
    Looking for a Senior Data Scientist with 4+ years experience in Python, Machine Learning, 
    Scikit-Learn, PyTorch, and SQL. Master's degree preferred.
    """
    
    result = matcher.screen_single(resume_text, jd_text)
    assert result["final_score"] >= 65.0
    assert result["candidate_name"] == "Jane Doe"
    assert "python" in result["skills"]["matched_skills"]
    assert "sql" in result["skills"]["matched_skills"]


def test_batch_screening(matcher):
    cand1 = ("cand1.txt", "Developer with Python, Django, REST API, React, 3 years exp.")
    cand2 = ("cand2.txt", "HR Manager with Talent Acquisition, Payroll, Workday, 8 years exp.")
    
    jd = "Seeking Full Stack Developer with Python, Django, REST API, React."
    
    batch_res = matcher.batch_screen([cand1, cand2], jd)
    assert batch_res["total_candidates"] == 2
    assert batch_res["ranked_candidates"][0]["filename"] == "cand1.txt"
    assert batch_res["ranked_candidates"][0]["final_score"] > batch_res["ranked_candidates"][1]["final_score"]
