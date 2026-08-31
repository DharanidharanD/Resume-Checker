"""
Unit tests for SkillExtractor, ContactExtractor, and ExperienceExtractor.
"""
import pytest
from src.extractors.skill_extractor import SkillExtractor
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.experience_extractor import ExperienceExtractor


def test_skill_extraction():
    extractor = SkillExtractor()
    sample_text = """
    Senior Data Scientist with 5 years experience in Python, Machine Learning, TensorFlow, 
    Natural Language Processing, PyTorch, Docker, and Kubernetes.
    """
    res = extractor.extract_skills(sample_text)
    assert res["skill_count"] >= 5
    assert "python" in res["all_skills"]
    assert "machine learning" in res["all_skills"]
    assert "tensorflow" in res["all_skills"]
    assert "kubernetes" in res["all_skills"]


def test_skill_overlap():
    extractor = SkillExtractor()
    resume_skills = ["python", "machine learning", "sql", "docker", "fastapi"]
    jd_skills = ["python", "machine learning", "kubernetes", "aws"]
    
    overlap = extractor.compute_skill_overlap(resume_skills, jd_skills)
    assert "python" in overlap["matched_skills"]
    assert "machine learning" in overlap["matched_skills"]
    assert "kubernetes" in overlap["missing_skills"]
    assert "docker" in overlap["additional_skills"]
    assert overlap["match_ratio"] == 0.5


def test_contact_extraction():
    sample_text = """
    Alice Johnson
    Senior Software Engineer | New York, NY
    Email: alice.johnson@example.com | Phone: +1 (555) 234-5678
    LinkedIn: linkedin.com/in/alice-johnson-123 | GitHub: github.com/alicejohnson
    """
    contacts = ContactExtractor.extract_contacts(sample_text)
    assert contacts["email"] == "alice.johnson@example.com"
    assert "alice-johnson-123" in contacts["linkedin"]
    assert "alicejohnson" in contacts["github"]
    assert contacts["location"] == "New York"


def test_experience_and_education_extraction():
    sample_text = """
    Education: Master of Science in Computer Science, Stanford University (2018 - 2020)
    Bachelor of Technology in IT (2014 - 2018)
    Experience: Over 6 years of experience building distributed systems.
    """
    exp = ExperienceExtractor.extract_experience_and_education(sample_text)
    assert "Master's Degree" in exp["degrees"]
    assert "Bachelor's Degree" in exp["degrees"]
    assert exp["estimated_years_experience"] >= 6.0
    assert exp["seniority_level"] in ["Senior", "Lead / Staff"]
