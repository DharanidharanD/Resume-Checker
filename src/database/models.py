"""
SQLAlchemy ORM Data Models for TalentMatrix AI Enterprise ATS.
"""
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class JobPosting(Base):
    """
    Represents an open requisition / Job Description.
    """
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False, index=True)
    department = Column(String(100), default="Engineering")
    location = Column(String(100), default="Remote")
    description = Column(Text, nullable=False)
    required_skills_json = Column(Text, default="[]")
    min_experience_years = Column(Float, default=3.0)
    skill_weight = Column(Float, default=0.50)
    tfidf_weight = Column(Float, default=0.30)
    exp_weight = Column(Float, default=0.20)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    screening_records = relationship("ScreeningRecord", back_populates="job", cascade="all, delete-orphan")

    @property
    def required_skills(self) -> List[str]:
        try:
            return json.loads(self.required_skills_json or "[]")
        except Exception:
            return []

    @required_skills.setter
    def required_skills(self, skills: List[str]):
        self.required_skills_json = json.dumps(skills)


class Candidate(Base):
    """
    Represents a candidate profile and parsed resume data.
    """
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False, index=True)
    email = Column(String(150), index=True)
    phone = Column(String(50))
    linkedin = Column(String(250))
    github = Column(String(250))
    location = Column(String(100))
    highest_degree = Column(String(100))
    years_experience = Column(Float, default=0.0)
    seniority_level = Column(String(50), default="Entry-Level")
    skills_json = Column(Text, default="[]")
    raw_text = Column(Text)
    resume_filename = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    screening_records = relationship("ScreeningRecord", back_populates="candidate", cascade="all, delete-orphan")

    @property
    def skills(self) -> List[str]:
        try:
            return json.loads(self.skills_json or "[]")
        except Exception:
            return []

    @skills.setter
    def skills(self, skills_list: List[str]):
        self.skills_json = json.dumps(skills_list)


class ScreeningRecord(Base):
    """
    Tracks screening evaluation results, match metrics, and ATS pipeline progression.
    """
    __tablename__ = "screening_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("job_postings.id"), nullable=False)
    
    overall_score = Column(Float, nullable=False, index=True)
    skill_score = Column(Float, default=0.0)
    tfidf_score = Column(Float, default=0.0)
    exp_score = Column(Float, default=0.0)
    
    matched_skills_json = Column(Text, default="[]")
    missing_skills_json = Column(Text, default="[]")
    additional_skills_json = Column(Text, default="[]")
    
    status = Column(String(50), default="Screened", index=True)  # Applied, Screened, Shortlisted, Interviewing, Offered, Rejected
    recommendation = Column(String(100), default="Potential Match")
    recruiter_notes = Column(Text, default="")
    screened_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    candidate = relationship("Candidate", back_populates="screening_records")
    job = relationship("JobPosting", back_populates="screening_records")

    @property
    def matched_skills(self) -> List[str]:
        try:
            return json.loads(self.matched_skills_json or "[]")
        except Exception:
            return []

    @matched_skills.setter
    def matched_skills(self, skills: List[str]):
        self.matched_skills_json = json.dumps(skills)

    @property
    def missing_skills(self) -> List[str]:
        try:
            return json.loads(self.missing_skills_json or "[]")
        except Exception:
            return []

    @missing_skills.setter
    def missing_skills(self, skills: List[str]):
        self.missing_skills_json = json.dumps(skills)
