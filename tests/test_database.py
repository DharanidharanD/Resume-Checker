"""
Unit tests for SQLAlchemy Database Models and Session Operations.
"""
import pytest
from src.database.connection import get_db_session, init_db, seed_initial_data_if_empty
from src.database.models import JobPosting, Candidate, ScreeningRecord


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    seed_initial_data_if_empty()


def test_job_posting_crud():
    session = get_db_session()
    try:
        # Create
        job = JobPosting(
            title="Test ML Engineer",
            department="AI Research",
            description="Build deep learning models in PyTorch.",
            min_experience_years=3.0,
            skill_weight=0.6,
            tfidf_weight=0.2,
            exp_weight=0.2
        )
        job.required_skills = ["python", "pytorch", "deep learning"]
        session.add(job)
        session.commit()

        # Read
        retrieved = session.query(JobPosting).filter(JobPosting.title == "Test ML Engineer").first()
        assert retrieved is not None
        assert "pytorch" in retrieved.required_skills
        assert retrieved.min_experience_years == 3.0

        # Delete
        session.delete(retrieved)
        session.commit()
    finally:
        session.close()


def test_candidate_and_screening_relationship():
    session = get_db_session()
    try:
        job = JobPosting(
            title="Temp DevOps Job",
            department="Infra",
            description="Manage Kubernetes clusters."
        )
        session.add(job)
        session.flush()

        cand = Candidate(
            name="John Test",
            email="john.test@example.com",
            years_experience=5.0
        )
        session.add(cand)
        session.flush()

        record = ScreeningRecord(
            candidate_id=cand.id,
            job_id=job.id,
            overall_score=85.5,
            status="Shortlisted"
        )
        record.matched_skills = ["kubernetes", "docker"]
        session.add(record)
        session.commit()

        # Verify
        rec = session.query(ScreeningRecord).filter(ScreeningRecord.id == record.id).first()
        assert rec.candidate.name == "John Test"
        assert rec.job.title == "Temp DevOps Job"
        assert "docker" in rec.matched_skills

        # Cleanup
        session.delete(rec)
        session.delete(cand)
        session.delete(job)
        session.commit()
    finally:
        session.close()
