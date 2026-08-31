"""
Database Connection and Session Manager for SQLite / SQLAlchemy.
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import DATA_DIR
from src.database.models import Base, JobPosting, Candidate, ScreeningRecord

DB_PATH = DATA_DIR / "talentmatrix.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes database tables."""
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Session:
    """Yields a database session."""
    session = SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def seed_initial_data_if_empty():
    """Seeds sample jobs and candidates if database is currently fresh."""
    init_db()
    session = SessionLocal()
    try:
        if session.query(JobPosting).count() == 0:
            # Seed Sample Jobs
            job1 = JobPosting(
                title="Senior Data Scientist - AI & Analytics",
                department="Data Science",
                location="San Francisco, CA (Hybrid)",
                description="Lead machine learning, statistical modeling, and NLP pipelines using Python, PyTorch, and SQL.",
                min_experience_years=4.0,
                skill_weight=0.50,
                tfidf_weight=0.30,
                exp_weight=0.20
            )
            job1.required_skills = ["python", "machine learning", "pytorch", "scikit-learn", "sql", "pandas", "nlp"]

            job2 = JobPosting(
                title="Senior Full Stack Engineer (React / Node.js)",
                department="Engineering",
                location="New York, NY (Remote)",
                description="Build scalable customer-facing web platforms with React, Next.js, TypeScript, and Node.js.",
                min_experience_years=5.0,
                skill_weight=0.50,
                tfidf_weight=0.30,
                exp_weight=0.20
            )
            job2.required_skills = ["react", "node.js", "typescript", "javascript", "postgresql", "rest api", "tailwind css"]

            job3 = JobPosting(
                title="Lead DevOps & Cloud Solutions Architect",
                department="Infrastructure",
                location="Austin, TX (Remote)",
                description="Architect cloud infrastructure and CI/CD pipelines across AWS, Kubernetes, and Terraform.",
                min_experience_years=6.0,
                skill_weight=0.50,
                tfidf_weight=0.30,
                exp_weight=0.20
            )
            job3.required_skills = ["aws", "docker", "kubernetes", "terraform", "ci/cd", "linux", "prometheus"]

            session.add_all([job1, job2, job3])
            session.commit()
            print("[+] Seeded initial job requisitions into database.")
    finally:
        session.close()
