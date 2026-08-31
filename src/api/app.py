"""
FastAPI REST API Service for Resume Screening and Candidate Classification.
"""
import io
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.parsers.document_parser import DocumentParser
from src.preprocessing.text_cleaner import TextCleaner
from src.extractors.skill_extractor import SkillExtractor
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.experience_extractor import ExperienceExtractor
from src.models.classifier import ResumeClassifier
from src.screening.matcher import ResumeScreeningMatcher
from src.config import METRICS_REPORT_PATH, CATEGORIES

# Initialize FastAPI App
app = FastAPI(
    title="Resume Screening & Candidate Classification API",
    description="Production REST API for NLP-based resume parsing, candidate domain classification, and job description screening.",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate core engines
text_cleaner = TextCleaner()
skill_extractor = SkillExtractor()
contact_extractor = ContactExtractor()
experience_extractor = ExperienceExtractor()
matcher = ResumeScreeningMatcher()

# Global classifier instance (lazy load)
classifier = ResumeClassifier()


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------
class TextPayload(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text of the resume or document")

class ClassifyResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_k_predictions: List[Dict[str, Any]]
    all_probabilities: Dict[str, float]

class ScreenRequest(BaseModel):
    resume_text: str = Field(..., description="Raw text of the candidate resume")
    jd_text: str = Field(..., description="Raw text of the Job Description")
    candidate_name: Optional[str] = None
    skill_weight: Optional[float] = 0.50
    tfidf_weight: Optional[float] = 0.30
    experience_weight: Optional[float] = 0.20

class ParseResponse(BaseModel):
    contact_info: Dict[str, Any]
    skills: Dict[str, Any]
    experience_education: Dict[str, Any]
    text_statistics: Dict[str, Any]
    cleaned_text_preview: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["System"])
def root():
    return {
        "status": "online",
        "service": "Resume Screening & Candidate Classification AI",
        "version": "1.0.0",
        "supported_categories": CATEGORIES,
        "model_ready": classifier.is_ready
    }


@app.post("/api/v1/parse", response_model=ParseResponse, tags=["Parsing & Extraction"])
async def parse_resume_text(payload: TextPayload):
    """Parses raw text and extracts structured candidate profile and skills."""
    raw_text = payload.text
    contacts = contact_extractor.extract_contacts(raw_text)
    skills = skill_extractor.extract_skills(raw_text)
    exp_edu = experience_extractor.extract_experience_and_education(raw_text)
    cleaned = text_cleaner.clean_text(raw_text)

    words = raw_text.split()
    stats = {
        "word_count": len(words),
        "char_count": len(raw_text),
        "unique_words": len(set(w.lower() for w in words)),
        "skill_count": skills["skill_count"]
    }

    return {
        "contact_info": contacts,
        "skills": skills,
        "experience_education": exp_edu,
        "text_statistics": stats,
        "cleaned_text_preview": cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
    }


@app.post("/api/v1/parse-file", response_model=ParseResponse, tags=["Parsing & Extraction"])
async def parse_resume_file(file: UploadFile = File(...)):
    """Uploads and parses a PDF, DOCX, or TXT file."""
    try:
        content_bytes = await file.read()
        raw_text = DocumentParser.extract_text(content_bytes, filename=file.filename)
        
        contacts = contact_extractor.extract_contacts(raw_text)
        skills = skill_extractor.extract_skills(raw_text)
        exp_edu = experience_extractor.extract_experience_and_education(raw_text)
        cleaned = text_cleaner.clean_text(raw_text)

        words = raw_text.split()
        stats = {
            "word_count": len(words),
            "char_count": len(raw_text),
            "unique_words": len(set(w.lower() for w in words)),
            "skill_count": skills["skill_count"]
        }

        return {
            "contact_info": contacts,
            "skills": skills,
            "experience_education": exp_edu,
            "text_statistics": stats,
            "cleaned_text_preview": cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/classify", response_model=ClassifyResponse, tags=["Candidate Classification"])
async def classify_candidate(payload: TextPayload, top_k: int = Query(5, ge=1, le=12)):
    """Predicts candidate job domain category and class confidence probabilities."""
    if not classifier.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Classifier model is not trained yet. Run train_pipeline.py first."
        )
    return classifier.predict(payload.text, top_k=top_k)


@app.post("/api/v1/screen", tags=["Screening & Matching"])
async def screen_candidate(payload: ScreenRequest):
    """Screens candidate resume against a Job Description and returns composite match & skill gaps."""
    custom_matcher = ResumeScreeningMatcher(
        skill_weight=payload.skill_weight or 0.50,
        tfidf_weight=payload.tfidf_weight or 0.30,
        experience_weight=payload.experience_weight or 0.20
    )
    result = custom_matcher.screen_single(
        resume_text=payload.resume_text,
        jd_text=payload.jd_text,
        candidate_identifier=payload.candidate_name
    )
    return result


@app.post("/api/v1/batch-screen", tags=["Screening & Matching"])
async def batch_screen_resumes(
    files: List[UploadFile] = File(...),
    jd_text: str = Form(...)
):
    """Uploads multiple resume files and ranks all candidates against a target Job Description."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No resume files uploaded.")

    parsed_resumes = []
    for file in files:
        try:
            content = await file.read()
            text = DocumentParser.extract_text(content, filename=file.filename)
            parsed_resumes.append((file.filename, text))
        except Exception as e:
            parsed_resumes.append((file.filename, f"Error reading file: {str(e)}"))

    batch_results = matcher.batch_screen(parsed_resumes, jd_text)
    
    # Format response (convert DataFrame to JSON serializable list)
    return {
        "total_candidates": batch_results["total_candidates"],
        "ranked_candidates": batch_results["ranked_candidates"],
        "summary_table": batch_results["summary_df"].to_dict(orient="records")
    }


@app.get("/api/v1/models/metrics", tags=["Model Diagnostics"])
def get_model_metrics():
    """Returns saved model training diagnostics and evaluation report."""
    if not METRICS_REPORT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Evaluation metrics not found. Train model first."
        )
    with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
