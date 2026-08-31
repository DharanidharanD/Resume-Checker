# 📄 SmartResume AI: Resume Screening and Candidate Classification Using NLP & Feature Extraction

An end-to-end, production-grade Natural Language Processing (NLP) and Machine Learning system for automated resume parsing, candidate domain classification, entity & skill extraction, and job description (JD) matching & ranking.

---

## 🌟 Key Features

1. **Multi-Format Ingestion Engine**:
   - Parses `.pdf`, `.docx`, and `.txt` resumes and job descriptions.
   - Robust text cleaning, unicode normalization, and optional PII anonymization.

2. **Domain-Specific NLP & Skill Taxonomy**:
   - Deep extractor with 1000+ technical and soft skills categorized across 12+ domains (Data Science, Web Development, DevOps & Cloud, Cyber Security, Mobile, AI/ML, HR, Finance, QA, Management).
   - Candidate entity extraction (Email, Phone, LinkedIn, GitHub, Location).
   - Experience and Education recognition (Degrees, calculated years of experience, seniority tier).

3. **Multi-Class Candidate Classification**:
   - Benchmarks 4+ Machine Learning models (Linear SVM with probability calibration, Logistic Regression, Random Forest, Multinomial Naive Bayes).
   - Predicts candidate job domain with probability distribution and confidence scores.

4. **Job Description (JD) Matching & Gap Diagnostics**:
   - Composite Weighted Matching Score:
     $$\text{Final Score} = 0.50 \times \text{Skill Match} + 0.30 \times \text{TF-IDF Similarity} + 0.20 \times \text{Experience Alignment}$$
   - Skill Gap Analysis: Matched Skills, Missing Skills, and Bonus Skills.
   - Batch Screening & Candidate Leaderboard generation with downloadable CSV reports.

5. **Interactive UI & REST API Deployment**:
   - **Streamlit Web Application**: Multi-tab interactive UI for single/batch screening, skill exploration, domain classification, and model training studio.
   - **FastAPI REST API**: High-performance asynchronous REST endpoints with OpenAPI / Swagger documentation (`/docs`).
   - **Rich Command-Line Interface (CLI)**: CLI tool for terminal-based screening, parsing, and training.

---

## 🏗️ System Architecture

```
                       ┌─────────────────────────────┐
                       │  Resume (PDF / DOCX / TXT)  │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │   Multi-Format Doc Parser   │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │  NLP Preprocessing Engine   │
                       │  - Text Normalization       │
                       │  - Stopword Removal         │
                       │  - Lemmatization            │
                       └──────────────┬──────────────┘
                                      │
               ┌──────────────────────┼──────────────────────┐
               ▼                      ▼                      ▼
     ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
     │  Skill Extractor  │  │ Contact / Degree  │  │ TF-IDF Vectorizer │
     │  (1000+ Taxonomy) │  │ & Experience Extr │  │ (Unigram/Bigram)  │
     └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
               │                      │                      │
               │                      ▼                      │
               │            ┌───────────────────┐            │
               │            │   ML Classifier   │            │
               │            │ (SVM / RF / LogR) │            │
               │            └─────────┬─────────┘            │
               │                      │                      │
               ▼                      ▼                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Screening & JD Matching Engine                       │
│  - Skill Overlap (Matched / Missing / Bonus)                           │
│  - Cosine Similarity & Experience Ratio Alignment                      │
│  - Composite Fit Score & Hiring Recommendation                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ Streamlit UI │            │ FastAPI REST │            │ Terminal CLI │
└──────────────┘            └──────────────┘            └──────────────┘
```

---

## 📂 Project Structure

```
resume-screening-nlp/
│
├── data/
│   ├── processed/                # Preprocessed dataset (resumes_dataset.csv)
│   ├── sample_resumes/           # Sample candidate resumes (Data Science, DevOps, HR, etc.)
│   └── sample_jds/               # Sample target Job Descriptions
│
├── models/
│   └── saved_models/             # Serialized ML models, vectorizers, and diagnostic JSONs
│
├── src/
│   ├── parsers/
│   │   └── document_parser.py    # Multi-format document parser (PDF, DOCX, TXT)
│   ├── preprocessing/
│   │   └── text_cleaner.py       # Text cleaning, normalization, and tokenization
│   ├── extractors/
│   │   ├── skill_extractor.py    # 1000+ skill taxonomy & overlap engine
│   │   ├── contact_extractor.py  # Email, phone, LinkedIn, GitHub, location extractor
│   │   └── experience_extractor.py# Degree, years of experience, and seniority calculator
│   ├── features/
│   │   └── feature_engineering.py# TF-IDF and engineered statistical features
│   ├── models/
│   │   ├── trainer.py            # Model training, cross-validation & benchmark engine
│   │   ├── evaluator.py          # Classification reports & confusion matrices
│   │   └── classifier.py         # Inference classifier with probability distributions
│   ├── screening/
│   │   └── matcher.py            # Resume vs. JD screening, ranking & gap analyzer
│   ├── api/
│   │   └── app.py                # FastAPI REST API
│   ├── cli.py                    # Terminal CLI utility
│   └── config.py                 # System paths and hyperparameter configurations
│
├── app/
│   └── streamlit_app.py          # Interactive Streamlit Web Dashboard
│
├── scripts/
│   ├── generate_synthetic_data.py# Dataset & sample generator
│   └── train_pipeline.py         # End-to-end model training script
│
├── tests/                        # Comprehensive automated test suite
│   ├── test_cleaner.py
│   ├── test_extractors.py
│   ├── test_parsers.py
│   ├── test_matcher.py
│   ├── test_classifier.py
│   └── test_api.py
│
├── requirements.txt
├── ARCHITECTURE.md               # Detailed NLP & mathematical formulation
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Navigate to the project directory
cd C:\Users\My\.gemini\antigravity\scratch\resume-screening-nlp

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Machine Learning Pipeline

Generate dataset, benchmark models, and save artifacts:
```bash
python scripts/train_pipeline.py
```

### 3. Launch the Interactive Web Dashboard (Streamlit)

```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 4. Start the FastAPI REST API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```
Interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 💻 CLI Usage Examples

### 1. Parse Resume and Extract Skills
```bash
python -m src.cli parse data/sample_resumes/sample_resume_data_science.txt
```

### 2. Classify Candidate Resume into Job Domain
```bash
python -m src.cli classify data/sample_resumes/sample_resume_devops___cloud.txt --top-k 5
```

### 3. Screen Resume Against a Job Description
```bash
python -m src.cli screen --resume data/sample_resumes/sample_resume_data_science.txt --jd data/sample_jds/Senior_Data_Scientist_JD.txt
```

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:
```bash
pytest -v tests/
```

---

## 📊 Evaluation & Metrics

The system benchmarks multiple classifiers using **5-Fold Stratified Cross-Validation**:
- **Linear Support Vector Machine (LinearSVC)**
- **Logistic Regression (Multinomial)**
- **Random Forest Classifier**
- **Multinomial Naive Bayes**

Metrics tracked:
- Accuracy, Precision (Macro & Weighted), Recall (Macro & Weighted), F1-Score, Confusion Matrix.
