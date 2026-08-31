# Software Requirements Specification (SRS)
## TalentMatrix AI™: Resume Screening and Candidate Classification System
**Standard**: IEEE 830-1998 Format  
**Version**: 1.0.0  

---

### 1. Introduction
#### 1.1 Purpose
This document provides a formal Software Requirements Specification (SRS) for **TalentMatrix AI™**, specifying functional, non-functional, behavioral, and architectural constraints for automated candidate screening, NLP feature extraction, and candidate classification.

#### 1.2 Scope
TalentMatrix AI is an enterprise recruitment platform designed for HR departments, talent acquisition leaders, and recruitment agencies. The system ingests resumes in multiple document formats (PDF, DOCX, TXT), performs NLP feature extraction, executes multi-class domain classification, computes composite match scores against Job Descriptions, and persists applicant records in an ATS database.

#### 1.3 Definitions and Acronyms
- **ATS**: Applicant Tracking System
- **NLP**: Natural Language Processing
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **PII**: Personally Identifiable Information
- **SRS**: Software Requirements Specification
- **ORM**: Object-Relational Mapping (SQLAlchemy)

---

### 2. Overall Description
#### 2.1 Product Perspective
TalentMatrix AI operates as an integrated web application with an asynchronous REST API backend (FastAPI), an interactive frontend dashboard (Streamlit), and an embedded relational database (SQLite).

#### 2.2 User Classes and Characteristics
1. **Recruiter / HR Executive**: Uploads resumes, reviews candidate scorecards, advances candidates through ATS stages, and downloads PDF assessment reports.
2. **Hiring Manager / Department Head**: Creates and approves job requisitions, sets minimum experience thresholds, and customizes screening weights.
3. **System Administrator**: Trains and benchmarks ML models, inspects confusion matrices, and monitors system metrics.

---

### 3. System Features & Functional Requirements

#### 3.1 Document Ingestion & Text Extraction (FR-1)
- **FR-1.1**: The system MUST parse `.pdf`, `.docx`, `.txt`, and `.md` file formats.
- **FR-1.2**: The system MUST handle table layouts, headings, and paragraph blocks.
- **FR-1.3**: The system MUST support direct file paths and in-memory binary uploads.

#### 3.2 NLP Preprocessing & Entity Extraction (FR-2)
- **FR-2.1**: The system MUST preserve symbolic technology keywords (`C++`, `C#`, `.NET`, `Node.js`).
- **FR-2.2**: The system MUST extract email addresses, phone numbers, LinkedIn, and GitHub profiles.
- **FR-2.3**: The system MUST extract technical and soft skills mapped to a 1000+ skill taxonomy across 12 domains.
- **FR-2.4**: The system MUST calculate total professional experience from chronological date ranges and text statements.

#### 3.3 Candidate Classification Engine (FR-3)
- **FR-3.1**: The system MUST classify resumes into one of 12 industry categories.
- **FR-3.2**: The system MUST provide probability distribution scores for top predicted domains.

#### 3.4 Job Description Matching & Skill Gap Analysis (FR-4)
- **FR-4.1**: The system MUST calculate a composite fit score between 0% and 100%.
- **FR-4.2**: The system MUST allow custom screening weights ($w_{\text{skill}}, w_{\text{tfidf}}, w_{\text{exp}}$).
- **FR-4.3**: The system MUST categorize skills into Matched, Missing, and Bonus skills.

#### 3.5 ATS Pipeline & Database Persistence (FR-5)
- **FR-5.1**: The system MUST persist job postings, candidate profiles, and screening history in SQLite.
- **FR-5.2**: The system MUST allow updating candidate status across 5 ATS stages.

#### 3.6 Official Assessment PDF Report Generation (FR-6)
- **FR-6.1**: The system MUST generate official, branded candidate evaluation PDF reports with scorecards and signature blocks.

---

### 4. Non-Functional Requirements

#### 4.1 Performance Requirements
- Single resume parsing and screening inference MUST complete in $\le 1.5$ seconds.
- Batch ranking of 20 resumes MUST execute in $\le 10.0$ seconds.

#### 4.2 Security & Compliance
- The system MUST provide a Blind Screening Mode to mask PII.
- All database operations MUST use parameterized ORM queries to prevent SQL injection.

#### 4.3 Maintainability & Portability
- Codebase MUST be modular, type-annotated, and compatible with Windows, Linux, and macOS.
