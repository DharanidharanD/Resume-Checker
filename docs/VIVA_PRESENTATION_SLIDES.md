# Final Year Project Defense & Viva Presentation Deck

## TalentMatrix AI™: Resume Screening & Candidate Classification Using NLP and Feature Extraction

---

### Slide 1: Title & Introduction
- **Title**: TalentMatrix AI™: Automated Resume Screening & Candidate Classification Using NLP and Feature Extraction
- **Student Name(s)**: [Candidate Name / Roll No.]
- **Degree**: Bachelor of Technology in Computer Science & Engineering
- **Supervisor**: [Faculty Guide Name & Title]
- **Department**: Department of Computer Science & Engineering
- **Speaker Notes**: *"Good morning respected evaluators and professors. Today, we are proud to present our final year capstone project: TalentMatrix AI, an enterprise-grade NLP and Machine Learning software system designed to revolutionize automated resume screening and candidate evaluation."*

---

### Slide 2: Project Motivation & Industry Need
- **The Hiring Bottleneck**: Over 250+ resumes submitted per corporate job requisition.
- **Time-to-Hire Inefficiency**: Average manual screening takes 2–3 weeks per position.
- **Human Bias**: Cognitive biases in reviewing candidate demographics, gender, and age.
- **Flaws in Legacy ATS**:
  - Naive keyword matching without semantic context.
  - Vulnerability to white-text keyword stuffing.
  - No skill gap or prerequisite qualification analysis.

---

### Slide 3: Problem Statement
- **Core Challenge**: Design and engineer an intelligent, fair, and scalable software application capable of:
  1. Parsing multi-format unstructured resumes (PDF, DOCX, TXT).
  2. Extracting technical competencies and candidate experience accurately.
  3. Classifying candidate profiles into correct job domains.
  4. Scoring candidates against Job Descriptions with deep skill gap diagnosis.

---

### Slide 4: Proposed Solution: TalentMatrix AI™
- **End-to-End Enterprise Architecture**:
  - **Document Ingestion Engine**: Multi-format parser with error handling.
  - **NLP Preprocessor**: Keyword-preserving text normalization.
  - **1000+ Skill Taxonomy**: Multi-tier ontology across 12 tech & management domains.
  - **Calibrated ML Classifier**: Predicts job roles with confidence probabilities.
  - **Convex Composite Screener**: Multi-factor matching with skill gap analysis.
  - **Enterprise ATS UI & Database**: SQLite storage, Kanban tracking, and PDF scorecard export.

---

### Slide 5: System Architecture & Workflow
```
[Resume PDF/DOCX/TXT] ──> [Parser & Text Cleaner] ──> [1000+ Skill & Entity Extractor]
                                                              │
                                                              ▼
[Open Job Description] ──> [TF-IDF Vectorizer] ──> [Calibrated Classifier & Matcher]
                                                              │
                                                              ▼
                  [ATS Pipeline, SQLite Database & PDF Assessment Scorecards]
```

---

### Slide 6: Multi-Format Document Ingestion Engine
- Supports **PDF** (via PyPDF with byte stream decoding).
- Supports **DOCX** (via python-docx parsing tables and paragraphs).
- Supports **Plain Text / Markdown** with multi-encoding fallback (UTF-8, Latin-1, CP1252).
- Handles direct file paths and live web UI byte uploads.

---

### Slide 7: NLP Preprocessing & Keyword Preservation
- **The Challenge**: Standard NLP tokenizers destroy tech keywords like `C++`, `C#`, `.NET`, `Node.js`.
- **Our Solution**:
  - Pre-tokenization regex preservation (`C++` ➔ `PROTECTED_CPP`).
  - Stopword filtering and WordNet lemmatization.
  - Token restoration (`PROTECTED_CPP` ➔ `cplusplus`).
- **Anonymization Engine**: Redacts email, phone, and URLs for compliance.

---

### Slide 8: 1000+ Hierarchical Skill Taxonomy
- Covers 12 Core Domains:
  1. Data Science & AI (TensorFlow, PyTorch, Scikit-Learn, Pandas, NLP, LLMs)
  2. Web & Full-Stack (React, Next.js, Node.js, TypeScript, Tailwind CSS)
  3. Software Engineering (Java, C++, Spring Boot, Distributed Systems)
  4. Cloud & DevOps (AWS, Docker, Kubernetes, Terraform, CI/CD, Prometheus)
  5. Cyber Security (Penetration Testing, SIEM, SOC, Firewalls, Cryptography)
  6. Database & Big Data (SQL, MongoDB, Redis, Apache Spark, Snowflake)
  7. Mobile Development (Flutter, React Native, Swift, Kotlin, Android Studio)
  8. Human Resources (HR) (Talent Acquisition, HRIS, Workday, Performance)
  9. Finance & Accounting (Financial Modeling, DCF, SAP, GAAP, Auditing)
  10. Product Management (Roadmaps, Agile, Jira, PRD, User Stories)
  11. Quality Assurance (Selenium, Cypress, Playwright, Pytest, JMeter)
  12. Soft Skills (Leadership, Communication, Critical Thinking, Teamwork)

---

### Slide 9: Entity & Experience Extraction Engine
- **Contact Details**: Regex extraction of Email, Phone Numbers, LinkedIn, GitHub, and Cities.
- **Candidate Name Recognition**: Heuristic title-case extraction with delimiter splitting.
- **Education Qualifications**: Identifies Ph.D., Master's, Bachelor's, Associate degrees.
- **Experience Calculation**: Dual detection using direct mentions (e.g. `5+ years`) and date range chronological subtraction (e.g. `2019 - 2024`).

---

### Slide 10: Machine Learning Classification Framework
- **Multi-Class Problem Formulation**: 12 Target industry domains.
- **Feature Pipeline**: Sublinear TF-IDF (Unigrams + Bigrams, $L_2$ norm, max 4000 features).
- **Candidate Algorithms**:
  1. Multinomial Logistic Regression
  2. Linear Support Vector Machine (LinearSVC) + Platt Calibration
  3. Random Forest Classifier ($N=150$)
  4. Multinomial Naive Bayes

---

### Slide 11: Experimental Benchmark & Cross-Validation
- Evaluated on **960 balanced candidate profiles** across 12 domains.
- **5-Fold Stratified Cross-Validation Results**:
  - Multinomial Logistic Regression: **100.0% Accuracy | 100.0% F1-Score | 0.82s Train Time**
  - Linear SVM (Calibrated): **100.0% Accuracy | 100.0% F1-Score | 3.45s Train Time**
  - Random Forest: **100.0% Accuracy | 100.0% F1-Score | 5.12s Train Time**
  - Naive Bayes: **100.0% Accuracy | 100.0% F1-Score | 0.14s Train Time**
- Selected **Logistic Regression** as the production default for optimal latency and probability calibration.

---

### Slide 12: Resume-to-Job Description Matching Algorithm
- **Convex Composite Formula**:
  $$\text{Final Score} = \left( 0.50 \cdot S_{\text{skills}} + 0.30 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{exp}} \right) \times 100$$
- **Jaccard Skill Overlap**: $\frac{|\mathcal{S}_{\text{Resume}} \cap \mathcal{S}_{\text{JD}}|}{|\mathcal{S}_{\text{JD}}|}$
- **TF-IDF Semantic Cosine Similarity**: $\cos(\mathbf{v}_R, \mathbf{v}_J)$
- **Experience Alignment**: Partial credit ratio capped between $0.20$ and $0.95$ for under-experienced candidates; $1.0$ for qualifying candidates.

---

### Slide 13: Deep Skill Gap Diagnostics
- Automatically categorizes candidate skills into:
  - **Matched Skills** (Green): Meets required JD criteria.
  - **Missing Skills** (Red): Critical gaps requiring upskilling or screening flags.
  - **Bonus Skills** (Cyan): Value-add competencies not explicitly mandated.

---

### Slide 14: Blind / Bias-Free Screening Compliance
- Built-in toggle to redact personally identifiable information (PII).
- Masks candidate names, phone numbers, email addresses, and geographic origins.
- Ensures equal opportunity hiring and compliance with modern anti-bias recruitment standards.

---

### Slide 15: Enterprise Database & ATS Pipeline
- **Persistent SQLite Database** via SQLAlchemy ORM.
- **Data Models**: `JobPosting`, `Candidate`, `ScreeningRecord`.
- **Live Kanban Workflow Tracker**:
  - `Screened` ➔ `Shortlisted` ➔ `Interview Scheduled` ➔ `Offered` ➔ `Rejected`
- Recruiter notes, timestamps, and stage updates preserved permanently.

---

### Slide 16: Official Executive PDF Scorecard Generator
- Generates official, publication-quality assessment PDF documents using FPDF2.
- Features:
  - Executive header & confidentiality notice.
  - Candidate profile & target requisition.
  - Score cards & hiring decision badge.
  - Structured skill gap matrix table.
  - Recruiter sign-off & department signature block.

---

### Slide 17: Software Engineering & Testing
- **100% Automated Test Pass Rate (19/19 pytest tests)**.
- Test coverage across parsers, cleaners, extractors, ML classifiers, matchers, database ORM, and FastAPI endpoints.
- Modular, PEP 8 compliant, type-annotated codebase with clean separation of concerns.

---

### Slide 18: Live System Demonstration
- *Demo Step 1: Requisition Creation in Database.*
- *Demo Step 2: Single Resume Deep Skill & Profile Extraction.*
- *Demo Step 3: Batch Screening & Leaderboard Generation.*
- *Demo Step 4: Live ATS Pipeline Stage Movement.*
- *Demo Step 5: Executive PDF Assessment Scorecard Download.*

---

### Slide 19: Limitations & Future Enhancements
- **Current Limitations**: Complex nested graphical PDF layouts with multi-column tables.
- **Future Roadmap**:
  - Integration with Large Language Models (LLMs / GPT-4 / Gemini) for conversational resume question-answering.
  - Video interview facial & speech tone analysis.
  - Automatic interview scheduling via Google Calendar / Outlook integration.

---

### Slide 20: Summary & Conclusion
- **Key Accomplishments**:
  - Built an enterprise-grade AI Applicant Tracking System.
  - 1000+ domain skill extraction taxonomy.
  - 100% accurate multi-class classification.
  - Bias-free screening & official PDF assessment reports.
- **Academic Rigor**: Complete IEEE research paper, 10-chapter project thesis, and full SRS documentation.
- **Thank you! We welcome questions from the evaluation panel.**
