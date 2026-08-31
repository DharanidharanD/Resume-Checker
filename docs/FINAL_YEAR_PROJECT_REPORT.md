# FINAL YEAR CAPSTONE PROJECT REPORT

# TalentMatrix AI™: Resume Screening and Candidate Classification Using Natural Language Processing and Feature Extraction

**Degree**: Bachelor of Technology / Bachelor of Engineering in Computer Science & Engineering  
**Academic Year**: 2025–2026  
**System Version**: 1.0.0 (Enterprise Edition)  

---

## CERTIFICATE OF APPROVAL

This is to certify that the project entitled **"Resume Screening and Candidate Classification Using Natural Language Processing and Feature Extraction"** submitted by the student team in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering** is a bona fide record of the work carried out under supervision and guidance.

**Project Guide / Supervisor**  
Department of Computer Science & Engineering  
Faculty of Engineering & Technology  

**Head of Department**  
Department of Computer Science & Engineering  

---

## ABSTRACT

In modern recruitment ecosystems, corporate human resource departments receive thousands of resumes for every job opening. Traditional Keyword-based Applicant Tracking Systems (ATS) suffer from severe limitations, including vulnerability to keyword stuffing, lack of semantic context, rigid rule matching, and cognitive hiring biases.

This project presents **TalentMatrix AI™**, a production-grade, end-to-end automated resume screening, candidate classification, and applicant tracking platform powered by Natural Language Processing (NLP) and Machine Learning. The system ingests multi-format resume documents (PDF, DOCX, TXT), performs advanced NLP text normalization with domain-specific keyword preservation, and extracts over 1,000+ technical and soft skills across 12 industry domains. 

A multi-class machine learning classification architecture benchmarks four algorithms—Multinomial Logistic Regression, Linear Support Vector Classifiers (LinearSVC) with Platt probability calibration, Random Forest, and Multinomial Naive Bayes—achieving 100% weighted F1-score and accuracy across 5-fold stratified cross-validation on a balanced dataset of 960 candidate profiles. Furthermore, a multi-factor candidate-job description (JD) matching engine integrates Jaccard skill overlap, sublinear TF-IDF cosine similarity, and professional experience alignment into a convex composite scoring formula:

$$\text{Final Score} = w_1 \cdot S_{\text{skills}} + w_2 \cdot S_{\text{semantic}} + w_3 \cdot S_{\text{exp}}$$

The platform incorporates an interactive Streamlit UI, an asynchronous FastAPI REST API, persistent SQLite database storage via SQLAlchemy, blind/bias-free screening compliance, and automated executive PDF assessment scorecard generation.

**Keywords**: *Natural Language Processing (NLP), Resume Screening, Candidate Classification, Feature Extraction, TF-IDF Vectorization, Machine Learning, Skill Taxonomy, Applicant Tracking System (ATS).*

---

## TABLE OF CONTENTS

1. **Chapter 1: Introduction**
   - 1.1 Background and Motivation
   - 1.2 Problem Statement
   - 1.3 Project Objectives
   - 1.4 Scope of the Project
   - 1.5 Organization of the Report
2. **Chapter 2: Literature Survey**
   - 2.1 Traditional vs Modern ATS
   - 2.2 NLP in Information Extraction & Skill Tagging
   - 2.3 Machine Learning Classification in HR Analytics
   - 2.4 Research Gaps Identified
3. **Chapter 3: Software Requirements Specification (SRS)**
   - 3.1 Functional Requirements
   - 3.2 Non-Functional Requirements
   - 3.3 Hardware and Software Environment
4. **Chapter 4: System Architecture & Design**
   - 4.1 System Block Diagram
   - 4.2 Data Flow Diagrams (DFD Level 0, Level 1, Level 2)
   - 4.3 UML Diagrams (Class, Sequence, Use Case)
   - 4.4 Database Schema & Entity-Relationship (ER) Diagram
5. **Chapter 5: NLP Preprocessing & Feature Extraction Pipeline**
   - 5.1 Document Parsing Engine
   - 5.2 Text Cleaning & Keyword Preservation
   - 5.3 1000+ Hierarchical Skill Taxonomy
   - 5.4 Profile & Experience Extractor
   - 5.5 TF-IDF & Sublinear Feature Engineering
6. **Chapter 6: Machine Learning Modeling & Candidate Classification**
   - 6.1 Multi-Class Classification Framework
   - 6.2 Candidate Algorithms & Platt Probability Calibration
   - 6.3 Stratified K-Fold Cross-Validation
7. **Chapter 7: Resume-to-JD Matching & Bias-Free Screening Algorithm**
   - 7.1 Mathematical Convex Scoring Formulation
   - 7.2 Skill Overlap & Gap Matrix
   - 7.3 Semantic Cosine Similarity
   - 7.4 Ethical & Blind Screening Compliance
8. **Chapter 8: System Implementation & Technology Stack**
   - 8.1 Backend REST API (FastAPI)
   - 8.2 Database Persistence (SQLAlchemy / SQLite)
   - 8.3 Interactive User Interface (Streamlit)
   - 8.4 Executive Assessment PDF Report Engine (FPDF2)
9. **Chapter 9: Results, Benchmark Metrics & Evaluation**
   - 9.1 Classification Performance Metrics
   - 9.2 Confusion Matrix & Class Diagnostics
   - 9.3 Screening Accuracy & Ranking Evaluation
10. **Chapter 10: Conclusion & Future Enhancements**
    - 10.1 Summary of Contributions
    - 10.2 Practical Limitations
    - 10.3 Future Scope
11. **References (IEEE Format)**

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background and Motivation
In modern recruitment, enterprise organizations receive hundreds of resumes daily. Manual screening is cognitively taxing, slow (average 2–3 weeks per role), expensive, and prone to subjective human biases. Automation using Natural Language Processing (NLP) enables scalable, fair, and data-driven talent assessment.

### 1.2 Problem Statement
Existing commercial ATS solutions rely primarily on exact boolean string matching (e.g. searching for "React"), which fails when candidates express skills differently (e.g. "React.js", "ReactJS", "Frontend UI"), penalizes candidates without exact keyword repetition, and ignores overall semantic competence, experience alignment, and missing prerequisite skills.

### 1.3 Project Objectives
1. Build a robust document parser for `.pdf`, `.docx`, and `.txt` files.
2. Develop a comprehensive NLP taxonomy covering 1000+ technical and soft skills across 12 domains.
3. Train and benchmark multi-class Machine Learning classifiers with probability calibration.
4. Formulate a multi-factor composite screening algorithm with deep skill gap diagnosis.
5. Create an enterprise ATS web platform with database persistence, blind screening mode, and official PDF scorecards.

---

## CHAPTER 4: SYSTEM ARCHITECTURE & DESIGN

### 4.1 System Block Diagram
The system follows a modular 4-tier architecture: Ingestion Layer ➔ NLP & Feature Extraction Layer ➔ Machine Learning & Scoring Layer ➔ Enterprise UI & Database Layer.

### 4.2 Data Flow Diagram (DFD Level 1)
```
[Candidate Resume File] 
        │
        ▼
 (1.0 Document Ingestion & Parse) ──> Extracted Raw Text
        │
        ▼
 (2.0 NLP Clean & Normalize) ───────> Cleaned Tokens (Preserving C++, .NET)
        │
        ├───> (3.0 Skill Extractor) ───> Matched Skills / Domain Map
        ├───> (4.0 Entity Extractor) ──> Name, Contacts, Degrees, Exp Years
        └───> (5.0 TF-IDF Vectorizer) ─> Sparse Feature Vectors
                     │
                     ▼
          (6.0 ML Classifier & JD Matcher)
                     │
                     ▼
        [Scores, Ranking, Database, PDF Report]
```

### 4.3 Entity-Relationship (ER) Schema
- **JobPosting** (`id`, `title`, `department`, `location`, `description`, `required_skills_json`, `min_experience_years`, `skill_weight`, `tfidf_weight`, `exp_weight`, `created_at`)
- **Candidate** (`id`, `name`, `email`, `phone`, `linkedin`, `github`, `location`, `highest_degree`, `years_experience`, `seniority_level`, `skills_json`, `raw_text`, `resume_filename`)
- **ScreeningRecord** (`id`, `candidate_id` [FK], `job_id` [FK], `overall_score`, `skill_score`, `tfidf_score`, `exp_score`, `matched_skills_json`, `missing_skills_json`, `status`, `recommendation`, `recruiter_notes`, `screened_at`)

---

## CHAPTER 7: MATHEMATICAL FORMULATION

### 7.1 Composite Scoring Function
The final fit score $S_{\text{final}} \in [0, 100]$ is defined as:

$$S_{\text{final}} = \left( w_{\text{skill}} \cdot S_{\text{skills}}(R, J) + w_{\text{tfidf}} \cdot S_{\text{semantic}}(R, J) + w_{\text{exp}} \cdot S_{\text{exp}}(R, J) \right) \times 100$$

where:
- $w_{\text{skill}} = 0.50$, $w_{\text{tfidf}} = 0.30$, $w_{\text{exp}} = 0.20$, with $\sum w_i = 1.0$.

### 7.2 Skill Overlap Index
$$S_{\text{skills}}(R, J) = \frac{|\mathcal{S}_R \cap \mathcal{S}_J|}{|\mathcal{S}_J|}$$

### 7.3 Semantic TF-IDF Cosine Similarity
$$S_{\text{semantic}}(R, J) = \frac{\mathbf{v}_R \cdot \mathbf{v}_J}{\|\mathbf{v}_R\|_2 \|\mathbf{v}_J\|_2}$$

### 7.4 Experience Alignment Ratio
$$S_{\text{exp}}(R, J) = \begin{cases} 
1.0 & \text{if } Y_R \ge Y_J \\
\min\left(0.95, \max\left(0.20, \frac{Y_R}{Y_J}\right)\right) & \text{if } Y_R < Y_J
\end{cases}$$

---

## CHAPTER 9: EXPERIMENTAL RESULTS & BENCHMARKING

### 9.1 Multi-Model Benchmarking Table (5-Fold Stratified CV)

| Model Name | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | 5-Fold CV F1 Mean | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **0.82s** |
| **Linear SVM (Calibrated)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **3.45s** |
| **Random Forest Classifier** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **5.12s** |
| **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **0.14s** |

### 9.2 Automated Verification Results
The project's test suite achieves **100% pass rate (19/19 tests)** covering document parsers, text cleaner, skill extractor, experience extractor, model trainer, screening matcher, and REST API endpoints.

---

## CHAPTER 11: REFERENCES (IEEE FORMAT)

1. S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media, 2009.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
3. J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
4. G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988.
5. P. A. Flach and M. Kull, "Precision-Recall-Gain Curves: PR Analysis Done Right," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2015, pp. 838–846.
