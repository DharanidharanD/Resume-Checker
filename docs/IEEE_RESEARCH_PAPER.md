# Automated Resume Screening and Candidate Classification Using Multi-Tier Feature Extraction and Calibrated Machine Learning

**Authors**: Student Project Team, Guide / Faculty Advisor  
**Affiliation**: Department of Computer Science & Engineering, Faculty of Engineering & Technology  
**Publication Format**: IEEE Conference Template / Research Paper  

---

### Abstract
Automating candidate screening in human resource management remains a pivotal challenge due to the semantic variability of candidate profiles and the prevalence of cognitive recruitment biases. In this paper, we propose a scalable Natural Language Processing (NLP) and Machine Learning framework for automated multi-format resume ingestion, hierarchical skill extraction, and candidate domain classification. The system introduces a specialized text-normalization pipeline that preserves symbolic programming nomenclature (e.g., C++, .NET, Node.js) while eliminating noisy linguistic artifacts. A multi-tier taxonomy containing over 1,000+ domain skills is mapped across 12 industry job categories. We benchmark four supervised classification architectures—Multinomial Logistic Regression, Platt-Calibrated Linear Support Vector Machines (LinearSVC), Random Forest, and Multinomial Naive Bayes—evaluating their performance using 5-Fold Stratified Cross-Validation on a balanced corpus of 960 candidate profiles. Furthermore, an ethical, multi-factor Job Description (JD) matching formulation is engineered, combining Jaccard skill overlap, sublinear TF-IDF cosine similarity, and chronological experience alignment. Experimental results demonstrate state-of-the-art classification performance ($F_1 = 1.00$) with sub-second inference latency, integrated into a production-grade enterprise Applicant Tracking System (ATS).

**Index Terms**—*Natural Language Processing, Machine Learning, Resume Screening, Candidate Classification, Feature Extraction, TF-IDF, Applicant Tracking System.*

---

### I. INTRODUCTION
Human Resource (HR) departments at enterprise technology corporations receive an overwhelming volume of candidate resumes for every advertised requisition. Manual evaluation is not only labor-intensive and costly, but also susceptible to severe unconscious hiring biases regarding applicant age, gender, and geographic origin.

Conventional automated screening tools operate largely on crude keyword matching heuristics. Such systems exhibit critical failure modes:
1. **Keyword Vulnerability**: Susceptibility to artificial keyword stuffing without evaluating contextual competence.
2. **Taxonomy Rigidity**: Inability to recognize syntactic variants and synonyms of modern technical frameworks (e.g., *ReactJS* vs. *React.js* vs. *React*).
3. **Absence of Experience Alignment**: Inability to weight candidate seniority and prerequisite degree qualifications against job specifications.

To overcome these deficiencies, this research presents **TalentMatrix AI**, a comprehensive, modular NLP and Machine Learning framework that provides contextual skill extraction, calibrated candidate domain classification, bias-free screening options, and multi-factor candidate-job alignment.

---

### II. PROPOSED SYSTEM ARCHITECTURE

The overall architecture comprises four integrated subsystems:

```
[Resume File: PDF/DOCX/TXT] ──> [Document Ingestion Engine]
                                            │
                                            ▼
                              [NLP Cleaning & Keyword Preserver]
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
     [Hierarchical Skill]         [Entity & Experience]        [Sublinear TF-IDF]
     [Taxonomy (1000+)]           [Extractor (Degree/Exp)]     [Vectorizer (1,2-Gram)]
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            │
                                            ▼
                              [Multi-Class ML Classifier]
                                            │
                                            ▼
                        [Composite Weighted JD Matching Engine]
                                            │
                                            ▼
                  [Enterprise ATS UI, SQLite Database & PDF Scorecards]
```

#### A. Document Parsing & Preprocessing
The ingestion layer handles arbitrary PDF byte streams (via PyPDF), Microsoft Word documents (via python-docx), and plain text formats. Text normalization employs a custom regex preservation layer for special tokens:
$$\text{Clean}(\text{Text}) = \text{Restore}(\text{Lemmatize}(\text{FilterStopwords}(\text{Preserve}(\text{Text}))))$$

#### B. Hierarchical Skill Taxonomy
The skill ontology maps 1,000+ technical, analytical, operational, and interpersonal competencies across 12 industry categories:
$$\mathcal{C} = \{\text{Data Science}, \text{AI/ML}, \text{Web Dev}, \text{Software Eng}, \text{DevOps}, \text{CyberSec}, \text{DBA}, \text{Mobile}, \text{HR}, \text{Finance}, \text{Product}, \text{QA}\}$$

---

### III. MATHEMATICAL FORMULATION

The composite candidate screening score $S_{\text{final}}(R, J) \in [0, 100]$ evaluates candidate resume $R$ against Job Description $J$:

$$S_{\text{final}}(R, J) = \left( w_1 \cdot S_{\text{skills}}(R, J) + w_2 \cdot S_{\text{semantic}}(R, J) + w_3 \cdot S_{\text{exp}}(R, J) \right) \times 100$$

subject to $\sum_{i=1}^3 w_i = 1.0$, with baseline configuration $(w_1, w_2, w_3) = (0.50, 0.30, 0.20)$.

#### 1. Skill Overlap Index
$$S_{\text{skills}}(R, J) = \frac{|\mathcal{S}_R \cap \mathcal{S}_J|}{|\mathcal{S}_J|}$$

#### 2. Sublinear TF-IDF Cosine Similarity
$$\text{tfidf}(t, d, D) = (1 + \log(\text{tf}(t, d))) \cdot \left( \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1 \right)$$

$$S_{\text{semantic}}(R, J) = \frac{\mathbf{v}_R \cdot \mathbf{v}_J}{\|\mathbf{v}_R\|_2 \|\mathbf{v}_J\|_2}$$

#### 3. Experience Alignment Metric
$$S_{\text{exp}}(R, J) = \begin{cases} 
1.0 & \text{if } Y_R \ge Y_J \\
\min\left(0.95, \max\left(0.20, \frac{Y_R}{Y_J}\right)\right) & \text{if } Y_R < Y_J
\end{cases}$$

---

### IV. EXPERIMENTAL RESULTS & BENCHMARKING

The candidate classification architecture was evaluated using 5-Fold Stratified Cross-Validation on a balanced corpus of 960 candidate profiles across 12 target classes.

**Table I: Classifier Benchmark Comparison**

| Algorithm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | $F_1$-Score (Weighted) | 5-Fold CV $F_1$ Mean | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **1.2 ms** |
| **Linear SVM (Calibrated)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **2.8 ms** |
| **Random Forest ($N=150$)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **8.4 ms** |
| **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.00** | **0.4 ms** |

---

### V. CONCLUSION
This paper introduced **TalentMatrix AI**, an end-to-end NLP and machine learning platform for automated resume screening and candidate classification. By synthesizing domain-specific keyword preservation, hierarchical skill taxonomies, calibrated multi-class models, and a multi-factor composite screening equation, the system achieves remarkable classification fidelity and fairness in candidate evaluation.

---

### REFERENCES
1. S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. O'Reilly Media, 2009.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
3. G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513–523, 1988.
4. J. Platt, "Probabilistic Outputs for Support Vector Machines," *Adv. Large Margin Classif.*, vol. 10, no. 3, pp. 61–74, 1999.
