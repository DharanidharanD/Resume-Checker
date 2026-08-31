# Technical Architecture & Mathematical Formulation

## 1. Feature Representation & Vectorization

### TF-IDF Vectorization
Given a corpus of documents $D$, the Term Frequency-Inverse Document Frequency (TF-IDF) weight for term $t$ in document $d$ is:

$$\text{tf}(t, d) = \text{frequency of } t \text{ in } d$$

$$\text{idf}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{tfidf}(t, d, D) = (1 + \log(\text{tf}(t, d))) \times \text{idf}(t, D) \quad \text{(Sublinear TF scaling)}$$

Vectors are normalized using the Euclidean $L_2$ norm:

$$\mathbf{v}_d = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

---

## 2. Resume-to-Job-Description Screening Formulation

The overall match score $S_{\text{final}}$ between candidate resume $R$ and Job Description $J$ is calculated as a convex combination of three normalized scores:

$$S_{\text{final}} = \left( w_1 \cdot S_{\text{skills}}(R, J) + w_2 \cdot S_{\text{semantic}}(R, J) + w_3 \cdot S_{\text{exp}}(R, J) \right) \times 100$$

where:
- $w_1 = 0.50$ (Skill Match Weight)
- $w_2 = 0.30$ (Semantic TF-IDF Cosine Similarity Weight)
- $w_3 = 0.20$ (Experience Alignment Weight)
- $\sum_{i=1}^3 w_i = 1.0$

### A. Skill Match Score ($S_{\text{skills}}$)
Let $\mathcal{S}_R$ be the set of skills extracted from the candidate's resume, and $\mathcal{S}_J$ be the set of required skills in the job description:

$$S_{\text{skills}} = \frac{|\mathcal{S}_R \cap \mathcal{S}_J|}{|\mathcal{S}_J|}$$

### B. Semantic TF-IDF Cosine Similarity ($S_{\text{semantic}}$)
Given the normalized TF-IDF representations $\mathbf{u}_R$ and $\mathbf{u}_J$:

$$S_{\text{semantic}} = \cos(\mathbf{u}_R, \mathbf{u}_J) = \frac{\mathbf{u}_R \cdot \mathbf{u}_J}{\|\mathbf{u}_R\|_2 \|\mathbf{u}_J\|_2}$$

### C. Experience Alignment Score ($S_{\text{exp}}$)
Let $Y_R$ be the candidate's estimated years of experience, and $Y_J$ be the required years specified in the Job Description:

$$S_{\text{exp}} = \begin{cases} 
1.0 & \text{if } Y_R \ge Y_J \text{ or } Y_J \le 0 \\
\min\left(0.95, \max\left(0.20, \frac{Y_R}{Y_J}\right)\right) & \text{if } Y_R < Y_J
\end{cases}$$

---

## 3. Multi-Class Candidate Classification Formulation

For candidate category classification, the model predicts the probability distribution over $K = 12$ domains:

$$P(y = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$

where $z_k$ is the logit/decision output for class $k$.

### Probability Calibration
For margin-based classifiers (e.g. Linear Support Vector Machines), probabilities are calibrated using Platt scaling via 3-Fold cross-validation (`CalibratedClassifierCV`):

$$P(y = 1 \mid f) = \frac{1}{1 + \exp(A \cdot f + B)}$$

---

## 4. Evaluation Metrics

1. **Weighted F1-Score**:
   $$\text{F1}_{\text{weighted}} = \sum_{k=1}^K \frac{N_k}{N} \cdot \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

2. **Macro F1-Score**:
   $$\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
