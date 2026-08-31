"""
Resume-to-Job-Description Screening, Skill Gap Analysis, and Candidate Ranking Engine.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import TextCleaner
from src.extractors.skill_extractor import SkillExtractor
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.experience_extractor import ExperienceExtractor
from src.config import DEFAULT_SCREENING_WEIGHTS


class ResumeScreeningMatcher:
    """
    Evaluates resumes against job descriptions with multi-faceted scoring and gap analysis.
    """

    def __init__(
        self,
        skill_weight: float = DEFAULT_SCREENING_WEIGHTS["skill_match_weight"],
        tfidf_weight: float = DEFAULT_SCREENING_WEIGHTS["semantic_tfidf_weight"],
        experience_weight: float = DEFAULT_SCREENING_WEIGHTS["experience_match_weight"],
    ):
        # Normalize weights to sum to 1.0
        total_w = skill_weight + tfidf_weight + experience_weight
        self.w_skill = skill_weight / total_w
        self.w_tfidf = tfidf_weight / total_w
        self.w_exp = experience_weight / total_w

        self.text_cleaner = TextCleaner()
        self.skill_extractor = SkillExtractor()
        self.contact_extractor = ContactExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.vectorizer = TfidfVectorizer(
            stop_words="english", 
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    def compute_tfidf_similarity(self, resume_text: str, jd_text: str) -> float:
        """Computes Cosine Similarity between resume text and job description."""
        cleaned_resume = self.text_cleaner.clean_text(resume_text)
        cleaned_jd = self.text_cleaner.clean_text(jd_text)

        if not cleaned_resume or not cleaned_jd:
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([cleaned_resume, cleaned_jd])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0

    def compute_experience_match(
        self, 
        candidate_years: float, 
        jd_years_req: float
    ) -> float:
        """
        Calculates experience score based on required vs. candidate years.
        """
        if jd_years_req <= 0.0:
            return 1.0
        if candidate_years >= jd_years_req:
            return 1.0
        ratio = candidate_years / jd_years_req
        return float(np.clip(ratio, 0.2, 0.95))

    def screen_single(
        self, 
        resume_text: str, 
        jd_text: str, 
        candidate_identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conducts thorough screening of a single resume against a job description.
        """
        # 1. Extract Profile & Contacts
        contacts = self.contact_extractor.extract_contacts(resume_text)
        candidate_name = contacts["name"] or candidate_identifier or "Candidate"

        # 2. Extract Skills from Resume and JD
        resume_skills_info = self.skill_extractor.extract_skills(resume_text)
        jd_skills_info = self.skill_extractor.extract_skills(jd_text)

        overlap = self.skill_extractor.compute_skill_overlap(
            resume_skills_info["all_skills"], 
            jd_skills_info["all_skills"]
        )

        skill_match_score = overlap["match_ratio"]

        # 3. Compute TF-IDF Semantic Similarity
        tfidf_similarity = self.compute_tfidf_similarity(resume_text, jd_text)

        # 4. Extract & Compare Experience
        cand_exp = self.experience_extractor.extract_experience_and_education(resume_text)
        jd_exp = self.experience_extractor.extract_experience_and_education(jd_text)
        
        cand_years = cand_exp["estimated_years_experience"]
        jd_years_req = jd_exp["estimated_years_experience"]
        
        exp_score = self.compute_experience_match(cand_years, jd_years_req)

        # 5. Composite Final Score (0 to 100%)
        final_score = (
            (self.w_skill * skill_match_score) +
            (self.w_tfidf * tfidf_similarity) +
            (self.w_exp * exp_score)
        ) * 100.0
        final_score = round(float(np.clip(final_score, 0.0, 100.0)), 2)

        # 6. Fit Recommendation & Valid Colors (green, yellow, red)
        if final_score >= 80.0:
            status = "Strong Match (Recommended for Interview)"
            badge_color = "green"
        elif final_score >= 60.0:
            status = "Moderate Match (Review Portfolio/Skills)"
            badge_color = "yellow"
        elif final_score >= 40.0:
            status = "Potential Match (Skill Gaps Exist)"
            badge_color = "yellow"
        else:
            status = "Low Match (Not Recommended)"
            badge_color = "red"

        return {
            "candidate_name": candidate_name,
            "final_score": final_score,
            "status": status,
            "badge_color": badge_color,
            "scores": {
                "skill_match_pct": round(skill_match_score * 100.0, 2),
                "tfidf_similarity_pct": round(tfidf_similarity * 100.0, 2),
                "experience_match_pct": round(exp_score * 100.0, 2),
            },
            "skills": {
                "matched_skills": overlap["matched_skills"],
                "missing_skills": overlap["missing_skills"],
                "additional_skills": overlap["additional_skills"],
                "total_candidate_skills": resume_skills_info["skill_count"],
                "total_jd_skills": jd_skills_info["skill_count"]
            },
            "candidate_profile": {
                "name": candidate_name,
                "email": contacts["email"],
                "phone": contacts["phone"],
                "linkedin": contacts["linkedin"],
                "github": contacts["github"],
                "location": contacts["location"],
                "highest_degree": cand_exp["highest_degree"],
                "years_experience": cand_years,
                "seniority_level": cand_exp["seniority_level"]
            },
            "jd_requirements": {
                "required_years_experience": jd_years_req,
                "required_degree": jd_exp["highest_degree"]
            }
        }

    def batch_screen(
        self, 
        resumes: List[Tuple[str, str]], 
        jd_text: str
    ) -> Dict[str, Any]:
        """
        Screens and ranks multiple resumes against a single Job Description.
        """
        results = []
        for ident, text in resumes:
            res = self.screen_single(text, jd_text, candidate_identifier=ident)
            res["filename"] = ident
            results.append(res)

        # Sort descending by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        summary_rows = []
        for r in results:
            summary_rows.append({
                "Rank": r["rank"],
                "Candidate Name": r["candidate_name"],
                "Filename": r.get("filename", ""),
                "Overall Match (%)": r["final_score"],
                "Skill Match (%)": r["scores"]["skill_match_pct"],
                "TF-IDF Sim (%)": r["scores"]["tfidf_similarity_pct"],
                "Years Exp": r["candidate_profile"]["years_experience"],
                "Matched Skills Count": len(r["skills"]["matched_skills"]),
                "Missing Skills Count": len(r["skills"]["missing_skills"]),
                "Status": r["status"]
            })

        df_summary = pd.DataFrame(summary_rows)

        return {
            "ranked_candidates": results,
            "summary_df": df_summary,
            "total_candidates": len(resumes),
            "top_candidate": results[0] if results else None
        }
