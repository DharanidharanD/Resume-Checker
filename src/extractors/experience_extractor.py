"""
Experience and Education Entity Extractor.
Extracts degrees, calculated years of experience, and estimated seniority level.
"""
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ExperienceExtractor:
    """
    Extracts education degrees, computes years of professional experience, and estimates seniority.
    """

    DEGREE_PATTERNS = [
        (re.compile(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", re.IGNORECASE), "Doctorate / Ph.D."),
        (re.compile(r"\b(m\.?tech|m\.?s|master of science|master of engineering|m\.?c\.?a|mba|master of business administration|m\.?a|m\.?sc)\b", re.IGNORECASE), "Master's Degree"),
        (re.compile(r"\b(b\.?tech|b\.?e|bachelor of technology|bachelor of engineering|b\.?s|bachelor of science|b\.?c\.?a|bba|b\.?a|b\.?sc|bachelor)\b", re.IGNORECASE), "Bachelor's Degree"),
        (re.compile(r"\b(diploma|associate degree|associates degree)\b", re.IGNORECASE), "Associate / Diploma"),
        (re.compile(r"\b(high school|secondary education|12th grade)\b", re.IGNORECASE), "High School"),
    ]

    YEARS_EXP_DIRECT = re.compile(
        r"(\d+(?:\.\d+)?)\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)\b", 
        re.IGNORECASE
    )

    # Date range patterns: e.g. "2018 - 2023", "05/2019 - Present", "Jan 2020 - Current"
    DATE_RANGE_PATTERN = re.compile(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\.?\s*(?:19|20)\d{2})\s*(?:-|to|–|—)\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\.?\s*(?:19|20)\d{2}|present|current|now)",
        re.IGNORECASE
    )

    YEAR_ONLY_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

    @classmethod
    def extract_experience_and_education(cls, text: str) -> Dict[str, any]:
        """
        Extracts education degrees, estimated years of experience, and seniority category.
        """
        if not text:
            return {
                "degrees": [],
                "highest_degree": "Not Specified",
                "estimated_years_experience": 0.0,
                "seniority_level": "Entry-Level"
            }

        # 1. Degrees
        degrees = []
        highest_degree = "Not Specified"
        for pattern, degree_label in cls.DEGREE_PATTERNS:
            if pattern.search(text):
                degrees.append(degree_label)
                if highest_degree == "Not Specified":
                    highest_degree = degree_label

        # 2. Years of Experience
        years = cls._compute_years_experience(text)

        # 3. Seniority Classification
        seniority = cls._classify_seniority(years, text)

        return {
            "degrees": list(set(degrees)),
            "highest_degree": highest_degree,
            "estimated_years_experience": round(years, 1),
            "seniority_level": seniority
        }

    @classmethod
    def _compute_years_experience(cls, text: str) -> float:
        """
        Estimates total years of experience using direct mentions and date ranges.
        """
        # A. Check direct mentions first
        direct_matches = cls.YEARS_EXP_DIRECT.findall(text)
        if direct_matches:
            try:
                values = [float(v) for v in direct_matches if 0 < float(v) <= 40]
                if values:
                    return max(values)
            except Exception:
                pass

        # B. Parse date ranges
        current_year = datetime.now().year
        ranges = cls.DATE_RANGE_PATTERN.findall(text)
        total_months = 0

        for start_str, end_str in ranges:
            start_years = cls.YEAR_ONLY_PATTERN.findall(start_str)
            if not start_years:
                continue
            start_yr = int(start_years[0])
            
            if any(term in end_str.lower() for term in ["present", "current", "now"]):
                end_yr = current_year
            else:
                end_years = cls.YEAR_ONLY_PATTERN.findall(end_str)
                end_yr = int(end_years[0]) if end_years else current_year

            if 1970 <= start_yr <= current_year and start_yr <= end_yr <= current_year + 1:
                duration_years = end_yr - start_yr
                if 0 <= duration_years <= 30:
                    total_months += max(duration_years, 0.5) * 12

        if total_months > 0:
            calculated_years = total_months / 12.0
            return min(calculated_years, 35.0)

        # C. Default fallback
        return 1.0

    @classmethod
    def _classify_seniority(cls, years: float, text: str) -> str:
        """Classifies seniority into industry tiers."""
        text_lower = text.lower()
        if "lead" in text_lower or "principal" in text_lower or "director" in text_lower:
            if years >= 7:
                return "Lead / Staff"
            elif years >= 12:
                return "Principal / Director"

        if years < 2.0:
            return "Entry-Level (Junior)"
        elif 2.0 <= years < 5.0:
            return "Mid-Level"
        elif 5.0 <= years < 8.0:
            return "Senior"
        elif 8.0 <= years < 12.0:
            return "Lead / Staff"
        else:
            return "Principal / Executive"
