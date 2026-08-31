"""
Contact and Profile Entity Extractor.
Extracts Candidate Name, Email, Phone Numbers, LinkedIn, GitHub, and Location.
"""
import re
from typing import Dict, List, Optional


class ContactExtractor:
    """
    Extracts contact info, online profiles, and candidate identity details.
    """

    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    
    PHONE_REGEX = re.compile(
        r"(?:(?:\+?([1-9]\d{0,2})[\s.-]?)?(?:\(?(\d{2,4})\)?[\s.-]?)?(\d{3,4})[\s.-]?(\d{3,4}))"
    )
    
    LINKEDIN_REGEX = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_-]+)", re.IGNORECASE)
    GITHUB_REGEX = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)", re.IGNORECASE)
    
    LOCATION_HINTS = [
        "New York", "San Francisco", "Seattle", "Austin", "Boston", "Chicago", "Los Angeles",
        "London", "Berlin", "Paris", "Toronto", "Vancouver", "Sydney", "Singapore",
        "Bangalore", "Bengaluru", "Hyderabad", "Mumbai", "Pune", "Delhi", "Chennai", "Remote"
    ]

    HEADER_STOPWORDS = {
        "resume", "curriculum vitae", "cv", "profile", "contact", "summary", "experience",
        "education", "skills", "projects", "objective", "personal details", "phone", "email"
    }

    @classmethod
    def extract_contacts(cls, text: str) -> Dict[str, any]:
        """
        Extracts all identifiable contact entities from the text.
        """
        if not text:
            return {
                "name": None,
                "email": None,
                "phone": None,
                "linkedin": None,
                "github": None,
                "location": None
            }

        # 1. Email Extraction
        emails = cls.EMAIL_REGEX.findall(text)
        email = emails[0] if emails else None

        # 2. Phone Extraction
        phone = None
        for line in text.split("\n")[:20]: # Check first 20 lines
            matches = cls.PHONE_REGEX.findall(line)
            if matches:
                digits = re.sub(r"\D", "", line)
                if 9 <= len(digits) <= 15:
                    phone = line.strip()
                    break

        # 3. LinkedIn
        linkedin_match = cls.LINKEDIN_REGEX.search(text)
        linkedin = f"https://linkedin.com/in/{linkedin_match.group(1)}" if linkedin_match else None

        # 4. GitHub
        github_match = cls.GITHUB_REGEX.search(text)
        github = f"https://github.com/{github_match.group(1)}" if github_match else None

        # 5. Candidate Name Extraction
        name = cls._extract_candidate_name(text)

        # 6. Location Detection
        location = cls._extract_location(text)

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "location": location
        }

    @classmethod
    def _extract_candidate_name(cls, text: str) -> Optional[str]:
        """
        Heuristic: Identifies candidate name from top lines,
        handling pipes, dashes, and titles.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:8]:
            lower_line = line.lower()
            if any(stop in lower_line for stop in cls.HEADER_STOPWORDS):
                continue
            if "@" in line or "http" in lower_line:
                continue
            
            # Split candidate name if header contains title delimiter (e.g. "Jane Doe | Senior Data Scientist")
            candidate_part = line
            for delim in ["|", "–", "—", "-", "•"]:
                if delim in candidate_part:
                    candidate_part = candidate_part.split(delim)[0].strip()

            words = candidate_part.split()
            if 2 <= len(words) <= 4:
                if all(w.replace(".", "").isalpha() for w in words):
                    return candidate_part
        return None

    @classmethod
    def _extract_location(cls, text: str) -> Optional[str]:
        """Scans for known metropolitan hubs and 'Remote'."""
        for loc in cls.LOCATION_HINTS:
            pattern = r"\b" + re.escape(loc) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                return loc
        return None
