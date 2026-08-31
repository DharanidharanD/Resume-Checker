"""
Skill Extraction Engine with Multi-Domain Taxonomy and N-Gram Matching.
"""
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class SkillExtractor:
    """
    Extracts, categorizes, and scores technical and soft skills from resume text.
    """

    # Domain Skill Taxonomy
    SKILL_TAXONOMY: Dict[str, List[str]] = {
        "Data Science & AI": [
            "python", "r", "machine learning", "deep learning", "nlp", "natural language processing",
            "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
            "numpy", "scipy", "xgboost", "lightgbm", "catboost", "llm", "large language models",
            "langchain", "hugging face", "transformers", "opencv", "bert", "gpt", "spacy", "nltk",
            "data visualization", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
            "data mining", "statistical analysis", "time series", "a/b testing", "feature engineering",
            "mlops", "mlflow", "dvc", "wandb", "prompt engineering", "reinforcement learning"
        ],
        "Web & Full-Stack Development": [
            "javascript", "typescript", "html", "html5", "css", "css3", "sass", "less",
            "react", "react.js", "next.js", "vue", "vue.js", "nuxt.js", "angular", "svelte",
            "node.js", "express", "express.js", "django", "flask", "fastapi", "spring boot",
            "asp.net", ".net core", "ruby on rails", "php", "laravel", "graphql", "rest api",
            "restful", "tailwind css", "bootstrap", "material ui", "redux", "zustand",
            "websockets", "microservices", "frontend development", "backend development"
        ],
        "Software Engineering & Systems": [
            "java", "c++", "c", "c#", "go", "golang", "rust", "scala", "kotlin", "swift",
            "object-oriented programming", "oop", "data structures", "algorithms", "design patterns",
            "system design", "concurrency", "multithreading", "clean code", "unit testing",
            "tdd", "test driven development", "solid principles", "mvc architecture", "distributed systems"
        ],
        "Cloud & DevOps": [
            "aws", "amazon web services", "azure", "google cloud", "gcp", "docker", "kubernetes",
            "k8s", "terraform", "ansible", "jenkins", "gitlab ci", "github actions", "circleci",
            "ci/cd", "continuous integration", "continuous deployment", "helm", "prometheus",
            "grafana", "elk stack", "elasticsearch", "logstash", "kibana", "nginx", "apache",
            "linux", "bash", "shell scripting", "serverless", "aws lambda", "cloudformation"
        ],
        "Databases & Big Data": [
            "sql", "mysql", "postgresql", "oracle", "microsoft sql server", "sqlite", "mongodb",
            "cassandra", "redis", "dynamodb", "neo4j", "couchdb", "hadoop", "spark",
            "apache spark", "pyspark", "kafka", "apache kafka", "hive", "snowflake", "bigquery",
            "redshift", "databricks", "etl", "data warehousing", "dbt", "airflow", "apache airflow"
        ],
        "Cyber Security": [
            "cybersecurity", "information security", "penetration testing", "ethical hacking",
            "vulnerability assessment", "siem", "soc", "firewalls", "ids/ips", "wireshark",
            "metasploit", "burp suite", "owasp", "incident response", "cryptography",
            "identity and access management", "iam", "cissp", "ceh", "comptia security+",
            "zero trust", "network security", "threat intelligence"
        ],
        "Mobile Development": [
            "android", "ios", "flutter", "react native", "swift", "kotlin", "dart",
            "xcode", "android studio", "jetpack compose", "swiftui", "mobile ui", "pwa",
            "app store deployment", "play store", "cross-platform"
        ],
        "Management & Agile": [
            "agile", "scrum", "kanban", "jira", "confluence", "trello", "sprint planning",
            "product roadmap", "stakeholder management", "project management", "pmp",
            "risk management", "user stories", "business analysis", "cross-functional leadership"
        ],
        "Human Resources (HR)": [
            "talent acquisition", "recruitment", "onboarding", "employee relations",
            "hris", "workday", "performance management", "payroll", "compliance",
            "compensation and benefits", "human resource management", "hr analytics",
            "employee engagement", "succession planning", "conflict resolution"
        ],
        "Finance & Accounting": [
            "financial analysis", "financial modeling", "accounting", "quickbooks", "sap",
            "budgeting", "forecasting", "variance analysis", "taxation", "auditing",
            "gaap", "ifrs", "risk analysis", "valuation", "portfolio management", "excel modeling"
        ],
        "Quality Assurance & Testing": [
            "qa", "software testing", "manual testing", "automation testing", "selenium",
            "cypress", "playwright", "junit", "pytest", "postman", "jmeter", "load testing",
            "regression testing", "test automation", "api testing", "bug tracking"
        ],
        "Soft Skills": [
            "communication", "teamwork", "leadership", "problem solving", "critical thinking",
            "time management", "adaptability", "collaboration", "work ethic", "creativity",
            "interpersonal skills", "negotiation", "conflict management", "active listening"
        ]
    }

    def __init__(self):
        # Pre-compile flattened skill patterns
        self.flat_skills: Dict[str, str] = {}  # lowercase_skill -> category
        self.regex_patterns = []

        for category, skills in self.SKILL_TAXONOMY.items():
            for skill in skills:
                skill_lower = skill.lower()
                self.flat_skills[skill_lower] = category

        # Sort skill keys by length descending to match multi-word phrases first
        # e.g., 'machine learning' before 'learning'
        sorted_skills = sorted(self.flat_skills.keys(), key=lambda s: len(s), reverse=True)

        for skill in sorted_skills:
            # Escape regex characters
            pattern_str = r"\b" + re.escape(skill) + r"\b"
            # Special treatment for skills with dots/plus (c++, node.js, .net)
            if skill in ["c++", "c#", ".net", "node.js", "vue.js", "react.js", "next.js", "nuxt.js"]:
                pattern_str = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
            self.regex_patterns.append((re.compile(pattern_str, re.IGNORECASE), skill, self.flat_skills[skill]))

    def extract_skills(self, text: str) -> Dict[str, any]:
        """
        Extracts all matched skills and categorizes them.
        
        Returns:
            Dict containing:
                - 'all_skills': List of unique extracted skills
                - 'by_category': Dict mapping Category -> List of skills
                - 'skill_count': Total number of unique skills
                - 'top_categories': List of (category, count) sorted by prevalence
        """
        if not text:
            return {
                "all_skills": [],
                "by_category": {},
                "skill_count": 0,
                "top_categories": []
            }

        extracted_set: Set[str] = set()
        by_category = defaultdict(list)

        for regex, canonical_skill, category in self.regex_patterns:
            if regex.search(text):
                if canonical_skill not in extracted_set:
                    extracted_set.add(canonical_skill)
                    by_category[category].append(canonical_skill)

        # Sort results
        top_categories = sorted(
            [(cat, len(skills)) for cat, skills in by_category.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "all_skills": sorted(list(extracted_set)),
            "by_category": dict(by_category),
            "skill_count": len(extracted_set),
            "top_categories": top_categories
        }

    def compute_skill_overlap(self, resume_skills: List[str], jd_skills: List[str]) -> Dict[str, any]:
        """
        Compares skills extracted from a resume against skills extracted from a Job Description.
        
        Returns:
            - 'matched_skills': List of overlapping skills
            - 'missing_skills': JD skills absent from resume
            - 'additional_skills': Resume skills not required by JD
            - 'match_ratio': Jaccard or overlap score between 0.0 and 1.0
        """
        r_set = set(s.lower() for s in resume_skills)
        j_set = set(s.lower() for s in jd_skills)

        if not j_set:
            return {
                "matched_skills": list(r_set),
                "missing_skills": [],
                "additional_skills": list(r_set),
                "match_ratio": 1.0 if r_set else 0.0
            }

        matched = r_set.intersection(j_set)
        missing = j_set.difference(r_set)
        additional = r_set.difference(j_set)
        match_ratio = len(matched) / len(j_set) if j_set else 0.0

        return {
            "matched_skills": sorted(list(matched)),
            "missing_skills": sorted(list(missing)),
            "additional_skills": sorted(list(additional)),
            "match_ratio": round(match_ratio, 4)
        }
