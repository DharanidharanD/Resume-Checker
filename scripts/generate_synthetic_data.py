"""
Synthetic Resume and Job Description Dataset Generator.
Generates realistic, rich multi-domain resume texts across 12 industry categories.
"""
import random
import csv
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

from src.config import DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RESUMES_DIR, SAMPLE_JDS_DIR, CATEGORIES

# Candidate Profile Components
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "David", "Emma", "Liam",
    "Sophia", "Noah", "Olivia", "Ethan", "Ava", "Lucas", "Mia", "Mason", "Isabella", "Aiden",
    "Priya", "Rahul", "Aarav", "Ananya", "Rohan", "Sneha", "Vikram", "Neha", "Arjun", "Pooja",
    "Wei", "Yuki", "Hao", "Mei", "Jin", "Carlos", "Elena", "Mateo", "Sofia", "Diego"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Sharma", "Patel", "Gupta", "Verma", "Singh", "Reddy", "Kumar", "Mehta", "Iyer", "Nair",
    "Chen", "Wang", "Zhang", "Liu", "Tanaka", "Sato", "Suzuki", "Kim", "Park", "Lee"
]

CITIES = [
    "New York, NY", "San Francisco, CA", "Seattle, WA", "Austin, TX", "Boston, MA",
    "Chicago, IL", "Los Angeles, CA", "Toronto, ON", "London, UK", "Berlin, Germany",
    "Bangalore, India", "Hyderabad, India", "Pune, India", "Singapore", "Remote"
]

UNIVERSITIES = [
    "Massachusetts Institute of Technology (MIT)", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "University of Washington", "Georgia Tech", "University of Texas at Austin",
    "University of Waterloo", "University of Toronto", "Imperial College London", "Technical University of Munich",
    "Indian Institute of Technology (IIT) Delhi", "IIT Bombay", "BITS Pilani", "National University of Singapore"
]

COMPANIES = [
    "Google", "Amazon", "Microsoft", "Meta", "Apple", "Netflix", "Uber", "Stripe", "Airbnb", "Spotify",
    "Salesforce", "Oracle", "Cisco", "IBM", "Databricks", "Snowflake", "Palantir", "Goldman Sachs",
    "JPMorgan Chase", "McKinsey & Company", "Deloitte", "Accenture", "Infosys", "TCS", "Cognizant"
]

CATEGORY_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "Data Science": {
        "titles": ["Data Scientist", "Lead Data Scientist", "Senior Data Scientist", "Quantitative Analyst", "Data Science Specialist"],
        "skills": ["Python", "R", "Machine Learning", "Scikit-Learn", "Pandas", "NumPy", "TensorFlow", "PyTorch", "SQL", "Data Visualization", "Matplotlib", "Seaborn", "Tableau", "Statistical Analysis", "A/B Testing", "Feature Engineering", "Time Series", "Data Mining"],
        "responsibilities": [
            "Developed predictive machine learning models improving customer retention by 22% using Scikit-Learn and XGBoost.",
            "Designed and analyzed A/B test experiments evaluating new recommendation algorithms for over 5M active users.",
            "Built automated ETL data pipelines in Python and SQL extracting insights from multi-terabyte data lakes.",
            "Formulated statistical hypothesis testing and time series forecasting models reducing inventory shortages by 18%.",
            "Delivered executive dashboards using Tableau and Matplotlib communicating complex data insights to business leaders."
        ],
        "projects": [
            "Customer Churn Prediction Engine with 89% AUC using Random Forest and LightGBM.",
            "Automated Sales Forecasting Pipeline utilizing ARIMA, Prophet, and PyTorch.",
            "Clustering Analysis & Customer Segmentation Platform using K-Means and PCA."
        ]
    },
    "Machine Learning / AI": {
        "titles": ["Machine Learning Engineer", "AI Researcher", "Senior ML Engineer", "Deep Learning Specialist", "LLM Engineer"],
        "skills": ["Python", "PyTorch", "TensorFlow", "Deep Learning", "NLP", "Computer Vision", "LLM", "Transformers", "Hugging Face", "BERT", "GPT", "LangChain", "MLOps", "MLflow", "Docker", "Kubernetes", "CUDA", "FastAPI", "Prompt Engineering"],
        "responsibilities": [
            "Fine-tuned transformer models (Llama, BERT) using Hugging Face and PyTorch for document summarization and Q&A.",
            "Deployed production MLOps pipelines using MLflow, Docker, and Kubernetes handling 100K+ daily inference requests.",
            "Optimized deep neural network architectures achieving 3x latency reduction using ONNX Runtime and TensorRT.",
            "Engineered computer vision classification and object detection pipelines using OpenCV and PyTorch.",
            "Constructed Retrieval-Augmented Generation (RAG) agents utilizing LangChain, ChromaDB, and OpenAI APIs."
        ],
        "projects": [
            "Enterprise RAG Question-Answering System powered by LangChain, FastAPI, and Milvus vector search.",
            "Real-Time Object Detection and Tracking using YOLOv8 and PyTorch on embedded edge devices.",
            "End-to-End LLM Prompt Evaluation and Fine-Tuning Pipeline with LoRA and PEFT."
        ]
    },
    "Web Development": {
        "titles": ["Full Stack Developer", "Frontend Engineer", "Backend Developer", "Senior React Developer", "Web Applications Engineer"],
        "skills": ["JavaScript", "TypeScript", "React", "React.js", "Next.js", "Node.js", "Express", "HTML5", "CSS3", "Tailwind CSS", "Redux", "REST API", "GraphQL", "PostgreSQL", "MongoDB", "Webpack", "WebSockets"],
        "responsibilities": [
            "Architected responsive, accessible user interfaces using Next.js, React, and Tailwind CSS serving 2M+ monthly visitors.",
            "Constructed robust RESTful and GraphQL APIs with Node.js, Express, and PostgreSQL with 99.9% uptime.",
            "Implemented real-time live collaboration features using WebSockets and Redis pub/sub.",
            "Enhanced web performance metrics, improving Core Web Vitals (LCP, INP, CLS) by 45%.",
            "Maintained state management across complex web applications using Redux Toolkit and React Query."
        ],
        "projects": [
            "E-Commerce Micro-Frontend Platform with Next.js 14, Stripe Payments, and Tailwind CSS.",
            "Real-Time Collaboration Dashboard using React, TypeScript, GraphQL Subscriptions, and Node.js.",
            "Content Management System (CMS) with Role-Based Access Control using Express and MongoDB."
        ]
    },
    "Software Engineering": {
        "titles": ["Software Engineer", "Senior Software Engineer", "Backend Systems Engineer", "Java Developer", "C++ Systems Developer"],
        "skills": ["Java", "C++", "C#", "Go", "Golang", "Data Structures", "Algorithms", "System Design", "Microservices", "Design Patterns", "Spring Boot", "OOP", "Clean Code", "Unit Testing", "Multithreading", "Distributed Systems"],
        "responsibilities": [
            "Engineered high-throughput, low-latency microservices handling 50K+ QPS using Java and Spring Boot.",
            "Refactored legacy monolithic services into decoupled event-driven microservices reducing deployment friction.",
            "Designed scalable relational and NoSQL data architectures ensuring high concurrency and fault tolerance.",
            "Enforced clean architecture, design patterns, and unit testing practices achieving 90%+ code coverage with JUnit.",
            "Optimized system memory footprint and multi-threaded throughput in high-performance C++ backend services."
        ],
        "projects": [
            "Distributed Key-Value Store with Raft consensus protocol implemented in Go.",
            "High-Throughput Financial Order Matching Engine in C++ with sub-millisecond execution times.",
            "Scalable Microservices Gateway with Spring Cloud, JWT authentication, and rate limiting."
        ]
    },
    "DevOps & Cloud": {
        "titles": ["DevOps Engineer", "Cloud Solutions Architect", "Site Reliability Engineer (SRE)", "Infrastructure Engineer", "Platform Engineer"],
        "skills": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "Linux", "Bash", "Prometheus", "Grafana", "ELK Stack", "Nginx", "Helm"],
        "responsibilities": [
            "Automated multi-region cloud infrastructure provisioning using Terraform and AWS CloudFormation.",
            "Administered production Kubernetes clusters (EKS/GKE), writing custom Helm charts and managing service meshes.",
            "Designed end-to-end CI/CD release pipelines with GitHub Actions and Jenkins, cutting deployment cycle times by 60%.",
            "Established comprehensive observability, alerting, and log aggregation using Prometheus, Grafana, and ELK Stack.",
            "Maintained 99.99% system availability through proactive reliability engineering and disaster recovery drills."
        ],
        "projects": [
            "Zero-Downtime Multi-Cluster Kubernetes Deployment Platform on AWS with ArgoCD.",
            "Automated Infrastructure as Code (IaC) Framework managing 100+ AWS environments via Terraform.",
            "Unified Observability Stack with Grafana dashboards, Prometheus metrics, and automated Slack alerts."
        ]
    },
    "Cyber Security": {
        "titles": ["Cyber Security Analyst", "Information Security Engineer", "Penetration Tester", "SOC Analyst", "Security Architect"],
        "skills": ["Cybersecurity", "Penetration Testing", "Ethical Hacking", "Vulnerability Assessment", "SIEM", "SOC", "Firewalls", "Wireshark", "Metasploit", "Burp Suite", "OWASP", "Incident Response", "Network Security", "Cryptography", "CISSP", "IAM"],
        "responsibilities": [
            "Executed red-team penetration tests and web application vulnerability assessments using Burp Suite and Metasploit.",
            "Monitored enterprise SIEM systems (Splunk/Sentinel), investigating anomalous alerts and mitigating cyber threats.",
            "Implemented Identity and Access Management (IAM) policies and Zero Trust architecture across cloud workloads.",
            "Conducted comprehensive incident response investigations following security breaches and malware incidents.",
            "Audited code repositories for OWASP Top 10 vulnerabilities and integrated SAST/DAST into CI/CD pipelines."
        ],
        "projects": [
            "Automated Vulnerability Scanning and Remediation Orchestrator with Python and Burp Suite API.",
            "Enterprise Zero-Trust Network Architecture Migration covering 2,500+ endpoints.",
            "Threat Intelligence Feeds Aggregator and Custom SIEM Correlation Rules Engine."
        ]
    },
    "Database Administration": {
        "titles": ["Database Administrator (DBA)", "Data Engineer", "Database Architect", "Big Data Engineer", "SQL Developer"],
        "skills": ["SQL", "PostgreSQL", "MySQL", "Oracle", "MongoDB", "Redis", "Apache Spark", "PySpark", "Kafka", "Snowflake", "Airflow", "ETL", "Data Warehousing", "Database Tuning", "Replication", "Sharding"],
        "responsibilities": [
            "Managed and tuned high-availability PostgreSQL and Oracle database clusters with read replicas and sharding.",
            "Engineered scalable data warehouse architectures on Snowflake and Google BigQuery supporting BI reporting.",
            "Constructed robust streaming ETL data pipelines with Apache Spark, Kafka, and Apache Airflow.",
            "Optimized slow SQL queries, indexed strategic columns, and resolved database locking and concurrency bottlenecks.",
            "Automated backup, recovery, database failover, and disaster recovery procedures ensuring zero data loss."
        ],
        "projects": [
            "Real-Time Streaming Analytics Pipeline processing 10M events/day with Kafka and PySpark.",
            "Automated Database Failover and Disaster Recovery Orchestration for PostgreSQL Clusters.",
            "Enterprise Data Lake Migration from on-premise Hadoop to Snowflake & AWS S3."
        ]
    },
    "Mobile App Development": {
        "titles": ["Mobile App Developer", "iOS Developer", "Android Developer", "Flutter Engineer", "React Native Developer"],
        "skills": ["Flutter", "React Native", "Swift", "Kotlin", "Dart", "Android Studio", "Xcode", "iOS", "Android", "Jetpack Compose", "SwiftUI", "REST API", "Mobile UI", "App Store Deployment", "Firebase"],
        "responsibilities": [
            "Engineered cross-platform mobile apps for iOS and Android using Flutter and React Native with 1M+ downloads.",
            "Designed elegant native user experiences using SwiftUI on iOS and Jetpack Compose on Android.",
            "Integrated push notifications, offline caching, and real-time backend synchronization using Firebase.",
            "Managed end-to-end App Store and Google Play Store submission, compliance, and beta testing with TestFlight.",
            "Optimized mobile application memory usage, battery consumption, and rendering frame rates to consistent 60 FPS."
        ],
        "projects": [
            "Fintech Mobile Wallet Application with biometric authentication and NFC payments (Flutter/Dart).",
            "Fitness Tracking iOS App with CoreML workout detection and HealthKit integration (Swift/SwiftUI).",
            "On-Demand Delivery Mobile Application with real-time GPS tracking (React Native)."
        ]
    },
    "Human Resources (HR)": {
        "titles": ["HR Manager", "Talent Acquisition Specialist", "HR Business Partner (HRBP)", "Recruiting Specialist", "People Operations Lead"],
        "skills": ["Talent Acquisition", "Recruitment", "Onboarding", "Employee Relations", "HRIS", "Workday", "Performance Management", "Payroll", "HR Analytics", "Compliance", "Employee Engagement", "Conflict Resolution", "Compensation and Benefits"],
        "responsibilities": [
            "Spearheaded full-cycle talent acquisition across technical and executive roles, sourcing and hiring 80+ candidates annually.",
            "Implemented modern HRIS and Applicant Tracking Systems (Workday, Greenhouse) streamlining hiring workflows.",
            "Partnered with department directors to orchestrate annual performance reviews, promotion cycles, and succession plans.",
            "Resolved sensitive employee relations issues and maintained full compliance with federal and state labor standards.",
            "Conducted employee engagement surveys and spearheaded wellness programs, reducing voluntary turnover by 15%."
        ],
        "projects": [
            "Global Technical Hiring Campaign scaling engineering headcount from 50 to 180 within 12 months.",
            "Company-Wide Performance Appraisal & Compensation Matrix Revamp across 500+ employees.",
            "Comprehensive Diversity, Equity, and Inclusion (DEI) Recruitment Framework."
        ]
    },
    "Finance & Accounting": {
        "titles": ["Financial Analyst", "Senior Accountant", "Corporate Finance Manager", "Investment Analyst", "Budget & Planning Specialist"],
        "skills": ["Financial Analysis", "Financial Modeling", "Accounting", "QuickBooks", "SAP", "Budgeting", "Forecasting", "Variance Analysis", "Taxation", "Auditing", "GAAP", "Excel Modeling", "Valuation", "Risk Analysis", "IFRS"],
        "responsibilities": [
            "Constructed multi-year financial forecast models, DCF valuations, and variance analyses for senior leadership.",
            "Managed month-end and year-end accounting close processes adhering to GAAP and IFRS compliance standards.",
            "Led budgeting processes across 6 corporate divisions with annual budgets exceeding $45M.",
            "Conducted internal financial audits, optimizing corporate tax filings and identifying $400K in cost efficiencies.",
            "Prepared monthly executive financial decks communicating P&L, EBITDA, balance sheet, and cash flow trajectories."
        ],
        "projects": [
            "Three-Statement Financial Model & DCF Valuation for a $60M Corporate Acquisition.",
            "ERP Financial System Migration from QuickBooks to SAP S/4HANA.",
            "Automated Corporate Expense Tracking and Variance Analysis Dashboard in Advanced Excel."
        ]
    },
    "Product Management": {
        "titles": ["Product Manager", "Senior Product Manager", "Technical Product Manager (TPM)", "Associate Product Manager", "Group Product Manager"],
        "skills": ["Product Roadmap", "Agile", "Scrum", "Jira", "Stakeholder Management", "User Stories", "A/B Testing", "Product Analytics", "Mixpanel", "Market Research", "Feature Prioritization", "Cross-Functional Leadership", "Wireframing"],
        "responsibilities": [
            "Defined multi-year product strategy, roadmap, and PRDs for flagship SaaS applications driving $12M ARR.",
            "Collaborated cross-functionally with engineering, UX design, and sales to launch 6 major enterprise features.",
            "Led Agile sprint ceremonies, backlog grooming, and user story definitions in Jira as Product Owner.",
            "Conducted user interviews, market research, and telemetry analysis using Mixpanel to inform feature prioritization.",
            "Executed data-driven conversion funnel optimization improving onboarding completion rates by 28%."
        ],
        "projects": [
            "Enterprise B2B SaaS Workflow Automation Module generating $3.5M in incremental revenue.",
            "Customer Onboarding Flow Redesign resulting in a 35% increase in product adoption.",
            "Self-Serve Pricing and Checkout Tier Launch with A/B testing and telemetry."
        ]
    },
    "Operations & QA": {
        "titles": ["QA Automation Engineer", "Operations Manager", "Software Test Lead", "Quality Assurance Analyst", "SDET"],
        "skills": ["QA", "Software Testing", "Automation Testing", "Selenium", "Cypress", "Playwright", "Pytest", "Postman", "API Testing", "JMeter", "Regression Testing", "Bug Tracking", "Jira", "Test Automation"],
        "responsibilities": [
            "Architected automated end-to-end testing frameworks using Cypress, Playwright, and Selenium.",
            "Created and maintained 800+ automated API and UI regression test suites integrated into CI/CD pipelines.",
            "Conducted performance, load, and stress testing using JMeter identifying critical server bottlenecks.",
            "Authored detailed test plans, test cases, and bug triage reports in Jira collaborating with developers.",
            "Streamlined cross-team operational workflows, slashing production bug escape rates by 35%."
        ],
        "projects": [
            "Enterprise Test Automation Suite with 95% automated test coverage in Cypress and Pytest.",
            "Continuous Performance & Load Testing Framework using JMeter and Grafana.",
            "Microservices API Testing Automation Framework with Postman and Newman."
        ]
    }
}


def generate_synthetic_resume(category: str, candidate_id: int) -> Dict[str, str]:
    """Generates a structured, realistic resume text for a given category."""
    template = CATEGORY_TEMPLATES[category]
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    name = f"{first_name} {last_name}"
    email = f"{first_name.lower()}.{last_name.lower()}{random.randint(10, 99)}@example.com"
    phone = f"+1 ({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
    location = random.choice(CITIES)
    linkedin = f"https://linkedin.com/in/{first_name.lower()}-{last_name.lower()}-{random.randint(100, 999)}"
    github = f"https://github.com/{first_name.lower()}{last_name.lower()}"

    title = random.choice(template["titles"])
    num_skills = random.randint(8, 14)
    selected_skills = random.sample(template["skills"], min(num_skills, len(template["skills"])))

    # Add soft skills
    soft_pool = ["Leadership", "Communication", "Problem Solving", "Collaboration", "Agile", "Critical Thinking", "Adaptability"]
    selected_skills += random.sample(soft_pool, 3)

    # Degree & University
    degree_options = ["Bachelor of Science in Computer Science", "Master of Science in Software Engineering", "B.Tech in Information Technology", "Ph.D. in Computer Science", "MBA in Business Analytics", "Bachelor of Business Administration"]
    degree = random.choice(degree_options)
    univ = random.choice(UNIVERSITIES)
    grad_year = random.randint(2012, 2023)

    # Work experiences
    years_exp = 2026 - grad_year
    num_jobs = min(3, max(1, years_exp // 3))
    
    experience_blocks = []
    curr_yr = 2026
    for i in range(num_jobs):
        company = random.choice(COMPANIES)
        start_yr = max(grad_year, curr_yr - random.randint(2, 4))
        end_yr_str = "Present" if i == 0 else str(curr_yr)
        job_title = f"Senior {title}" if (i == 0 and years_exp > 5) else title
        resps = random.sample(template["responsibilities"], min(2, len(template["responsibilities"])))
        
        block = f"**{job_title}** | {company} ({start_yr} - {end_yr_str})\n"
        for r in resps:
            block += f"- {r}\n"
        experience_blocks.append(block)
        curr_yr = start_yr

    # Projects
    projects = random.sample(template["projects"], min(2, len(template["projects"])))
    projects_block = "\n".join([f"- **{p.split(' using ')[0] if ' using ' in p else p}**: {p}" for p in projects])

    # Formatted Full Resume Text
    resume_text = f"""{name}
{title} | {location}
Email: {email} | Phone: {phone}
LinkedIn: {linkedin} | GitHub: {github}

PROFESSIONAL SUMMARY
Results-driven {title} with {years_exp}+ years of experience in {category}. Proven track record of delivering scalable solutions, driving team success, and executing high-impact initiatives.

CORE SKILLS & TECHNOLOGIES
{", ".join(selected_skills)}

PROFESSIONAL EXPERIENCE
{"\n".join(experience_blocks)}
KEY PROJECTS
{projects_block}

EDUCATION & CERTIFICATIONS
- {degree}, {univ} (Graduated: {grad_year})
- Certified {category} Professional
"""

    return {
        "candidate_id": f"CAND_{candidate_id:04d}",
        "name": name,
        "category": category,
        "experience_years": years_exp,
        "resume_text": resume_text.strip()
    }


def generate_dataset(samples_per_category: int = 80) -> pd.DataFrame:
    """Generates complete dataset and writes to disk."""
    print(f"[*] Generating synthetic dataset with {len(CATEGORIES)} categories x {samples_per_category} samples = {len(CATEGORIES) * samples_per_category} total resumes...")
    records = []
    c_id = 1
    
    for category in CATEGORIES:
        for _ in range(samples_per_category):
            records.append(generate_synthetic_resume(category, c_id))
            c_id += 1

    df = pd.DataFrame(records)
    
    # Save CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DATA_DIR / "resumes_dataset.csv"
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved {len(df)} resume records to {out_csv}")
    return df


def generate_sample_resumes_and_jds():
    """Creates sample files (PDF, DOCX, TXT) and sample Job Descriptions."""
    SAMPLE_RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_JDS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sample Resumes in TXT format
    sample_categories = ["Data Science", "Web Development", "DevOps & Cloud", "Cyber Security", "Human Resources (HR)"]
    for i, cat in enumerate(sample_categories, 1):
        resume_data = generate_synthetic_resume(cat, 9000 + i)
        txt_path = SAMPLE_RESUMES_DIR / f"sample_resume_{cat.lower().replace(' ', '_').replace('/', '_')}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(resume_data["resume_text"])

    # 2. Sample Job Descriptions
    jds = {
        "Senior_Data_Scientist_JD.txt": """Job Title: Senior Data Scientist - AI & Analytics
Location: San Francisco, CA / Remote
Experience Required: 4+ years of professional experience

Job Overview:
We are seeking an experienced Senior Data Scientist to design, implement, and scale machine learning and NLP solutions. You will work on predictive modeling, recommendation engines, and LLM-powered applications.

Key Responsibilities:
- Build and evaluate machine learning models using Python, Scikit-Learn, PyTorch, and TensorFlow.
- Conduct A/B testing and statistical analysis to validate model performance and user impact.
- Design feature engineering pipelines and work with multi-terabyte data in SQL and Pandas.
- Collaborate with MLOps engineers to deploy containerized models via Docker and FastAPI.
- Present data visualizations and actionable insights to executive leadership using Tableau and Matplotlib.

Required Qualifications & Skills:
- 4+ years of experience in Data Science, Machine Learning, and NLP.
- Strong proficiency in Python, SQL, Scikit-Learn, Pandas, NumPy, and PyTorch.
- Experience with Deep Learning, Transformers, LLMs, and Data Visualization (Tableau, Seaborn).
- Master's degree or Ph.D. in Computer Science, Statistics, Data Science, or related technical field.
- Excellent communication and problem-solving skills.
""",
        "Full_Stack_Engineer_JD.txt": """Job Title: Senior Full Stack Engineer (React / Node.js)
Location: New York, NY / Hybrid
Experience Required: 5+ years of software development experience

About the Role:
We are looking for a Senior Full Stack Engineer to lead the architecture and development of our customer-facing web applications.

Key Responsibilities:
- Build modern, high-performance web applications using React, Next.js, and TypeScript.
- Architect scalable RESTful APIs and GraphQL services using Node.js, Express, and PostgreSQL.
- Implement responsive UI components using Tailwind CSS and maintain state with Redux.
- Optimize web performance, Core Web Vitals, and ensure accessibility standards (WCAG).
- Write comprehensive unit and integration tests using Jest and Cypress.

Required Skills:
- 5+ years of experience in Web Development and Full-Stack Engineering.
- Expertise in JavaScript, TypeScript, React, Next.js, Node.js, and HTML5/CSS3.
- Hands-on experience with PostgreSQL, MongoDB, REST API, Docker, and CI/CD.
- Bachelor's degree in Computer Science or equivalent practical experience.
""",
        "DevOps_Cloud_Architect_JD.txt": """Job Title: Lead DevOps & Cloud Infrastructure Engineer
Location: Austin, TX / Remote
Experience Required: 6+ years in DevOps / Cloud Operations

Job Description:
Join our Infrastructure team to manage large-scale cloud operations across AWS and GCP.

Responsibilities:
- Design and automate cloud infrastructure using Terraform, Ansible, and AWS CloudFormation.
- Manage production Kubernetes clusters (EKS), container orchestration, and Helm charts.
- Build resilient CI/CD pipelines with GitHub Actions, Jenkins, and Docker.
- Implement automated monitoring, alerting, and observability with Prometheus, Grafana, and ELK Stack.
- Ensure security compliance, Zero Trust network architectures, and 99.99% system availability.

Required Skills:
- 6+ years in Cloud & DevOps engineering.
- Deep expertise with AWS, Docker, Kubernetes, Terraform, Linux, CI/CD, and Python/Bash scripting.
- Strong knowledge of microservices, networking, Prometheus, and Grafana.
"""
    }

    for filename, content in jds.items():
        with open(SAMPLE_JDS_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"[+] Generated sample resumes in {SAMPLE_RESUMES_DIR}")
    print(f"[+] Generated sample Job Descriptions in {SAMPLE_JDS_DIR}")


if __name__ == "__main__":
    generate_dataset(samples_per_category=80)
    generate_sample_resumes_and_jds()
