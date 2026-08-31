"""
TalentMatrix AI(TM) - Official Enterprise Resume Screening & Candidate Classification System.
"""
import sys
import io
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.document_parser import DocumentParser
from src.preprocessing.text_cleaner import TextCleaner
from src.extractors.skill_extractor import SkillExtractor
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.experience_extractor import ExperienceExtractor
from src.models.classifier import ResumeClassifier
from src.models.trainer import ModelTrainer
from src.screening.matcher import ResumeScreeningMatcher
from src.reports.pdf_generator import CandidateReportGenerator
from src.database.connection import get_db_session, init_db, seed_initial_data_if_empty
from src.database.models import JobPosting, Candidate, ScreeningRecord
from src.config import CATEGORIES, METRICS_REPORT_PATH, SAMPLE_RESUMES_DIR, SAMPLE_JDS_DIR
from scripts.generate_synthetic_data import generate_dataset, generate_sample_resumes_and_jds

# Initialize Database
init_db()
seed_initial_data_if_empty()

# Page Configuration
st.set_page_config(
    page_title="TalentMatrix AI - Enterprise ATS & Resume Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise CSS Styling
st.markdown("""
<style>
    .enterprise-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .enterprise-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .skill-badge {
        display: inline-block;
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-badge-matched {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-badge {
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# Load NLP Core Components
@st.cache_resource
def load_nlp_components():
    return {
        "text_cleaner": TextCleaner(),
        "skill_extractor": SkillExtractor(),
        "contact_extractor": ContactExtractor(),
        "experience_extractor": ExperienceExtractor(),
        "matcher": ResumeScreeningMatcher(),
    }

nlp = load_nlp_components()


def get_classifier():
    classifier = ResumeClassifier()
    return classifier


# Sidebar Header
st.sidebar.markdown("## 🏛️ **TalentMatrix AI™**")
st.sidebar.caption("Enterprise ATS & Candidate Intelligence System\n*Final Year Capstone Project*")

# Navigation Menu
menu = st.sidebar.radio(
    "Application Modules:",
    [
        "📋 Requisition & Job Postings",
        "🎯 AI Screening & Gap Analysis",
        "👥 ATS Pipeline & Kanban Board",
        "🏷️ Candidate Domain Classifier",
        "📈 Executive Analytics & Reports",
        "🛠️ Model Benchmarking Studio"
    ]
)

st.sidebar.markdown("---")

# Global Compliance Setting
st.sidebar.markdown("### 🛡️ Compliance & Ethics")
bias_free_mode = st.sidebar.toggle("Blind Screening Mode (Anonymize PII)", value=False, help="Redacts Candidate Name, Email, and Phone for bias-free recruitment.")

st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Online** | DB: **Connected**")


# ==============================================================================
# MODULE 1: Requisition & Job Postings (Database CRUD)
# ==============================================================================
if menu == "📋 Requisition & Job Postings":
    st.markdown('<div class="enterprise-header">Job Requisition Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="enterprise-subtitle">Create and manage organizational job postings with custom screening weight configurations.</div>', unsafe_allow_html=True)

    session = get_db_session()
    jobs = session.query(JobPosting).filter(JobPosting.is_active == True).all()

    tab1, tab2 = st.tabs(["Active Job Requisitions", "➕ Create New Job Requisition"])

    with tab1:
        if not jobs:
            st.info("No active job postings found. Create one in the next tab.")
        else:
            for j in jobs:
                with st.expander(f"📌 **{j.title}** ({j.department} | {j.location})", expanded=False):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown(f"**Overview:**\n{j.description}")
                        st.markdown(f"**Required Experience:** `{j.min_experience_years}+ years`")
                    with c2:
                        st.markdown("**Required Skills:**")
                        badges = " ".join([f'<span class="skill-badge">{s}</span>' for s in j.required_skills])
                        st.markdown(badges or "_None specified_", unsafe_allow_html=True)
                    with c3:
                        st.markdown("**Screening Weights:**")
                        st.write(f"- Skill Match: `{int(j.skill_weight*100)}%`")
                        st.write(f"- TF-IDF Semantic: `{int(j.tfidf_weight*100)}%`")
                        st.write(f"- Experience: `{int(j.exp_weight*100)}%`")
                        st.caption(f"Created: {j.created_at.strftime('%Y-%m-%d')}")

    with tab2:
        with st.form("create_job_form"):
            st.markdown("#### New Job Requisition Details")
            j_title = st.text_input("Job Title*", placeholder="e.g. Senior Machine Learning Engineer")
            col_d, col_l = st.columns(2)
            with col_d:
                j_dept = st.selectbox("Department", ["Engineering", "Data Science", "Cloud & Infrastructure", "Product", "Cyber Security", "Operations", "Human Resources", "Finance"])
            with col_l:
                j_loc = st.text_input("Location", value="Remote / Hybrid")

            j_desc = st.text_area("Job Description & Responsibilities*", height=120, placeholder="Describe role responsibilities, required tech stack, and qualifications...")
            j_skills_input = st.text_input("Required Skills (Comma separated)*", placeholder="python, machine learning, pytorch, sql, docker")
            j_min_exp = st.slider("Minimum Years of Experience", 0.0, 15.0, 4.0, 0.5)

            st.markdown("##### Custom Screening Weights")
            cw1, cw2, cw3 = st.columns(3)
            with cw1:
                sw = st.slider("Skill Match Weight", 0.0, 1.0, 0.50, 0.05)
            with cw2:
                tw = st.slider("TF-IDF Semantic Weight", 0.0, 1.0, 0.30, 0.05)
            with cw3:
                ew = st.slider("Experience Alignment Weight", 0.0, 1.0, 0.20, 0.05)

            submitted = st.form_submit_button("💼 Post Job Requisition", type="primary")

            if submitted:
                if not j_title.strip() or not j_desc.strip():
                    st.error("Please fill in Job Title and Description.")
                else:
                    skills_list = [s.strip().lower() for s in j_skills_input.split(",") if s.strip()]
                    new_job = JobPosting(
                        title=j_title.strip(),
                        department=j_dept,
                        location=j_loc.strip(),
                        description=j_desc.strip(),
                        min_experience_years=j_min_exp,
                        skill_weight=sw,
                        tfidf_weight=tw,
                        exp_weight=ew
                    )
                    new_job.required_skills = skills_list
                    session.add(new_job)
                    session.commit()
                    st.success(f"Job Posting **'{j_title}'** successfully created in database!")
                    st.rerun()

    session.close()


# ==============================================================================
# MODULE 2: AI Screening & Skill Gap Analysis
# ==============================================================================
elif menu == "🎯 AI Screening & Gap Analysis":
    st.markdown('<div class="enterprise-header">AI-Powered Resume Screening & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="enterprise-subtitle">Screen resumes against open job requisitions with multi-factor matching, gap diagnosis, and PDF export.</div>', unsafe_allow_html=True)

    session = get_db_session()
    jobs = session.query(JobPosting).filter(JobPosting.is_active == True).all()

    if not jobs:
        st.warning("No job requisitions found. Please create one in the Requisition Manager tab.")
    else:
        # Job Selector
        job_options = {f"{j.id}: {j.title} ({j.department})": j for j in jobs}
        selected_job_str = st.selectbox("Select Target Job Requisition:", list(job_options.keys()))
        target_job = job_options[selected_job_str]

        st.info(f"Target Role: **{target_job.title}** | Min Exp: **{target_job.min_experience_years} yrs** | Required Skills: `{', '.join(target_job.required_skills[:8])}`")

        screen_mode = st.radio("Screening Ingestion Mode:", ["📄 Single Resume Deep Evaluation", "📁 Multi-Resume Batch Ranker"], horizontal=True)

        if screen_mode == "📄 Single Resume Deep Evaluation":
            col1, col2 = st.columns([1, 1])
            with col1:
                res_file = st.file_uploader("Upload Resume File (.pdf, .docx, .txt):", type=["pdf", "docx", "txt"])
            with col2:
                sample_files = list(SAMPLE_RESUMES_DIR.glob("*.txt")) if SAMPLE_RESUMES_DIR.exists() else []
                sample_choice = st.selectbox("Or choose a sample resume:", ["-- Select Sample --"] + [f.name for f in sample_files])

            resume_text = ""
            filename = "Uploaded_Resume"
            if res_file:
                filename = res_file.name
                resume_text = DocumentParser.extract_text(res_file.read(), filename=filename)
            elif sample_choice != "-- Select Sample --":
                filename = sample_choice
                resume_text = DocumentParser.extract_text(SAMPLE_RESUMES_DIR / sample_choice)

            if resume_text:
                matcher = ResumeScreeningMatcher(
                    skill_weight=target_job.skill_weight,
                    tfidf_weight=target_job.tfidf_weight,
                    experience_weight=target_job.exp_weight
                )

                # Screen
                result = matcher.screen_single(resume_text, target_job.description + " " + " ".join(target_job.required_skills), candidate_identifier=filename)

                # Anonymize if blind screening mode is on
                display_name = "ANONYMOUS_CANDIDATE" if bias_free_mode else result["candidate_name"]

                st.markdown("---")
                st.markdown(f"### 🏆 Screening Assessment: **{display_name}**")

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Overall Match Score", f"{result['final_score']}%")
                with m2:
                    st.metric("Skill Match", f"{result['scores']['skill_match_pct']}%")
                with m3:
                    st.metric("Semantic Similarity", f"{result['scores']['tfidf_similarity_pct']}%")
                with m4:
                    st.metric("Experience Match", f"{result['scores']['experience_match_pct']}%")

                # Decision Recommendation
                if result['final_score'] >= 80:
                    st.success(f"**Recommendation:** {result['status']}")
                elif result['final_score'] >= 60:
                    st.warning(f"**Recommendation:** {result['status']}")
                else:
                    st.error(f"**Recommendation:** {result['status']}")

                # Skill Gap Matrix
                st.markdown("### 🔍 Deep Skill Gap Diagnostics")
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.markdown(f"**✅ Matched Skills ({len(result['skills']['matched_skills'])})**")
                    matched_html = " ".join([f'<span class="skill-badge-matched">{s}</span>' for s in result['skills']['matched_skills']])
                    st.markdown(matched_html or "_None_", unsafe_allow_html=True)
                with g2:
                    st.markdown(f"**❌ Missing Required Skills ({len(result['skills']['missing_skills'])})**")
                    missing_html = " ".join([f'<span class="skill-badge-missing">{s}</span>' for s in result['skills']['missing_skills']])
                    st.markdown(missing_html or "_None_", unsafe_allow_html=True)
                with g3:
                    st.markdown(f"**✨ Bonus Skills ({len(result['skills']['additional_skills'])})**")
                    add_html = " ".join([f'<span class="skill-badge">{s}</span>' for s in result['skills']['additional_skills'][:10]])
                    st.markdown(add_html or "_None_", unsafe_allow_html=True)

                st.markdown("---")

                # Actions: Save to Database & Download Official PDF Report
                act1, act2 = st.columns([1, 1])
                with act1:
                    if st.button("💾 Save Candidate to ATS Database", type="primary"):
                        cand = Candidate(
                            name=result["candidate_name"],
                            email=result["candidate_profile"]["email"],
                            phone=result["candidate_profile"]["phone"],
                            linkedin=result["candidate_profile"]["linkedin"],
                            github=result["candidate_profile"]["github"],
                            location=result["candidate_profile"]["location"],
                            highest_degree=result["candidate_profile"]["highest_degree"],
                            years_experience=result["candidate_profile"]["years_experience"],
                            seniority_level=result["candidate_profile"]["seniority_level"],
                            raw_text=resume_text,
                            resume_filename=filename
                        )
                        cand.skills = result["skills"]["matched_skills"] + result["skills"]["additional_skills"]
                        session.add(cand)
                        session.flush()

                        rec = ScreeningRecord(
                            candidate_id=cand.id,
                            job_id=target_job.id,
                            overall_score=result["final_score"],
                            skill_score=result["scores"]["skill_match_pct"],
                            tfidf_score=result["scores"]["tfidf_similarity_pct"],
                            exp_score=result["scores"]["experience_match_pct"],
                            status="Shortlisted" if result["final_score"] >= 65 else "Screened",
                            recommendation=result["status"],
                            recruiter_notes=f"Auto-evaluated on {datetime.now().strftime('%Y-%m-%d')}"
                        )
                        rec.matched_skills = result["skills"]["matched_skills"]
                        rec.missing_skills = result["skills"]["missing_skills"]
                        session.add(rec)
                        session.commit()
                        st.success(f"Candidate **{result['candidate_name']}** saved to ATS Pipeline!")

                with act2:
                    # Generate Official PDF Scorecard
                    pdf_bytes = CandidateReportGenerator.generate_pdf_bytes(result, job_title=target_job.title)
                    st.download_button(
                        label="📄 Download Official Assessment Scorecard (PDF)",
                        data=pdf_bytes,
                        file_name=f"Assessment_{result['candidate_name'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

        else:
            # Batch Ranker Mode
            st.markdown("#### Batch Resume Upload & Ranking")
            batch_files = st.file_uploader("Upload Multiple Resumes:", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="ats_batch_upload")
            use_samples = st.checkbox("Include bundled sample candidates", value=len(batch_files) == 0)

            if st.button("⚡ Execute Batch Screening & Leaderboard", type="primary"):
                resumes_to_screen = []
                if batch_files:
                    for bf in batch_files:
                        text = DocumentParser.extract_text(bf.read(), filename=bf.name)
                        resumes_to_screen.append((bf.name, text))
                elif use_samples and SAMPLE_RESUMES_DIR.exists():
                    for sf in SAMPLE_RESUMES_DIR.glob("*.txt"):
                        text = DocumentParser.extract_text(sf)
                        resumes_to_screen.append((sf.name, text))

                if not resumes_to_screen:
                    st.error("Please upload resumes or enable sample resumes.")
                else:
                    matcher = ResumeScreeningMatcher(
                        skill_weight=target_job.skill_weight,
                        tfidf_weight=target_job.tfidf_weight,
                        experience_weight=target_job.exp_weight
                    )
                    batch_res = matcher.batch_screen(resumes_to_screen, target_job.description + " " + " ".join(target_job.required_skills))
                    df = batch_res["summary_df"]

                    if bias_free_mode:
                        df["Candidate Name"] = [f"Candidate #{i+1}" for i in range(len(df))]

                    st.success(f"Screened and ranked **{len(resumes_to_screen)}** candidates!")

                    # Leaderboard Table
                    st.dataframe(
                        df.style.background_gradient(subset=["Overall Match (%)"], cmap="Blues"),
                        use_container_width=True
                    )

                    # Export Leaderboard CSV
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Candidate Ranking Leaderboard (CSV)",
                        data=csv_data,
                        file_name=f"Ranking_{target_job.title.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

    session.close()


# ==============================================================================
# MODULE 3: ATS Pipeline & Kanban Board
# ==============================================================================
elif menu == "👥 ATS Pipeline & Kanban Board":
    st.markdown('<div class="enterprise-header">Applicant Tracking System (ATS) Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="enterprise-subtitle">Manage candidate hiring workflow across stages with live database status updates.</div>', unsafe_allow_html=True)

    session = get_db_session()
    records = session.query(ScreeningRecord).join(Candidate).join(JobPosting).all()

    stages = ["Screened", "Shortlisted", "Interviewing", "Offered", "Rejected"]

    # Filter by job
    job_titles = list(set([r.job.title for r in records])) if records else []
    selected_filter_job = st.selectbox("Filter Pipeline by Job Posting:", ["-- All Requisitions --"] + job_titles)

    filtered_records = records
    if selected_filter_job != "-- All Requisitions --":
        filtered_records = [r for r in records if r.job.title == selected_filter_job]

    # Kanban Columns
    cols = st.columns(len(stages))

    for idx, stage in enumerate(stages):
        with cols[idx]:
            stage_records = [r for r in filtered_records if r.status == stage]
            st.markdown(f"#### {stage} ({len(stage_records)})")
            st.markdown("---")

            for r in stage_records:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>{r.candidate.name if not bias_free_mode else 'Candidate #' + str(r.candidate.id)}</b><br>
                        <small>Role: {r.job.title[:25]}...</small><br>
                        <b>Match: {r.overall_score}%</b> | Exp: {r.candidate.years_experience}y
                    </div>
                    """, unsafe_allow_html=True)

                    # Action to move stage
                    new_stage = st.selectbox(
                        "Move to Stage:", 
                        stages, 
                        index=stages.index(r.status), 
                        key=f"stage_select_{r.id}"
                    )
                    if new_stage != r.status:
                        r.status = new_stage
                        session.commit()
                        st.success(f"Moved to {new_stage}")
                        st.rerun()

                    # PDF Download per candidate
                    res_dict = {
                        "candidate_name": r.candidate.name,
                        "final_score": r.overall_score,
                        "status": r.recommendation,
                        "scores": {
                            "skill_match_pct": r.skill_score,
                            "tfidf_similarity_pct": r.tfidf_score,
                            "experience_match_pct": r.exp_score
                        },
                        "skills": {
                            "matched_skills": r.matched_skills,
                            "missing_skills": r.missing_skills,
                            "additional_skills": []
                        },
                        "candidate_profile": {
                            "name": r.candidate.name,
                            "email": r.candidate.email,
                            "phone": r.candidate.phone,
                            "location": r.candidate.location,
                            "highest_degree": r.candidate.highest_degree,
                            "years_experience": r.candidate.years_experience,
                            "seniority_level": r.candidate.seniority_level
                        }
                    }
                    pdf_bytes = CandidateReportGenerator.generate_pdf_bytes(res_dict, job_title=r.job.title)
                    st.download_button(
                        label="📄 PDF",
                        data=pdf_bytes,
                        file_name=f"{r.candidate.name}_Scorecard.pdf",
                        mime="application/pdf",
                        key=f"pdf_btn_{r.id}"
                    )
                    st.write("")

    session.close()


# ==============================================================================
# MODULE 4: Candidate Domain Classifier
# ==============================================================================
elif menu == "🏷️ Candidate Domain Classifier":
    st.markdown('<div class="enterprise-header">Multi-Class Domain Classification Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="enterprise-subtitle">Predict candidate professional domain across 12 industry categories with calibrated probabilities.</div>', unsafe_allow_html=True)

    clf = get_classifier()

    if not clf.is_ready:
        st.warning("⚠️ Classification models not trained yet. Go to Model Benchmarking Studio to train.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            clf_file = st.file_uploader("Upload Resume:", type=["pdf", "docx", "txt"], key="clf_file_tab")
            sample_files = list(SAMPLE_RESUMES_DIR.glob("*.txt")) if SAMPLE_RESUMES_DIR.exists() else []
            sample_sel = st.selectbox("Or choose sample resume:", ["-- Select --"] + [f.name for f in sample_files], key="clf_sample_tab")

            clf_text = ""
            if clf_file:
                clf_text = DocumentParser.extract_text(clf_file.read(), filename=clf_file.name)
            elif sample_sel != "-- Select --":
                clf_text = DocumentParser.extract_text(SAMPLE_RESUMES_DIR / sample_sel)
            else:
                clf_text = st.text_area("Or paste Resume text:", height=150)

        with col2:
            if clf_text.strip():
                preds = clf.predict(clf_text, top_k=5)
                st.markdown("### 🎯 Classification Prediction")
                st.success(f"**Predicted Domain:** {preds['predicted_category']} ({preds['confidence'] * 100:.1f}% confidence)")

                top_cats = [p["category"] for p in preds["top_k_predictions"]]
                top_probs = [p["probability"] * 100.0 for p in preds["top_k_predictions"]]

                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.barplot(x=top_probs, y=top_cats, palette="viridis", ax=ax)
                ax.set_xlabel("Probability (%)")
                ax.set_title("Top 5 Predicted Categories")
                st.pyplot(fig)


# ==============================================================================
# MODULE 5: Executive Analytics & Reports
# ==============================================================================
elif menu == "📈 Executive Analytics & Reports":
    st.markdown('<div class="enterprise-header">Executive Analytics & Talent Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="enterprise-subtitle">Macro-level recruitment metrics, hiring pipeline velocity, and candidate score distributions.</div>', unsafe_allow_html=True)

    session = get_db_session()
    records = session.query(ScreeningRecord).join(Candidate).join(JobPosting).all()

    if not records:
        st.info("No candidate screening records found yet in database. Perform screenings to populate analytics.")
    else:
        # High Level KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Screened Candidates", len(records))
        with k2:
            shortlisted_cnt = len([r for r in records if r.status in ["Shortlisted", "Interviewing", "Offered"]])
            st.metric("Shortlisted / Interviewing", shortlisted_cnt)
        with k3:
            avg_score = np.mean([r.overall_score for r in records])
            st.metric("Average Match Score", f"{avg_score:.1f}%")
        with k4:
            total_jobs = session.query(JobPosting).count()
            st.metric("Active Job Postings", total_jobs)

        st.markdown("---")

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("##### Candidate Score Distribution")
            scores = [r.overall_score for r in records]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(scores, bins=10, kde=True, color="#3B82F6", ax=ax)
            ax.set_xlabel("Overall Match Score (%)")
            ax.set_ylabel("Candidate Count")
            st.pyplot(fig)

        with c_chart2:
            st.markdown("##### Hiring Pipeline Stage Funnel")
            stages_count = {}
            for r in records:
                stages_count[r.status] = stages_count.get(r.status, 0) + 1

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=list(stages_count.values()), y=list(stages_count.keys()), palette="crest", ax=ax)
            ax.set_xlabel("Candidates Count")
            st.pyplot(fig)

    session.close()


# ==============================================================================
# MODULE 6: Model Benchmarking Studio
# ==============================================================================
elif menu == "🛠️ Model Benchmarking Studio":
    st.markdown('<div class="enterprise-header">Machine Learning Model Benchmarking Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="enterprise-subtitle">Train, cross-validate, and benchmark candidate classifiers with confusion matrix diagnostics.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Dataset & Training Controls")
        samples_per_cat = st.slider("Samples per category:", min_value=30, max_value=120, value=70, step=10)
        
        if st.button("🚀 Train & Benchmark Models", type="primary"):
            with st.spinner("Generating dataset, extracting features, and benchmarking classifiers across 5-fold CV..."):
                trainer = ModelTrainer(random_state=42)
                df = generate_dataset(samples_per_category=samples_per_cat)
                generate_sample_resumes_and_jds()
                results = trainer.train_and_evaluate(df["resume_text"].tolist(), df["category"].tolist())
                st.success(f"Training complete! Best Model: **{results['best_model_name']}** (F1: {results['best_f1']*100:.2f}%)")

    with c2:
        if METRICS_REPORT_PATH.exists():
            with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            st.markdown("#### Winning Model Performance")
            st.metric("Best Model", metrics_data.get("best_model_name", "N/A"))
            m = metrics_data.get("metrics", {})
            k1, k2, k3 = st.columns(3)
            k1.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
            k2.metric("Weighted F1", f"{m.get('f1_weighted', 0)*100:.1f}%")
            k3.metric("Macro F1", f"{m.get('f1_macro', 0)*100:.1f}%")

            if "comparison_results" in metrics_data:
                st.markdown("##### Classifier Comparison Leaderboard")
                comp_df = pd.DataFrame(metrics_data["comparison_results"])
                display_cols = ["model_name", "accuracy", "f1_weighted", "cv_f1_mean", "train_time_sec"]
                st.dataframe(comp_df[display_cols], use_container_width=True)

            if "confusion_matrix" in m and "target_names" in m:
                with st.expander("📊 Confusion Matrix"):
                    cm = np.array(m["confusion_matrix"])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                                xticklabels=m["target_names"], yticklabels=m["target_names"], ax=ax)
                    plt.xticks(rotation=45, ha="right", fontsize=8)
                    plt.yticks(fontsize=8)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)
