"""
Official Executive Candidate Evaluation PDF Report Generator.
Generates publication-quality assessment scorecards using FPDF2.
"""
import io
from datetime import datetime
from typing import Dict, Any, Optional
from fpdf import FPDF


class CandidatePDF(FPDF):
    """Custom FPDF subclass with official headers and footers."""

    def header(self):
        # Top Banner
        self.set_fill_color(30, 58, 138)  # Deep Navy Blue
        self.rect(0, 0, 210, 24, "F")

        # Title
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 7, "TalentMatrix AI(TM) - Executive Candidate Assessment", 0, 1, "L")
        
        self.set_font("Helvetica", "", 9)
        self.set_text_color(203, 213, 225)
        self.set_xy(10, 14)
        self.cell(0, 5, "Official NLP Screening & Candidate Evaluation Scorecard", 0, 1, "L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"TalentMatrix AI Confidential Assessment Report | Page {self.page_no()}", 0, 0, "C")


class CandidateReportGenerator:
    """
    Constructs comprehensive candidate evaluation PDF documents.
    """

    @staticmethod
    def generate_pdf_bytes(screening_data: Dict[str, Any], job_title: str = "Target Position") -> bytes:
        """
        Builds and returns binary PDF bytes of the assessment report.
        """
        pdf = CandidatePDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        cand_name = screening_data.get("candidate_name", "Candidate")
        final_score = screening_data.get("final_score", 0.0)
        status = screening_data.get("status", "Screened")
        scores = screening_data.get("scores", {})
        skills = screening_data.get("skills", {})
        profile = screening_data.get("candidate_profile", {})

        # Section 1: Candidate Overview Card
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(10, 28, 190, 36, "DF")

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(15, 23, 42)
        pdf.set_xy(14, 32)
        pdf.cell(100, 6, f"Candidate: {cand_name}", 0, 0)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(80, 6, f"Target Role: {job_title}", 0, 1, "R")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.set_xy(14, 40)
        pdf.cell(90, 5, f"Email: {profile.get('email') or 'N/A'}", 0, 0)
        pdf.cell(90, 5, f"Location: {profile.get('location') or 'Not Specified'}", 0, 1)

        pdf.set_xy(14, 46)
        pdf.cell(90, 5, f"Experience: {profile.get('years_experience', 0)} years ({profile.get('seniority_level', 'Mid-Level')})", 0, 0)
        pdf.cell(90, 5, f"Education: {profile.get('highest_degree') or 'Degree Not Specified'}", 0, 1)

        pdf.set_xy(14, 52)
        pdf.cell(90, 5, f"Phone: {profile.get('phone') or 'N/A'}", 0, 0)
        pdf.cell(90, 5, f"Assessment Date: {datetime.now().strftime('%B %d, %Y')}", 0, 1)

        pdf.ln(12)

        # Section 2: Executive Scoring Matrix
        pdf.set_xy(10, 68)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "1. Overall Fit & Screening Scores", 0, 1)
        pdf.ln(2)

        # Score Boxes
        box_w = 44
        box_h = 24
        x_start = 10
        y_pos = 76

        scores_list = [
            ("Overall Match", f"{final_score}%", (30, 58, 138)),
            ("Skill Match", f"{scores.get('skill_match_pct', 0)}%", (16, 185, 129)),
            ("Semantic Sim", f"{scores.get('tfidf_similarity_pct', 0)}%", (59, 130, 246)),
            ("Experience Match", f"{scores.get('experience_match_pct', 0)}%", (168, 85, 247))
        ]

        for idx, (label, val, col) in enumerate(scores_list):
            cur_x = x_start + idx * (box_w + 4)
            pdf.set_fill_color(248, 250, 252)
            pdf.set_draw_color(col[0], col[1], col[2])
            pdf.rect(cur_x, y_pos, box_w, box_h, "DF")

            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(col[0], col[1], col[2])
            pdf.set_xy(cur_x, y_pos + 4)
            pdf.cell(box_w, 8, val, 0, 1, "C")

            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.set_xy(cur_x, y_pos + 13)
            pdf.cell(box_w, 6, label, 0, 1, "C")

        # Recommendation Banner
        pdf.set_xy(10, 105)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(10, 105, 190, 12, "DF")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.set_xy(14, 108)
        pdf.cell(0, 6, f"Hiring Decision Recommendation: {status}", 0, 1)

        pdf.ln(10)

        # Section 3: Skill Gap Analysis Table
        pdf.set_xy(10, 122)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "2. Deep Skill Gap Analysis", 0, 1)
        pdf.ln(2)

        matched_skills = skills.get("matched_skills", [])
        missing_skills = skills.get("missing_skills", [])
        bonus_skills = skills.get("additional_skills", [])

        # Table Headers
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(45, 7, "Category", 1, 0, "L", True)
        pdf.cell(20, 7, "Count", 1, 0, "C", True)
        pdf.cell(125, 7, "Extracted Skills", 1, 1, "L", True)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(15, 23, 42)

        # Row 1: Matched Skills
        pdf.cell(45, 12, "Matched Skills (JD)", 1, 0, "L")
        pdf.cell(20, 12, str(len(matched_skills)), 1, 0, "C")
        m_str = ", ".join(matched_skills) if matched_skills else "None"
        pdf.multi_cell(125, 6, m_str[:250], 1, "L")

        # Row 2: Missing Skills
        pdf.cell(45, 12, "Missing Required Skills", 1, 0, "L")
        pdf.cell(20, 12, str(len(missing_skills)), 1, 0, "C")
        mis_str = ", ".join(missing_skills) if missing_skills else "None"
        pdf.multi_cell(125, 6, mis_str[:250], 1, "L")

        # Row 3: Bonus Skills
        pdf.cell(45, 12, "Bonus / Additional Skills", 1, 0, "L")
        pdf.cell(20, 12, str(len(bonus_skills)), 1, 0, "C")
        b_str = ", ".join(bonus_skills[:12]) if bonus_skills else "None"
        pdf.multi_cell(125, 6, b_str[:250], 1, "L")

        pdf.ln(8)

        # Section 4: Sign-off & Recruiter Evaluation
        pdf.set_xy(10, 190)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, "3. Recruiter Evaluation & Sign-off", 0, 1)
        pdf.ln(2)

        pdf.set_draw_color(203, 213, 225)
        pdf.rect(10, 198, 190, 40)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.set_xy(14, 202)
        pdf.cell(0, 5, "Evaluator Notes / Interviewer Feedback:", 0, 1)

        # Signature Blocks
        pdf.set_xy(14, 226)
        pdf.cell(80, 5, "Evaluator Signature: _______________________", 0, 0)
        pdf.cell(80, 5, "Department Head: _______________________", 0, 1)

        pdf.set_xy(14, 232)
        pdf.cell(80, 5, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 0)
        pdf.cell(80, 5, "Status: Official Verified Record", 0, 1)

        # Output bytes
        return bytes(pdf.output())
