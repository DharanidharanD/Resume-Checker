"""
Command-Line Interface (CLI) for Resume Screening and Candidate Classification System.
"""
import sys
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.document_parser import DocumentParser
from src.extractors.skill_extractor import SkillExtractor
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.experience_extractor import ExperienceExtractor
from src.models.classifier import ResumeClassifier
from src.screening.matcher import ResumeScreeningMatcher
from scripts.train_pipeline import run_pipeline

console = Console()


def parse_cmd(args):
    """Parses a resume file and prints structured information."""
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        return

    console.print(f"[cyan]Reading {file_path.name}...[/cyan]")
    text = DocumentParser.extract_text(file_path)

    contacts = ContactExtractor.extract_contacts(text)
    skills = SkillExtractor().extract_skills(text)
    exp = ExperienceExtractor.extract_experience_and_education(text)

    # Contact Panel
    contact_text = f"[bold]Name:[/bold] {contacts['name'] or 'N/A'}\n" \
                   f"[bold]Email:[/bold] {contacts['email'] or 'N/A'}\n" \
                   f"[bold]Phone:[/bold] {contacts['phone'] or 'N/A'}\n" \
                   f"[bold]LinkedIn:[/bold] {contacts['linkedin'] or 'N/A'}\n" \
                   f"[bold]GitHub:[/bold] {contacts['github'] or 'N/A'}\n" \
                   f"[bold]Location:[/bold] {contacts['location'] or 'N/A'}"
    console.print(Panel(contact_text, title="Candidate Profile & Contacts", border_style="green"))

    # Education & Exp Panel
    exp_text = f"[bold]Highest Degree:[/bold] {exp['highest_degree']}\n" \
               f"[bold]Estimated Experience:[/bold] {exp['estimated_years_experience']} years\n" \
               f"[bold]Seniority Level:[/bold] {exp['seniority_level']}"
    console.print(Panel(exp_text, title="Experience & Education", border_style="blue"))

    # Skills Table
    table = Table(title=f"Extracted Skills ({skills['skill_count']} Total)")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Skills", style="white")

    for cat, sk_list in skills["by_category"].items():
        table.add_row(cat, str(len(sk_list)), ", ".join(sk_list))

    console.print(table)


def classify_cmd(args):
    """Predicts candidate job domain for a given resume."""
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        return

    text = DocumentParser.extract_text(file_path)
    classifier = ResumeClassifier()
    if not classifier.is_ready:
        console.print("[bold red]Model not trained.[/bold red] Please run `python -m src.cli train` first.")
        return

    res = classifier.predict(text, top_k=args.top_k)

    console.print(Panel(
        f"[bold yellow]Predicted Domain:[/bold yellow] [bold green]{res['predicted_category']}[/bold green]\n"
        f"[bold yellow]Confidence:[/bold yellow] {res['confidence'] * 100:.2f}%",
        title="Candidate Classification Result",
        border_style="magenta"
    ))

    table = Table(title="Top Predictions Breakdown")
    table.add_column("Rank", justify="center")
    table.add_column("Domain Category", style="cyan")
    table.add_column("Probability", justify="right", style="green")

    for idx, pred in enumerate(res["top_k_predictions"], start=1):
        pct = f"{pred['probability'] * 100:.2f}%"
        table.add_row(str(idx), pred["category"], pct)

    console.print(table)


def screen_cmd(args):
    """Screens a resume against a Job Description."""
    resume_path = Path(args.resume)
    jd_path = Path(args.jd)

    if not resume_path.exists():
        console.print(f"[bold red]Error:[/bold red] Resume file not found: {resume_path}")
        return
    if not jd_path.exists():
        console.print(f"[bold red]Error:[/bold red] JD file not found: {jd_path}")
        return

    resume_text = DocumentParser.extract_text(resume_path)
    jd_text = DocumentParser.extract_text(jd_path)

    matcher = ResumeScreeningMatcher(
        skill_weight=args.skill_weight,
        tfidf_weight=args.tfidf_weight,
        experience_weight=args.exp_weight
    )

    result = matcher.screen_single(resume_text, jd_text, candidate_identifier=resume_path.name)

    console.print(Panel(
        f"[bold]Candidate Name:[/bold] {result['candidate_name']}\n"
        f"[bold]Overall Match Score:[/bold] [bold {result['badge_color']}]{result['final_score']}%[/bold {result['badge_color']}]\n"
        f"[bold]Status Recommendation:[/bold] {result['status']}\n\n"
        f"  • Skill Match: {result['scores']['skill_match_pct']}%\n"
        f"  • TF-IDF Semantic Similarity: {result['scores']['tfidf_similarity_pct']}%\n"
        f"  • Experience Alignment: {result['scores']['experience_match_pct']}%",
        title="Candidate Screening Report",
        border_style=result["badge_color"]
    ))

    # Skill Gap Breakdown
    gap_table = Table(title="Skill Gap Diagnostics")
    gap_table.add_column("Type", style="bold")
    gap_table.add_column("Count", justify="right")
    gap_table.add_column("Skills List")

    matched = result["skills"]["matched_skills"]
    missing = result["skills"]["missing_skills"]
    additional = result["skills"]["additional_skills"]

    gap_table.add_row("[green]Matched Skills[/green]", str(len(matched)), ", ".join(matched) if matched else "None")
    gap_table.add_row("[red]Missing JD Skills[/red]", str(len(missing)), ", ".join(missing) if missing else "None")
    gap_table.add_row("[cyan]Bonus / Other Skills[/cyan]", str(len(additional)), ", ".join(additional[:10]) + ("..." if len(additional) > 10 else ""))

    console.print(gap_table)


def train_cmd(args):
    """Runs the training pipeline."""
    run_pipeline(samples_per_category=args.samples)


def main():
    parser = argparse.ArgumentParser(description="Resume Screening and Classification CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: parse
    p_parse = subparsers.add_parser("parse", help="Parse and extract profile/skills from a resume")
    p_parse.add_argument("file", help="Path to resume file (PDF/DOCX/TXT)")

    # Subcommand: classify
    p_classify = subparsers.add_parser("classify", help="Classify candidate resume into job domain")
    p_classify.add_argument("file", help="Path to resume file")
    p_classify.add_argument("--top-k", type=int, default=5, help="Number of top categories to display")

    # Subcommand: screen
    p_screen = subparsers.add_parser("screen", help="Screen resume against a Job Description")
    p_screen.add_argument("--resume", required=True, help="Path to candidate resume")
    p_screen.add_argument("--jd", required=True, help="Path to Job Description")
    p_screen.add_argument("--skill-weight", type=float, default=0.50, help="Weight for skill match (0-1)")
    p_screen.add_argument("--tfidf-weight", type=float, default=0.30, help="Weight for TF-IDF similarity (0-1)")
    p_screen.add_argument("--exp-weight", type=float, default=0.20, help="Weight for experience match (0-1)")

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train and evaluate classification models")
    p_train.add_argument("--samples", type=int, default=80, help="Number of samples per category")

    args = parser.parse_args()

    if args.command == "parse":
        parse_cmd(args)
    elif args.command == "classify":
        classify_cmd(args)
    elif args.command == "screen":
        screen_cmd(args)
    elif args.command == "train":
        train_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
