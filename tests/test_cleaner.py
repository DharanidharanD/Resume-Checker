"""
Unit tests for TextCleaner.
"""
import pytest
from src.preprocessing.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_clean_text_basic(cleaner):
    raw = "Hello World! This is a simple test resume for Python developer."
    cleaned = cleaner.clean_text(raw)
    assert "python" in cleaned
    assert "developer" in cleaned
    assert "this" not in cleaned  # stopword removed


def test_preserve_tech_keywords(cleaner):
    raw = "Expert in C++, C#, .NET, Node.js, and React.js."
    cleaned = cleaner.clean_text(raw)
    assert "cplusplus" in cleaned
    assert "csharp" in cleaned
    assert "dotnet" in cleaned
    assert "nodejs" in cleaned
    assert "reactjs" in cleaned


def test_url_and_email_removal(cleaner):
    raw = "Contact me at john.doe@email.com or visit https://myportfolio.com"
    cleaned = cleaner.clean_text(raw, anonymize=False)
    assert "@" not in cleaned
    assert "http" not in cleaned


def test_anonymization(cleaner):
    raw = "Email: jane.smith@domain.org Phone: +1 555-123-4567 Site: https://site.org"
    cleaned = cleaner.clean_text(raw, anonymize=True)
    assert "anonymized_email" in cleaned
    assert "anonymized_phone" in cleaned
    assert "anonymized_url" in cleaned


def test_empty_or_none_input(cleaner):
    assert cleaner.clean_text("") == ""
    assert cleaner.clean_text(None) == ""
