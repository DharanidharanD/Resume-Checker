"""
Text Preprocessing and Normalization Module for NLP Feature Extraction and Modeling.
"""
import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lazy download of required NLTK datasets
def ensure_nltk_resources():
    required = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in required:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

ensure_nltk_resources()


class TextCleaner:
    """
    Comprehensive text cleaning and normalization for resumes and job descriptions.
    """

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = {
                "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
                "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
                "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
                "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
                "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", 
                "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
                "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", 
                "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", 
                "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", 
                "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", 
                "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
                "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
                "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
                "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
                "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
                "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
                "weren't", "what", "what's", "when", "when's", "where", "where's", "which", 
                "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
                "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
                "yourself", "yourselves"
            }
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

        # Regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
        self.phone_pattern = re.compile(r"\(?\+?\d{1,3}\)?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
        self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s\+\#\.\-/_]")

    def clean_text(
        self, 
        text: str, 
        remove_stopwords: bool = True, 
        lemmatize: bool = True,
        anonymize: bool = False
    ) -> str:
        """
        Full cleaning pipeline for machine learning feature extraction.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Remove/Anonymize URLs, emails, phone numbers
        if anonymize:
            cleaned = self.url_pattern.sub(" ANONYMIZED_URL ", text)
            cleaned = self.email_pattern.sub(" ANONYMIZED_EMAIL ", cleaned)
            cleaned = self.phone_pattern.sub(" ANONYMIZED_PHONE ", cleaned)
        else:
            cleaned = self.url_pattern.sub(" ", text)
            cleaned = self.email_pattern.sub(" ", cleaned)
            cleaned = self.phone_pattern.sub(" ", cleaned)

        # Step 2: Protect special keywords
        cleaned = self._preserve_tech_keywords(cleaned)
        cleaned = self.special_chars_pattern.sub(" ", cleaned)
        cleaned = self._restore_tech_keywords(cleaned)

        # Step 3: Lowercase
        cleaned = cleaned.lower()

        # Step 4: Tokenize
        tokens = cleaned.split()

        # Step 5: Filter Stopwords & Short/Long tokens
        filtered_tokens = []
        for token in tokens:
            token = token.strip(string.punctuation)
            if not token:
                continue
            if len(token) < 2 or len(token) > 35:
                continue
            if remove_stopwords and token in self.stop_words and not token.startswith("anonymized_"):
                continue
            
            # Step 6: Lemmatize
            if lemmatize and self.lemmatizer and not token.startswith("anonymized_"):
                try:
                    token = self.lemmatizer.lemmatize(token)
                except Exception:
                    pass
            filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def _preserve_tech_keywords(self, text: str) -> str:
        """Protects tokens with symbols like C++, C#, .NET from being stripped."""
        text = re.sub(r"\bC\+\+", "PROTECTED_CPP", text, flags=re.IGNORECASE)
        text = re.sub(r"\bC\#", "PROTECTED_CSHARP", text, flags=re.IGNORECASE)
        text = re.sub(r"\.NET\b", "PROTECTED_DOTNET", text, flags=re.IGNORECASE)
        text = re.sub(r"\bNode\.js\b", "PROTECTED_NODEJS", text, flags=re.IGNORECASE)
        text = re.sub(r"\bVue\.js\b", "PROTECTED_VUEJS", text, flags=re.IGNORECASE)
        text = re.sub(r"\bReact\.js\b", "PROTECTED_REACTJS", text, flags=re.IGNORECASE)
        return text

    def _restore_tech_keywords(self, text: str) -> str:
        """Restores protected tech keywords back to clean alphanumeric names."""
        text = text.replace("PROTECTED_CPP", "cplusplus")
        text = text.replace("PROTECTED_CSHARP", "csharp")
        text = text.replace("PROTECTED_DOTNET", "dotnet")
        text = text.replace("PROTECTED_NODEJS", "nodejs")
        text = text.replace("PROTECTED_VUEJS", "vuejs")
        text = text.replace("PROTECTED_REACTJS", "reactjs")
        return text

    def get_tokens(self, text: str) -> List[str]:
        """Returns cleaned tokens list."""
        cleaned = self.clean_text(text)
        return cleaned.split()
