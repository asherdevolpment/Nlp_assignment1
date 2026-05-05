"""
Dataset B preprocessing comparison.

This script compares the local English Amazon reviews with a Masakhane African
language sample. It uses Yoruba from MasakhaNER by default because Yoruba text
contains useful spelling/diacritic variation for this assignment.

Run from the project root:
    .venv\\Scripts\\python.exe dataset_b\\african_language_preprocessing.py
"""

from __future__ import annotations

import html
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dataset_b" / "outputs"
DATA_DIR = ROOT / "dataset_b" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_CODE = "yor"
LANGUAGE_NAME = "Yoruba"
MAX_DOCS = 5000

MASAKHANER_BASE = "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data"
MASAKHANER_FILES = ["train.txt", "dev.txt", "test.txt"]

ENGLISH_SENTIMENT_WORDS = {
    "not", "no", "never", "cannot", "can't", "dont", "don't", "didnt", "didn't",
    "good", "great", "excellent", "amazing", "bad", "awful", "terrible", "poor",
    "love", "loved", "hate", "hated", "best", "worst", "disappointed",
}

# Small Yoruba polarity cue list for sentiment-clarity proxy only.
# This is intentionally small and discussed as a limitation in the report.
YORUBA_SENTIMENT_WORDS = {
    "dara", "buru", "rere", "ko", "kii", "kò", "kìí", "fẹ", "fẹran", "korira",
    "inu", "dun", "binu", "pupọ", "gan",
}

AFRICAN_LETTER_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
REPEATED_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)
PUNCT_RE = re.compile(r"[^\w\s!?'\u00C0-\u024F\u1E00-\u1EFF-]", flags=re.UNICODE)


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def ensure_masakhane_data() -> list[Path]:
    paths = []
    for filename in MASAKHANER_FILES:
        url = f"{MASAKHANER_BASE}/{LANGUAGE_CODE}/{filename}"
        path = DATA_DIR / f"{LANGUAGE_CODE}_{filename}"
        download_file(url, path)
        paths.append(path)
    return paths


def read_masakhaner_sentences(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        split = path.name.split("_", 1)[1].replace(".txt", "")
        tokens: list[str] = []
        tags: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                if tokens:
                    rows.append({
                        "language": LANGUAGE_NAME,
                        "split": split,
                        "Text": " ".join(tokens),
                        "ner_tags": " ".join(tags),
                    })
                    tokens = []
                    tags = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
        if tokens:
            rows.append({
                "language": LANGUAGE_NAME,
                "split": split,
                "Text": " ".join(tokens),
                "ner_tags": " ".join(tags),
            })
    return pd.DataFrame(rows)


def load_english_reviews() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "Reviews.csv", usecols=["Score", "Text"]).dropna()
    df["language"] = "English"
    df["sentiment"] = df["Score"].map(
        lambda score: "positive" if score >= 4 else "negative" if score <= 2 else "neutral"
    )
    return df.sample(min(MAX_DOCS, len(df)), random_state=42).reset_index(drop=True)


def load_african_text() -> pd.DataFrame:
    paths = ensure_masakhane_data()
    df = read_masakhaner_sentences(paths)
    return df.sample(min(MAX_DOCS, len(df)), random_state=42).reset_index(drop=True)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize_unicode(text: str) -> list[str]:
    return AFRICAN_LETTER_RE.findall(str(text).lower())


def rule_clean(text: str, preserve_diacritics: bool = True) -> str:
    text = html.unescape(str(text)).lower()
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = URL_RE.sub(" ", text)
    text = REPEATED_RE.sub(r"\1\1", text)
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not preserve_diacritics:
        text = strip_accents(text)
    return text


def fit_tfidf_vocabulary(series: pd.Series, min_df: int = 2, max_df: float = 0.75) -> set[str]:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_unicode,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
    )
    vectorizer.fit(series)
    return set(vectorizer.get_feature_names_out())


def tfidf_clean(text: str, vocabulary: set[str], protected_words: set[str]) -> str:
    return " ".join(
        token for token in tokenize_unicode(text)
        if token in vocabulary or token in protected_words
    )


def simple_morph_clean(text: str, protected_words: set[str]) -> str:
    """Lightweight morphology-aware fallback when language models are unavailable."""
    cleaned = []
    for token in tokenize_unicode(text):
        if token in protected_words:
            cleaned.append(token)
            continue
        if len(token) > 5 and token.endswith(("ing", "ed", "ly", "s")):
            token = re.sub(r"(ing|ed|ly|s)$", "", token)
        cleaned.append(token)
    return " ".join(cleaned)


def transformer_tokenize(series: pd.Series) -> tuple[pd.Series, str]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        processed = []
        for text in series:
            tokens = tokenizer.tokenize(str(text), truncation=True, max_length=256)
            processed.append(" ".join(tokens))
        return pd.Series(processed), "xlm-roberta-base multilingual subword tokenizer"
    except Exception as error:
        fallback = series.apply(lambda text: simple_morph_clean(text, set()))
        return fallback, f"fallback regex morphology cleaner; tokenizer unavailable: {error}"


def vocab_size(series: pd.Series) -> int:
    vocab: set[str] = set()
    for text in series:
        vocab.update(tokenize_unicode(text))
    return len(vocab)


def avg_words(series: pd.Series) -> float:
    counts = [len(tokenize_unicode(text)) for text in series]
    return float(np.mean(counts)) if counts else 0.0


def type_token_ratio(series: pd.Series) -> float:
    tokens: list[str] = []
    for text in series:
        tokens.extend(tokenize_unicode(text))
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def accent_variant_count(series: pd.Series) -> int:
    variants: dict[str, set[str]] = {}
    for text in series:
        for token in tokenize_unicode(text):
            variants.setdefault(strip_accents(token), set()).add(token)
    return sum(1 for forms in variants.values() if len(forms) > 1)


def noise_summary(df: pd.DataFrame, language: str) -> dict[str, object]:
    text = df["Text"].astype(str)
    tokens = []
    for value in text:
        tokens.extend(tokenize_unicode(value))
    counter = Counter(tokens)
    return {
        "language": language,
        "documents": len(df),
        "html_documents": int(text.str.contains(HTML_RE).sum()),
        "url_documents": int(text.str.contains(URL_RE).sum()),
        "repeated_char_documents": int(text.apply(lambda value: bool(REPEATED_RE.search(value))).sum()),
        "vocabulary_size": len(counter),
        "average_words": round(avg_words(text), 2),
        "type_token_ratio": round(type_token_ratio(text), 4),
        "accent_variant_groups": accent_variant_count(text),
        "top_tokens": ", ".join([word for word, _ in counter.most_common(12)]),
    }


def preprocessing_summary(df: pd.DataFrame, language: str) -> pd.DataFrame:
    protected = ENGLISH_SENTIMENT_WORDS if language == "English" else YORUBA_SENTIMENT_WORDS
    preserve_diacritics = language != "English"

    result = df.copy()
    result["rule_clean"] = result["Text"].apply(lambda text: rule_clean(text, preserve_diacritics))

    tfidf_vocab = fit_tfidf_vocabulary(result["rule_clean"])
    result["tfidf_clean"] = result["rule_clean"].apply(lambda text: tfidf_clean(text, tfidf_vocab, protected))

    start = time.time()
    result["deep_clean"], deep_note = transformer_tokenize(result["rule_clean"])
    deep_seconds = time.time() - start

    versions = ["Text", "rule_clean", "tfidf_clean", "deep_clean"]
    rows = []
    for version in versions:
        rows.append({
            "language": language,
            "version": "raw" if version == "Text" else version,
            "vocabulary_size": vocab_size(result[version]),
            "average_words": round(avg_words(result[version]), 2),
            "type_token_ratio": round(type_token_ratio(result[version]), 4),
            "sentiment_cue_rate": round(sentiment_cue_rate(result[version], protected), 4),
            "accent_variant_groups": accent_variant_count(result[version]),
            "deep_method_note": deep_note if version == "deep_clean" else "",
            "deep_seconds": round(deep_seconds, 2) if version == "deep_clean" else "",
        })
    result[["Text", "rule_clean", "tfidf_clean", "deep_clean"]].head(20).to_csv(
        OUT_DIR / f"{language.lower()}_examples.csv", index=False, encoding="utf-8"
    )
    return pd.DataFrame(rows)


def sentiment_cue_rate(series: pd.Series, cue_words: set[str]) -> float:
    counts = []
    for text in series:
        tokens = tokenize_unicode(text)
        if not tokens:
            continue
        counts.append(sum(token in cue_words for token in tokens) / len(tokens))
    return float(np.mean(counts)) if counts else 0.0


def make_report(noise: pd.DataFrame, summary: pd.DataFrame) -> str:
    english = summary[summary["language"] == "English"]
    african = summary[summary["language"] == LANGUAGE_NAME]

    def reduction(frame: pd.DataFrame, version: str) -> float:
        raw = int(frame.loc[frame["version"] == "raw", "vocabulary_size"].iloc[0])
        new = int(frame.loc[frame["version"] == version, "vocabulary_size"].iloc[0])
        return round(100 * (raw - new) / raw, 2) if raw else 0.0

    return f"""# Dataset B: English vs {LANGUAGE_NAME} Preprocessing

Dataset A is the local Amazon Fine Food Reviews file. Dataset B is {LANGUAGE_NAME}
text from MasakhaNER, a Masakhane named-entity dataset. The MasakhaNER source is:
https://github.com/masakhane-io/masakhane-ner

## Noise Patterns

{markdown_table(noise)}

English reviews show product-review noise such as HTML, URLs, punctuation, repeated
characters, informal casing, and spelling variation. {LANGUAGE_NAME} text has less
HTML/product noise because it is a curated NER corpus, but it carries multilingual
preprocessing issues: diacritics, named entities, token spelling variation, and
language-specific morphology.

## Vocabulary and Token Consistency

{markdown_table(summary.drop(columns=["deep_method_note"]))}

Rule-based cleaning reduced English vocabulary by {reduction(english, "rule_clean")}%
and {LANGUAGE_NAME} vocabulary by {reduction(african, "rule_clean")}%. TF-IDF
filtering reduced English vocabulary by {reduction(english, "tfidf_clean")}%
and {LANGUAGE_NAME} vocabulary by {reduction(african, "tfidf_clean")}%. The
multilingual tokenizer gives a different kind of preprocessing: it improves token
coverage for rare forms by splitting words into subwords, but this increases token
counts and is harder to read than word-level cleaning.

## Technique Comparison

Rule-based preprocessing is fastest and most interpretable. It removes visible
noise but can damage African-language text if it drops accented characters, so the
{LANGUAGE_NAME} pipeline preserves Unicode letters and diacritics.

Statistical preprocessing with TF-IDF is useful for vocabulary reduction. It handles
rare spelling variants by removing them, but this can also remove legitimate
morphological forms, named entities, and low-frequency African-language words.

Deep-learning preprocessing uses a multilingual subword tokenizer when available:
{summary.loc[summary["version"] == "deep_clean", "deep_method_note"].iloc[0]}.
Subword tokenization handles unseen words better than simple rules, but it is more
computationally expensive and less transparent.

## Morphology, Spelling Variation, and Sentiment Clarity

English benefits from lemmatization and sentiment-cue preservation because the
review scores provide a natural sentiment context. {LANGUAGE_NAME} does not have
sentiment labels in this MasakhaNER dataset, so sentiment clarity is only measured
with a small cue-word proxy. This makes the comparison linguistically useful but not
a true sentiment evaluation for Dataset B.

The main multilingual challenge is that a preprocessing rule that is safe for
English can be harmful for African languages. Removing accents, applying English
stop-word lists, or forcing ASCII can erase meaning. A better multilingual pipeline
uses Unicode-aware tokenization, language-specific protected words, conservative
statistical thresholds, and multilingual subword models.

## Computational and Linguistic Challenges

Rule-based cleaning is cheap but brittle. TF-IDF requires fitting a vocabulary and
needs careful thresholds for small datasets. Multilingual transformer tokenization
is slower and may require model downloads, but it is stronger for rare names,
loanwords, and spelling variation. For low-resource languages, the biggest issue is
not only computation; it is knowing which linguistic features should be preserved.
"""


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without requiring tabulate."""
    display = frame.fillna("").astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()

    def clean(value: str) -> str:
        return value.replace("\n", " ").replace("|", "\\|")

    widths = [
        max(len(clean(str(value))) for value in [header] + [row[index] for row in rows])
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        clean(header).ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(clean(row[index]).ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Loading English reviews...")
    english_df = load_english_reviews()

    print(f"Loading Masakhane {LANGUAGE_NAME} text...")
    african_df = load_african_text()

    noise = pd.DataFrame([
        noise_summary(english_df, "English"),
        noise_summary(african_df, LANGUAGE_NAME),
    ])

    print("Running preprocessing comparisons...")
    summary = pd.concat([
        preprocessing_summary(english_df, "English"),
        preprocessing_summary(african_df, LANGUAGE_NAME),
    ], ignore_index=True)

    noise.to_csv(OUT_DIR / "noise_summary.csv", index=False, encoding="utf-8")
    summary.to_csv(OUT_DIR / "preprocessing_summary.csv", index=False, encoding="utf-8")
    (OUT_DIR / "dataset_b_report.md").write_text(make_report(noise, summary), encoding="utf-8")

    print("\nNoise summary")
    print(noise)
    print("\nPreprocessing summary")
    print(summary)
    print(f"\nWrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
