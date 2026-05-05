# # Amazon Fine Food Reviews Text Preprocessing Assignment
#
# **Dataset:** Amazon Fine Food Reviews  
# **Source:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews  
# **File used:** `Reviews.csv`
#
# This notebook shows text noise inspection, rule-based cleaning, statistical filtering, spaCy/transformer preprocessing, and comparison of preprocessing effects.

# ## 1. Install Libraries
#
# Run these cells once. If the packages are already installed, you can skip them later.

# Install these in a terminal or notebook before running the script:
# python -m pip install pandas numpy beautifulsoup4 scikit-learn spacy pyspellchecker transformers torch
# python -m spacy download en_core_web_sm

# ## 2. Import Libraries

import re
import time

import numpy as np
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer

# ## 3. Load Dataset

df = pd.read_csv("Reviews.csv")
df = df[["Score", "Text"]].dropna()

df["sentiment"] = df["Score"].apply(
    lambda x: "positive" if x >= 4 else "negative" if x <= 2 else "neutral"
)

# Use a smaller sample so the notebook runs faster.
data = df.sample(min(20000, len(df)), random_state=42).reset_index(drop=True)

print("Full dataset:", len(df))
print("Sample used:", len(data))
data.head()

# ## 4. Inspect Noise
#
# Here we check for HTML tags, repeated characters, and possible spelling errors.

data["has_html"] = data["Text"].str.contains(r"<[^>]+>", regex=True)
data["has_repeated_chars"] = data["Text"].str.contains(r"(.)\1{2,}", regex=True)

noise_summary = pd.DataFrame({
    "noise_type": ["HTML tags", "Repeated characters"],
    "review_count": [data["has_html"].sum(), data["has_repeated_chars"].sum()]
})

noise_summary

spell = SpellChecker()
words = re.findall(r"[a-zA-Z]+", " ".join(data["Text"].head(1000)).lower())
possible_spelling_errors = sorted(spell.unknown(words))[:30]

possible_spelling_errors

# ## 5. Rule-Based Cleaning
#
# This removes HTML, URLs, extra symbols, and repeated characters using simple rules and regex.

sentiment_tokens = {
    "not", "no", "never", "cannot", "can't", "don't", "didn't",
    "good", "great", "excellent", "bad", "awful", "terrible",
    "love", "loved", "hate", "hated", "best", "worst"
}

def rule_clean(text):
    text = str(text).lower()
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-zA-Z!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["rule_clean"] = data["Text"].apply(rule_clean)
data[["Text", "rule_clean"]].head()

# ## 6. Statistical Filtering Using TF-IDF
#
# TF-IDF filtering removes very rare words and words that appear in too many reviews. Sentiment words are preserved.

vectorizer = TfidfVectorizer(
    min_df=5,
    max_df=0.60,
    max_features=30000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b"
)

vectorizer.fit(data["rule_clean"])
tfidf_words = set(vectorizer.get_feature_names_out())

def tfidf_clean(text):
    return " ".join(
        word for word in str(text).split()
        if word in tfidf_words or word in sentiment_tokens
    )

data["tfidf_clean"] = data["rule_clean"].apply(tfidf_clean)
data[["rule_clean", "tfidf_clean"]].head()

# ## 7. spaCy-Based Preprocessing
#
# spaCy performs tokenization and lemmatization. Stop words are removed, but sentiment-critical words such as negations are preserved.

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def spacy_clean(text):
    doc = nlp(str(text))
    cleaned = []

    for token in doc:
        word = token.text.lower()

        if word in sentiment_tokens:
            cleaned.append(word)
        elif token.is_stop or token.is_punct or token.is_space:
            continue
        elif token.is_alpha:
            cleaned.append(token.lemma_.lower())

    return " ".join(cleaned)

data["spacy_clean"] = data["rule_clean"].apply(spacy_clean)
data[["rule_clean", "spacy_clean"]].head()

# ## 8. Optional Transformer Sentiment Clarity Check
#
# This checks whether the cleaned text still preserves sentiment. It uses a small sample because transformer models can be slow.

try:
    from transformers import pipeline

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    eval_data = data.sample(min(100, len(data)), random_state=42).copy()

    for col in ["Text", "rule_clean", "tfidf_clean", "spacy_clean"]:
        scores = []
        for text in eval_data[col]:
            result = sentiment_model(str(text)[:512])[0]
            scores.append(result["score"])
        eval_data[col + "_confidence"] = scores

    transformer_summary = eval_data[[
        "Text_confidence",
        "rule_clean_confidence",
        "tfidf_clean_confidence",
        "spacy_clean_confidence"
    ]].mean().reset_index()
    transformer_summary.columns = ["text_version", "average_sentiment_confidence"]

except Exception as error:
    transformer_summary = pd.DataFrame({
        "message": ["Transformer check skipped. Run the install cell or check internet/model access."],
        "error": [str(error)]
    })

transformer_summary

# ## 9. Compare Vocabulary Reduction

def vocab_size(series):
    words = set()
    for text in series:
        words.update(str(text).split())
    return len(words)

def average_length(series):
    return np.mean([len(str(text).split()) for text in series])

comparison = pd.DataFrame({
    "version": ["raw", "rule_based", "tfidf", "spacy"],
    "vocabulary_size": [
        vocab_size(data["Text"]),
        vocab_size(data["rule_clean"]),
        vocab_size(data["tfidf_clean"]),
        vocab_size(data["spacy_clean"])
    ],
    "average_words_per_review": [
        average_length(data["Text"]),
        average_length(data["rule_clean"]),
        average_length(data["tfidf_clean"]),
        average_length(data["spacy_clean"])
    ]
})

comparison

# ## 10. Compare Computational Cost

small_sample = data["Text"].head(1000)

start = time.time()
small_sample.apply(rule_clean)
rule_seconds = time.time() - start

start = time.time()
small_sample.apply(lambda text: tfidf_clean(rule_clean(text)))
tfidf_seconds = time.time() - start

start = time.time()
small_sample.apply(lambda text: spacy_clean(rule_clean(text)))
spacy_seconds = time.time() - start

cost_comparison = pd.DataFrame({
    "method": ["rule_based", "tfidf_statistical", "spacy_deep_preprocessing"],
    "seconds_for_1000_reviews": [rule_seconds, tfidf_seconds, spacy_seconds],
    "cost_comment": [
        "Fastest and easiest to scale",
        "Moderate cost because TF-IDF must be fitted first",
        "Slowest but gives more linguistically informed text"
    ]
})

cost_comparison

# ## 11. Discussion and Conclusion
#
# The dataset contains noise such as HTML tags, repeated characters, and possible spelling errors. Rule-based cleaning is useful for quickly removing visible noise like tags, URLs, symbols, and repeated characters.
#
# Statistical filtering with TF-IDF reduces the vocabulary by removing very rare words and overly common words. This can improve text quality, but thresholds must be chosen carefully because useful domain words may be removed.
#
# spaCy preprocessing improves the text using tokenization, lemmatization, and stopword removal. Important sentiment tokens such as negations and strong opinion words are preserved because removing them may change the meaning of the review.
#
# Transformer sentiment confidence can be used to check whether the cleaned text still carries sentiment clearly. This method is more computationally expensive, so it is best used on smaller samples or with stronger hardware.
#
# Overall, rule-based cleaning is the fastest, TF-IDF filtering is moderate in cost, and spaCy/transformer-based preprocessing is slower but gives better language-aware cleaning.
