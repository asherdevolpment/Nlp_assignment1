# Dataset B: English vs Yoruba Preprocessing

Dataset A is the local Amazon Fine Food Reviews file. Dataset B is Yoruba
text from MasakhaNER, a Masakhane named-entity dataset. The MasakhaNER source is:
https://github.com/masakhane-io/masakhane-ner

## Noise Patterns

| language | documents | html_documents | url_documents | repeated_char_documents | vocabulary_size | average_words | type_token_ratio | accent_variant_groups | top_tokens                                        |
| -------- | --------- | -------------- | ------------- | ----------------------- | --------------- | ------------- | ---------------- | --------------------- | ------------------------------------------------- |
| English  | 5000      | 1257           | 87            | 899                     | 14238           | 80.91         | 0.0352           | 0                     | the, i, and, a, it, to, of, is, this, br, in, for |
| Yoruba   | 3121      | 0              | 0             | 357                     | 6341            | 25.38         | 0.0801           | 645                   | tí, ó, ní, àwọn, ọ, ẹ, ti, ń, àti, ṣe, wọ, fún    |

English reviews show product-review noise such as HTML, URLs, punctuation, repeated
characters, informal casing, and spelling variation. Yoruba text has less
HTML/product noise because it is a curated NER corpus, but it carries multilingual
preprocessing issues: diacritics, named entities, token spelling variation, and
language-specific morphology.

## Vocabulary and Token Consistency

| language | version     | vocabulary_size | average_words | type_token_ratio | sentiment_cue_rate | accent_variant_groups | deep_seconds |
| -------- | ----------- | --------------- | ------------- | ---------------- | ------------------ | --------------------- | ------------ |
| English  | raw         | 14238           | 80.91         | 0.0352           | 0.0325             | 0                     |              |
| English  | rule_clean  | 14088           | 79.56         | 0.0354           | 0.0328             | 0                     |              |
| English  | tfidf_clean | 7678            | 69.81         | 0.022            | 0.0373             | 0                     |              |
| English  | deep_clean  | 7236            | 92.2          | 0.0157           | 0.03               | 0                     | 23.76        |
| Yoruba   | raw         | 6341            | 25.38         | 0.0801           | 0.0121             | 645                   |              |
| Yoruba   | rule_clean  | 6341            | 25.38         | 0.0801           | 0.0121             | 645                   |              |
| Yoruba   | tfidf_clean | 3068            | 24.32         | 0.0404           | 0.0125             | 301                   |              |
| Yoruba   | deep_clean  | 2532            | 58.36         | 0.0139           | 0.0017             | 221                   | 5.25         |

Rule-based cleaning reduced English vocabulary by 1.05%
and Yoruba vocabulary by 0.0%. TF-IDF
filtering reduced English vocabulary by 46.07%
and Yoruba vocabulary by 51.62%. The
multilingual tokenizer gives a different kind of preprocessing: it improves token
coverage for rare forms by splitting words into subwords, but this increases token
counts and is harder to read than word-level cleaning.

## Technique Comparison

Rule-based preprocessing is fastest and most interpretable. It removes visible
noise but can damage African-language text if it drops accented characters, so the
Yoruba pipeline preserves Unicode letters and diacritics.

Statistical preprocessing with TF-IDF is useful for vocabulary reduction. It handles
rare spelling variants by removing them, but this can also remove legitimate
morphological forms, named entities, and low-frequency African-language words.

Deep-learning preprocessing uses a multilingual subword tokenizer when available:
xlm-roberta-base multilingual subword tokenizer.
Subword tokenization handles unseen words better than simple rules, but it is more
computationally expensive and less transparent.

## Morphology, Spelling Variation, and Sentiment Clarity

English benefits from lemmatization and sentiment-cue preservation because the
review scores provide a natural sentiment context. Yoruba does not have
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
