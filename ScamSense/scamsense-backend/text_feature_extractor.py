# text_feature_extractor.py
#
# This turns a raw SMS message string into the exact cleaned-token
# format that the TF-IDF vectorizer (text_tfidf.pkl) was fitted on from
# scam_detector.ipynb.

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _nltk_ensure(resource: str, path: str) -> None:
    """
    Makes sure an NLTK data resource (e.g. the stopwords corpus) is
    available before it's used, downloading it only if it is not cached.

    Checking nltk.data.find() first avoids touching the network when
    the data is already cached, and the download uses a
    try/except so if there is no internet access, it will still start up
    normally and just prints a warning instead of crashing.
    """
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            print(
                f"Could not download '{resource}' (no internet access) "
                f"and it is not already cached locally."
            )


# Run once so that the stopword list and lemmatizer dictionary are ready before the first request arrives.
_nltk_ensure("stopwords", "corpora/stopwords")
_nltk_ensure("wordnet", "corpora/wordnet")
_nltk_ensure("omw-1.4", "corpora/omw-1.4")  # Open Multilingual Wordnet which is a WordNetLemmatizer dependency

# Built once and reused across every request rather than re-created
# WordNetLemmatizer() loads a large dictionary
_lemmatizer = WordNetLemmatizer()
_STOPWORDS_SET = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """
    Mirrors scam_detector.ipynb Section 3.2 exactly, so the TF-IDF
    vectorizer sees the same token distribution it was fitted on.

    Pipeline, in order (same order as the notebook):
      1. Lowercase everything - so "FREE" and "free" count as the same word.
      2. Strip HTML tags - removes any "<html>"markup in the input.
      3. Replace URLs with the token "URLTOKEN" - the model focuses on the presence of a link,
         so using a shared token lets the classifier learn "contains a link" as a single reusable feature.
      4. Replace long digit sequences with "PHONETOKEN" which is the same as URLTOKEN but for
         phone numbers.
      5. Replace currency amounts (£, $, €) with "MONEYTOKEN".
      6. Strip all remaining punctuation/digits, keeping only letters and
         spaces.
      7. Split into individual words, then drop English stopwords ("the",
         "is", "at", ...) and single-character leftovers.
      8. Lemmatize each remaining word to its dictionary base form (e.g.
         "prizes" -> "prize", "winning" -> "win").

    The cleaned tokens are joined back into a single space-separated
    string.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", text)  # URLs
    text = re.sub(r"\+?[\d][\d\s\-().]{7,}[\d]", " PHONETOKEN ", text)  # phones
    text = re.sub(r"[$£€]\s*[\d,]+\.?\d*", " MONEYTOKEN ", text)  # currency
    text = re.sub(r"[^a-z\s]", " ", text)  # punctuation
    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _STOPWORDS_SET and len(t) > 1
    ]
    return " ".join(tokens)
