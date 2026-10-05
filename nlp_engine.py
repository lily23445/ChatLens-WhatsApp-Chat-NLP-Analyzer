"""
ChatLens NLP Engine
Core analysis module: VADER sentiment, LDA topics, n-grams,
activity patterns, word frequency, and emoji stats.
"""

import pandas as pd
import numpy as np
from collections import Counter

import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import HINGLISH_STOPWORDS

# ── Bootstrap VADER lexicon ──────────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _scope(df, sender):
    """Return a copy filtered to *sender*; pass None for everyone."""
    if sender and sender != "Everyone":
        return df[df["sender"] == sender].copy()
    return df.copy()


def _meaningful_texts(df):
    """Return only non-trivial cleaned messages (len > 3)."""
    return df["clean_text"][df["clean_text"].str.len() > 3].tolist()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Basic Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_stats(df, sender=None):
    """Overview numbers for the selected scope."""
    sub = _scope(df, sender)
    word_counts = sub["clean_text"].str.split().str.len()
    all_words = " ".join(sub["clean_text"]).lower().split()
    unique = set(all_words) - HINGLISH_STOPWORDS

    return {
        "total_messages": len(sub),
        "total_words": int(word_counts.sum()),
        "media_shared": int(
            sub["raw_message"]
            .str.contains(r"<[Mm]edia omitted>", regex=True)
            .sum()
        ),
        "links_shared": int(
            sub["raw_message"].str.contains(r"https?://", regex=True).sum()
        ),
        "avg_msg_length": round(word_counts.mean(), 1) if len(sub) else 0,
        "vocabulary_richness": round(
            len(unique) / max(len(all_words), 1), 3
        ),
    }


def most_active_senders(df, top_n=10):
    """Rank senders by volume; return DataFrame with count + share %."""
    counts = df["sender"].value_counts().head(top_n)
    pct = round((counts / len(df)) * 100, 1)
    return pd.DataFrame({"Messages": counts, "Share %": pct})


def chat_date_range(df, sender=None):
    """Return (first_timestamp, last_timestamp, span_in_days)."""
    sub = _scope(df, sender)
    first, last = sub["timestamp"].min(), sub["timestamp"].max()
    return first, last, (last - first).days


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sentiment Analysis  (VADER)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compound(text):
    return _vader.polarity_scores(text)["compound"]


def sentiment_timeline(df, sender=None, freq=None):
    """Average compound sentiment resampled at *freq* (auto-selected based on date span if None)."""
    sub = _scope(df, sender)
    if sub.empty:
        return pd.DataFrame(columns=["Date", "Sentiment"])
    
    sub = sub.copy()
    sub["_sent"] = sub["clean_text"].apply(_compound)

    if freq is None:
        span_days = (sub["timestamp"].max() - sub["timestamp"].min()).days
        if span_days <= 14:
            freq = "D"
        elif span_days <= 180:
            freq = "W"
        else:
            freq = "ME"

    trend = sub.set_index("timestamp")["_sent"].resample(freq).mean().dropna()
    return trend.reset_index().rename(columns={"timestamp": "Date", "_sent": "Sentiment"})


def sentiment_distribution(df, sender=None):
    """Count of positive / neutral / negative messages."""
    sub = _scope(df, sender)
    scores = sub["clean_text"].apply(_compound)
    pos = int((scores > 0.05).sum())
    neg = int((scores < -0.05).sum())
    neu = len(scores) - pos - neg
    return {"Positive 😊": pos, "Neutral 😐": neu, "Negative 😞": neg}


def sentiment_by_sender(df):
    """Mean compound score per sender, sorted ascending."""
    df = df.copy()
    df["_sent"] = df["clean_text"].apply(_compound)
    return df.groupby("sender")["_sent"].mean().sort_values()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Topic Modelling  (LDA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _combined_stopwords():
    """Merge our Hinglish list with sklearn's English stopwords."""
    return list(HINGLISH_STOPWORDS | set(ENGLISH_STOP_WORDS))


def discover_topics(df, sender=None, n_topics=5, n_words=8):
    """
    Fit an LDA model and return a list of dicts
    [{"Topic": "Topic 1", "Key Words": "word1, word2, …"}, …]
    """
    texts = _meaningful_texts(_scope(df, sender))
    if len(texts) < 20:
        return []

    stop = _combined_stopwords()
    vec = CountVectorizer(
        max_df=0.85,
        min_df=5,
        stop_words=stop,
        max_features=1500,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # only alpha tokens, 3+ chars
    )
    try:
        dtm = vec.fit_transform(texts)
    except ValueError:
        return []

    if dtm.shape[1] < 5:
        return []  # not enough vocabulary to form meaningful topics

    k = min(n_topics, max(2, dtm.shape[1] // 15))
    lda = LatentDirichletAllocation(
        n_components=k, random_state=42, max_iter=30,
        learning_method="online",
    )
    lda.fit(dtm)

    names = vec.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_ids = comp.argsort()[-n_words:][::-1]
        topics.append({
            "Topic": f"Topic {i + 1}",
            "Key Words": ", ".join(names[j] for j in top_ids),
        })
    return topics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N-gram Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def top_ngrams(df, sender=None, n=2, top_k=15):
    """
    Most frequent n-grams (default bigrams).
    Returns a DataFrame with columns [Phrase, Count].
    """
    texts = _meaningful_texts(_scope(df, sender))
    if len(texts) < 5:
        return pd.DataFrame(columns=["Phrase", "Count"])

    stop = _combined_stopwords()
    vec = CountVectorizer(
        ngram_range=(n, n), stop_words=stop, max_features=top_k,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    try:
        mat = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["Phrase", "Count"])

    pairs = sorted(
        zip(vec.get_feature_names_out(), mat.sum(axis=0).A1),
        key=lambda x: x[1], reverse=True,
    )
    return pd.DataFrame(pairs, columns=["Phrase", "Count"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Word Frequency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def word_frequencies(df, sender=None, top_k=30):
    """Top *top_k* words after removing stopwords."""
    sub = _scope(df, sender)
    tokens = " ".join(sub["clean_text"]).lower().split()
    all_stop = HINGLISH_STOPWORDS | set(ENGLISH_STOP_WORDS)
    filtered = [w for w in tokens if w not in all_stop and len(w) > 2 and w.isalpha()]
    return Counter(filtered).most_common(top_k)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Activity Patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def monthly_activity(df, sender=None):
    sub = _scope(df, sender)
    if sub.empty:
        return pd.DataFrame(columns=["Month", "Messages"])
    grp = sub.groupby(sub["timestamp"].dt.to_period("M")).size()
    res = grp.reset_index(name="Messages")
    res["Month"] = res["timestamp"].dt.strftime("%b %Y")
    return res[["Month", "Messages"]]


def daily_activity(df, sender=None):
    sub = _scope(df, sender)
    return (
        sub.groupby("date_only").size()
        .reset_index(name="Messages")
        .rename(columns={"date_only": "Date"})
    )


def weekday_counts(df, sender=None):
    sub = _scope(df, sender)
    counts = sub["weekday"].value_counts()
    order = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
    return counts.reindex([d for d in order if d in counts.index])


def month_name_counts(df, sender=None):
    sub = _scope(df, sender)
    return sub["month"].value_counts()


def hourly_heatmap(df, sender=None):
    """Pivot table  weekday × hour  for a heatmap."""
    sub = _scope(df, sender)
    pivot = sub.pivot_table(
        index="weekday", columns="hour",
        values="raw_message", aggfunc="count", fill_value=0,
    )
    order = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
    return pivot.reindex([d for d in order if d in pivot.index])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Emoji Stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def emoji_stats(df, sender=None):
    """
    Returns (emoji_df, msgs_with_emoji, msgs_without_emoji).
    emoji_df has columns [Emoji, Count].
    """
    sub = _scope(df, sender)
    flat = [e for elist in sub["emojis"] for e in elist]

    if not flat:
        return pd.DataFrame(columns=["Emoji", "Count"]), 0, len(sub)

    top = Counter(flat).most_common(25)
    edf = pd.DataFrame(top, columns=["Emoji", "Count"])

    with_e = int(sub["emojis"].apply(len).gt(0).sum())
    without_e = len(sub) - with_e
    return edf, with_e, without_e
