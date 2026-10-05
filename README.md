# 🔍 ChatLens — WhatsApp Chat NLP Analyzer

> Deep NLP insights from your WhatsApp conversations. Upload a chat export and uncover sentiment trends, hidden topics, language patterns, and more.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://chatlens-whatsapp-chat-nlp-analyzer.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-VADER%20·%20LDA%20·%20sklearn-6C63FF)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Smart Statistics** | Message counts, word counts, media/link sharing, vocabulary richness (Type-Token Ratio) |
| 😊 **Sentiment Analysis** | VADER-powered compound scoring — weekly trend lines, per-user averages, mood distribution |
| 🧠 **Topic Discovery** | LDA topic modeling surfaces hidden conversation themes with top keywords |
| 🔗 **N-gram Analysis** | Most frequent bigrams and trigrams reveal common phrases and expressions |
| 🕒 **Activity Heatmaps** | Hourly × weekday heatmap, monthly trends, day-of-week and month breakdowns |
| 👑 **Participant Ranking** | Most active chatters with message share percentages |
| 💬 **Word Frequency** | Top words after Hinglish + English stopword removal |
| 😀 **Emoji Insights** | Top emojis, emoji vs. non-emoji message ratio |
| 📈 **Interactive Charts** | All visualizations built with Plotly — zoomable, hoverable, exportable |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS (gradient cards, hero landing page)
- **Charts**: [Plotly](https://plotly.com/python/) (interactive, dark-themed)
- **Sentiment**: [NLTK VADER](https://www.nltk.org/howto/sentiment.html) — tuned for social media text
- **Topics**: [scikit-learn LDA](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda) with CountVectorizer
- **N-grams**: scikit-learn CountVectorizer with configurable n-gram range
- **Parsing**: Custom regex-based parser with auto date/time format detection

---

## 📁 Project Structure

```
ChatLens/
├── app.py              # Streamlit UI — layout, charts, user interaction
├── chat_parser.py      # WhatsApp chat export parser (auto-detects formats)
├── nlp_engine.py       # NLP analysis — sentiment, LDA, n-grams, stats
├── config.py           # Stopwords (Hinglish + English), colors, app config
├── requirements.txt    # Python dependencies
├── sample_chat.txt     # Sample chat file for testing
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ChatLens.git
cd ChatLens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Upload a chat

- Open any WhatsApp chat → **⋮ Menu** → **More** → **Export Chat** → **Without Media**
- Upload the `.txt` file in the sidebar
- Select a participant (or **Everyone**) and click **🚀 Run Analysis**

> A `sample_chat.txt` is included for quick testing.

---

## 🧪 NLP Techniques Used

### VADER Sentiment Analysis
Uses the **Valence Aware Dictionary and sEntiment Reasoner** from NLTK, specifically designed for social media text. It handles:
- Slang and informal language
- Capitalization emphasis (e.g., "AMAZING" scores higher than "amazing")
- Punctuation (exclamation marks boost intensity)
- Negation ("not good" → negative)
- Emoji sentiment

### Latent Dirichlet Allocation (LDA)
Unsupervised topic modeling that discovers hidden thematic structure:
- Messages are vectorized using **Bag-of-Words** (CountVectorizer)
- Only alphabetic tokens with **3+ characters** are considered
- Combined **Hinglish + sklearn English stopwords** are removed
- Topics are extracted with configurable count and keyword depth

### N-gram Extraction
Identifies frequently co-occurring word sequences:
- **Bigrams** (2-word phrases) and **Trigrams** (3-word phrases)
- Stopword-filtered to surface meaningful phrases only

---

## 🌐 Stopword Handling

ChatLens includes a **custom Hinglish stopword list** (150+ words) covering:
- Hindi transliterated fillers: `hai`, `kya`, `nahi`, `toh`, `bhi`, `yaar`, `bhai`...
- English contractions fragments: `ll`, `ve`, `re`, `don`, `didn`...
- WhatsApp noise: `media`, `omitted`, `deleted`, `sticker`...
- Common generic words: `good`, `nice`, `really`, `gonna`, `wanna`...

These are merged with **sklearn's built-in English stopwords** for comprehensive filtering.

---

## 📸 Screenshots

*Upload a WhatsApp chat to see: gradient metric cards, sentiment trend lines, LDA topic tables, bigram/trigram bar charts, hourly activity heatmaps, and interactive Plotly visualizations — all in a dark-themed UI.*

---

## 👨‍💻 Developers

- **Kalash Pandey**
- **Shreesh Nalawade**
- **Divya Patil**

---

## 📄 License

This project is for educational purposes (NLP Lab Mini Project).

---

Built with ❤️ by Kalash Pandey, Shreesh Nalawade & Divya Patil — using Python, Streamlit, NLTK, and scikit-learn.
