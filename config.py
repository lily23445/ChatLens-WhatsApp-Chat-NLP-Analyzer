"""
ChatLens Configuration
Central config for stopwords, color palette, and app settings.
"""

# Hinglish stopwords (Hindi transliterated + English common words in Indian WhatsApp chats)
HINGLISH_STOPWORDS = {
    # Hindi transliterated
    "hai", "kya", "nahi", "toh", "bhi", "ko", "ka", "ki", "ke", "se",
    "me", "mai", "mein", "ye", "yeh", "wo", "woh", "koi", "kuch", "ek",
    "do", "ho", "haan", "na", "aur", "par", "pe", "ja", "ab", "jab",
    "tab", "tak", "bas", "hum", "tum", "aap", "tera", "mera", "uska",
    "iska", "unka", "sabko", "sab", "log", "wala", "wali", "wale",
    "raha", "rahi", "rahe", "tha", "thi", "the", "hoga", "hogi",
    "kar", "karo", "karna", "karke", "liye", "diya", "diye", "de",
    "le", "lo", "baat", "bol", "bolo", "bolna", "accha", "achha",
    "theek", "thik", "okay", "ok", "hmm", "hmmm", "ohh", "ahh",
    "arrey", "arre", "yaar", "bro", "sir", "ji", "nhi", "kro",
    "krna", "krke", "krta", "krti", "hota", "hoti", "hote",
    "abhi", "phir", "fir", "bhai", "dude", "guys", "lol",
    "haha", "hehe", "hihi", "omg", "idk", "btw", "tbh",
    # English filler
    "the", "is", "in", "it", "of", "and", "to", "a", "i", "you",
    "that", "was", "for", "on", "are", "with", "this", "but", "be",
    "have", "not", "or", "an", "my", "if", "so", "just", "about",
    "what", "can", "will", "do", "no", "yes", "am", "has", "its",
    "like", "how", "all", "been", "from", "at", "by", "we", "he",
    "she", "they", "them", "their", "our", "your", "would", "could",
    "should", "did", "get", "got", "also", "than", "then", "now",
    "very", "when", "who", "which", "there", "here", "much", "more",
    "some", "any", "only", "well", "back", "after", "over", "such",
    "even", "most", "into", "too", "had", "her", "him", "his",
    "been", "one", "two", "three", "know", "think", "want", "going",
    "come", "go", "see", "say", "said", "tell", "told", "make",
    # Contraction fragments (left after apostrophe stripping)
    "ll", "ve", "re", "don", "didn", "doesn", "isn", "wasn",
    "won", "wouldn", "couldn", "shouldn", "hasn", "hadn",
    "aren", "weren", "ain", "let", "im", "ive", "youre",
    "thats", "dont", "cant", "wont", "isnt", "doesnt",
    # Common filler / ultra-short tokens
    "ya", "yo", "ur", "uu", "aa", "ee", "oo", "pls", "plz",
    "thx", "thnx", "msg", "msgs", "pic", "pics", "rn", "dm",
    "gt", "lt", "amp", "etc", "gonna", "wanna", "gotta",
    "good", "nice", "great", "really", "thing", "things",
    "today", "tomorrow", "yesterday", "time", "day", "night",
    # WhatsApp noise
    "media", "omitted", "deleted", "message", "null", "edited",
    "missed", "voice", "call", "video", "sticker",
}

# Color palette for Plotly charts (dark theme friendly)
COLORS = {
    "primary": "#6C63FF",       # Indigo
    "secondary": "#FF6584",     # Coral pink
    "accent": "#43E97B",        # Mint green
    "warning": "#FFD93D",       # Gold
    "neutral": "#8B8FA3",       # Slate
    "bg_dark": "#0E1117",       # Streamlit dark bg
    "card_bg": "#1E1E2E",       # Card background
    "text": "#FAFAFA",          # Light text
    "positive": "#43E97B",
    "negative": "#FF6584",
    "neutral_sent": "#FFD93D",
}

PLOTLY_PALETTE = [
    "#6C63FF", "#FF6584", "#43E97B", "#FFD93D", "#38B6FF",
    "#FF9F43", "#A29BFE", "#FD79A8", "#00CEC9", "#E17055",
    "#74B9FF", "#FDCB6E", "#B8E994", "#E056A0", "#7EFFF5",
]

APP_NAME = "ChatLens"
APP_ICON = "🔍"
APP_TAGLINE = "Deep NLP Insights from Your WhatsApp Conversations"
