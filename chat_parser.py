"""
ChatLens Chat Parser
Parses WhatsApp exported chat files into structured DataFrames.
Supports Android (24h, 12h, dot/slash/hyphen dates, en-dash/em-dash) and iOS (bracketed) formats,
along with invisible Unicode characters (BOM, LTR/RTL marks, narrow no-break space).
"""

import re
import pandas as pd
import emoji

# Pattern matching start of a WhatsApp message line.
# iOS style: [DD/MM/YY, HH:MM:SS AM/PM] or [DD.MM.YYYY, HH:MM]
# Android style: DD/MM/YY, HH:MM - or DD/MM/YYYY, 9:15 am - or 2024-01-12, 09:15 -
LINE_START_REGEX = re.compile(
    r"^(?:"
    r"\[\s*\d{1,4}[/.-]\d{1,4}[/.-]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?\s*\]"
    r"|"
    r"\d{1,4}[/.-]\d{1,4}[/.-]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?\s*(?:[-–—]|\s-)"
    r")",
    re.UNICODE
)

# Pattern for extracting timestamp string and rest of the line from a merged entry.
ENTRY_REGEX = re.compile(
    r"^(?:"
    r"\[\s*(?P<ts_ios>\d{1,4}[/.-]\d{1,4}[/.-]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?)\s*\]\s*(?P<rest_ios>.*)"
    r"|"
    r"(?P<ts_and>\d{1,4}[/.-]\d{1,4}[/.-]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?)\s*(?:[-–—]|\s-)\s*(?P<rest_and>.*)"
    r")$",
    re.DOTALL
)


def _clean_raw_text(raw_text: str) -> str:
    """Strip hidden Unicode control characters and normalize spaces."""
    # Strip BOM (\ufeff), Left-to-Right mark (\u200e), Right-to-Left mark (\u200f)
    cleaned = raw_text.replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    # Normalize narrow no-break space (\u202f) and non-breaking space (\xa0)
    cleaned = cleaned.replace("\u202f", " ").replace("\xa0", " ")
    return cleaned


def _parse_timestamps(ts_series: pd.Series) -> pd.Series:
    """
    Parse a series of timestamp strings into pd.Timestamp objects robustly.
    Handles DD/MM vs MM/DD, dot separators, 12h/24h clocks, and optional seconds.
    """
    cleaned_ts = ts_series.str.replace(r"\.", "/", regex=True)

    try:
        ts_parsed = pd.to_datetime(cleaned_ts, errors="coerce", format="mixed", dayfirst=True)
    except Exception:
        ts_parsed = pd.to_datetime(cleaned_ts, errors="coerce", dayfirst=True)

    if ts_parsed.isna().any():
        try:
            fallback = pd.to_datetime(cleaned_ts, errors="coerce", format="mixed", dayfirst=False)
        except Exception:
            fallback = pd.to_datetime(cleaned_ts, errors="coerce", dayfirst=False)
        ts_parsed = ts_parsed.fillna(fallback)

    return ts_parsed


def _extract_emojis(text: str) -> list:
    """Return a list of emoji characters found in text."""
    return [ch for ch in text if ch in emoji.EMOJI_DATA]


def _clean_text(text: str) -> str:
    """Strip media tags, URLs, emojis, and collapse whitespace."""
    out = emoji.replace_emoji(text, replace="")
    out = re.sub(
        r"<media omitted>|<this message was edited>|this message was deleted|\bnull\b",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"https?://\S+|www\.\S+", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def parse_chat(raw_text: str) -> pd.DataFrame:
    """
    Parse a raw WhatsApp chat-export string into a tidy DataFrame.

    Columns returned:
    timestamp, sender, raw_message, clean_text, emojis,
    year, month, month_num, day, hour, minute, weekday, date_only
    """
    cleaned_raw = _clean_raw_text(raw_text)

    # ── Merge continuation lines back onto their parent message ──
    merged = []
    for line in cleaned_raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if LINE_START_REGEX.match(stripped):
            merged.append(stripped)
        elif merged:
            merged[-1] += "\n" + stripped

    if not merged:
        raise ValueError(
            "No messages could be parsed. "
            "Make sure you uploaded a WhatsApp chat export (.txt)."
        )

    # ── Extract timestamp, sender, and message body ──
    records = []
    for entry in merged:
        m = ENTRY_REGEX.match(entry)
        if not m:
            continue
        raw_ts = m.group("ts_ios") or m.group("ts_and")
        rest = m.group("rest_ios") if m.group("ts_ios") else m.group("rest_and")

        # Separate sender : message
        m_sender = re.match(r"^([^:\n]+?):\s*(.*)$", rest, re.DOTALL)
        if m_sender:
            sender = m_sender.group(1).strip()
            message = m_sender.group(2).strip()
        else:
            sender = "__system__"
            message = rest.strip()

        records.append(
            {"ts_raw": raw_ts, "sender": sender, "raw_message": message}
        )

    if not records:
        raise ValueError(
            "No messages could be parsed. "
            "Make sure you uploaded a WhatsApp chat export (.txt)."
        )

    df = pd.DataFrame(records)

    # ── Reclassify system messages ──
    system_phrases = [
        "end-to-end encrypted",
        "created group",
        "changed the group",
        "added",
        "removed",
        "left",
        "joined using",
        "security code changed",
        "changed their phone number",
        "reset this group",
        "waiting for this message",
    ]
    for phrase in system_phrases:
        df.loc[df["raw_message"].str.contains(phrase, case=False, na=False), "sender"] = "__system__"

    df.loc[df["sender"].str.len() > 50, "sender"] = "__system__"

    # ── Parse timestamps ──
    df["timestamp"] = _parse_timestamps(df["ts_raw"])
    df.dropna(subset=["timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Drop system notifications ──
    df = df[df["sender"] != "__system__"].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No user messages found after parsing (only system notifications were detected)."
        )

    # ── Derive time columns ──
    ts = df["timestamp"]
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month_name()
    df["month_num"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["weekday"] = ts.dt.day_name()
    df["date_only"] = ts.dt.date

    # ── Emoji extraction & text cleaning ──
    df["emojis"] = df["raw_message"].apply(_extract_emojis)
    df["clean_text"] = df["raw_message"].apply(_clean_text)

    df.drop(columns=["ts_raw"], inplace=True)
    return df
