"""
ChatLens — Deep NLP Insights from Your WhatsApp Conversations
Streamlit application entry point.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from chat_parser import parse_chat
from nlp_engine import (
    compute_stats,
    most_active_senders,
    chat_date_range,
    sentiment_timeline,
    sentiment_distribution,
    sentiment_by_sender,
    discover_topics,
    top_ngrams,
    monthly_activity,
    daily_activity,
    weekday_counts,
    month_name_counts,
    hourly_heatmap,
    word_frequencies,
    emoji_stats,
)
from config import COLORS, PLOTLY_PALETTE, APP_NAME, APP_ICON, APP_TAGLINE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Page setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title=f"{APP_NAME} — WhatsApp NLP Analyzer",
    page_icon=APP_ICON,
    layout="wide",
)

# ── Inject custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── metric cards ───────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1E1E2E 0%, #2D2B55 100%);
    border-radius: 16px;
    padding: 22px 16px;
    text-align: center;
    border: 1px solid rgba(108, 99, 255, 0.18);
    transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(108, 99, 255, 0.18);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #FF6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #8B8FA3;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: .6px;
    font-weight: 600;
}

/* ── section headers ────────────────────────────────────── */
.sec-hdr {
    font-size: 1.45rem;
    font-weight: 700;
    margin: 2.2rem 0 .6rem;
    padding-bottom: 6px;
    border-bottom: 3px solid #6C63FF;
    display: inline-block;
}

/* ── sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
}

/* ── landing hero ───────────────────────────────────────── */
.hero { text-align: center; padding: 3rem 1rem; }
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #FF6584 50%, #43E97B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #8B8FA3; font-size: 1.15rem; margin-top: .4rem; }
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
    margin-top: 2rem;
}
.feature-item {
    background: #1E1E2E;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(108, 99, 255, 0.12);
    transition: border-color .2s;
}
.feature-item:hover { border-color: #6C63FF; }
.feature-item .icon { font-size: 1.8rem; }
.feature-item .title { font-weight: 700; margin-top: 6px; }
.feature-item .desc  { font-size: .82rem; color: #8B8FA3; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Plotly shared layout ─────────────────────────────────────────
PL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=COLORS["text"]),
    margin=dict(l=40, r=40, t=50, b=40),
)


def _metric(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.markdown(f"# {APP_ICON} {APP_NAME}")
st.sidebar.caption(APP_TAGLINE)
st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "📁 Upload WhatsApp Chat Export", type="txt",
    help="Open a chat → ⋮ → More → Export Chat → Without Media",
)

# ── Landing page ─────────────────────────────────────────────────
if uploaded is None:
    st.markdown(f"""
    <div class="hero">
        <h1>{APP_ICON} {APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
    </div>
    <div class="feature-grid">
        <div class="feature-item">
            <div class="icon">📊</div>
            <div class="title">Smart Statistics</div>
            <div class="desc">Message counts, vocabulary richness, and more.</div>
        </div>
        <div class="feature-item">
            <div class="icon">😊</div>
            <div class="title">Sentiment Analysis</div>
            <div class="desc">VADER-powered mood tracking over time.</div>
        </div>
        <div class="feature-item">
            <div class="icon">🧠</div>
            <div class="title">Topic Discovery</div>
            <div class="desc">LDA topic modelling reveals conversation themes.</div>
        </div>
        <div class="feature-item">
            <div class="icon">🔗</div>
            <div class="title">N-gram Analysis</div>
            <div class="desc">Most frequent bigrams &amp; trigrams in your chat.</div>
        </div>
        <div class="feature-item">
            <div class="icon">🕒</div>
            <div class="title">Activity Heatmaps</div>
            <div class="desc">Discover when you chat the most.</div>
        </div>
        <div class="feature-item">
            <div class="icon">😀</div>
            <div class="title">Emoji Insights</div>
            <div class="desc">Usage patterns and favourite emojis.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Parse the uploaded chat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

raw_bytes = uploaded.getvalue()
try:
    raw = raw_bytes.decode("utf-8-sig")
except UnicodeDecodeError:
    try:
        raw = raw_bytes.decode("utf-16")
    except UnicodeDecodeError:
        raw = raw_bytes.decode("latin-1", errors="replace")

try:
    df = parse_chat(raw)
except Exception as exc:
    st.error(f"⚠️ Could not parse chat file: {exc}")
    st.stop()

if df.empty:
    st.warning("The parsed DataFrame is empty — double-check the file format.")
    st.stop()

st.sidebar.success(
    f"✅ **{len(df):,}** messages · **{df['sender'].nunique()}** participants"
)

# ── Sender selector ──────────────────────────────────────────────
senders = sorted(df["sender"].unique().tolist())
senders.insert(0, "Everyone")
chosen = st.sidebar.selectbox("👤 Analyze", senders)
who = None if chosen == "Everyone" else chosen

st.sidebar.divider()
go_btn = st.sidebar.button("🚀 Run Analysis", use_container_width=True, type="primary")

if not go_btn:
    st.info("Pick a participant (or **Everyone**) and hit **Run Analysis** to begin.")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Page header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

first_dt, last_dt, span = chat_date_range(df, who)
label = f"**{chosen}**" if chosen != "Everyone" else "**Everyone**"
st.markdown(f"# {APP_ICON} {label} — Chat Analysis")
st.caption(
    f"📅 {first_dt.strftime('%b %d, %Y')} → {last_dt.strftime('%b %d, %Y')}  ·  "
    f"**{span:,}** days of conversation"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1 · Key Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

stats = compute_stats(df, who)
c1, c2, c3, c4, c5, c6 = st.columns(6)
items = [
    ("Messages", f"{stats['total_messages']:,}"),
    ("Words", f"{stats['total_words']:,}"),
    ("Media", f"{stats['media_shared']:,}"),
    ("Links", f"{stats['links_shared']:,}"),
    ("Avg Words/Msg", f"{stats['avg_msg_length']}"),
    ("Vocab Richness", f"{stats['vocabulary_richness']:.1%}"),
]
for col, (lbl, val) in zip([c1, c2, c3, c4, c5, c6], items):
    with col:
        _metric(lbl, val)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2 · Sentiment Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="sec-hdr">😊 Sentiment Analysis</div>', unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    trend = sentiment_timeline(df, who)
    if not trend.empty:
        span_days = (last_dt - first_dt).days
        freq_lbl = "Daily" if span_days <= 14 else ("Weekly" if span_days <= 180 else "Monthly")
        fig = px.area(
            trend, x="Date", y="Sentiment",
            title=f"{freq_lbl} Sentiment Trend",
            color_discrete_sequence=[COLORS["primary"]],
            markers=True,
        )
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(108,99,255,0.12)")
        fig.update_layout(**PL)
        if len(trend) == 1:
            d = pd.to_datetime(trend["Date"].iloc[0])
            fig.update_xaxes(range=[d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)])
        st.plotly_chart(fig, use_container_width=True)

with right:
    dist = sentiment_distribution(df, who)
    fig = go.Figure(go.Pie(
        labels=list(dist.keys()), values=list(dist.values()),
        marker_colors=[COLORS["positive"], COLORS["neutral_sent"], COLORS["negative"]],
        hole=.55, textinfo="percent+label", textfont_size=12,
    ))
    fig.update_layout(title="Mood Split", showlegend=False, **PL)
    st.plotly_chart(fig, use_container_width=True)

if chosen == "Everyone":
    by_user = sentiment_by_sender(df)
    if not by_user.empty:
        fig = px.bar(
            x=by_user.values, y=by_user.index, orientation="h",
            title="Average Sentiment by Participant",
            color=by_user.values,
            color_continuous_scale=["#FF6584", "#FFD93D", "#43E97B"],
        )
        fig.update_layout(**PL, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3 · Topic Discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="sec-hdr">🧠 Topic Discovery (LDA)</div>', unsafe_allow_html=True)

topics = discover_topics(df, who, n_topics=5)
if topics:
    st.dataframe(pd.DataFrame(topics), use_container_width=True, hide_index=True)
else:
    st.info("Need ≥ 20 substantial messages to extract topics.")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4 · N-gram Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="sec-hdr">🔗 N-gram Analysis</div>', unsafe_allow_html=True)

col_b, col_t = st.columns(2)

with col_b:
    bi = top_ngrams(df, who, n=2, top_k=12)
    if not bi.empty:
        fig = px.bar(
            bi[::-1], x="Count", y="Phrase", orientation="h",
            title="Top Bigrams",
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for bigrams.")

with col_t:
    tri = top_ngrams(df, who, n=3, top_k=12)
    if not tri.empty:
        fig = px.bar(
            tri[::-1], x="Count", y="Phrase", orientation="h",
            title="Top Trigrams",
            color_discrete_sequence=[COLORS["secondary"]],
        )
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for trigrams.")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5 · Activity Patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="sec-hdr">📊 Activity Patterns</div>', unsafe_allow_html=True)

# Monthly trend
mo = monthly_activity(df, who)
if not mo.empty:
    fig = px.bar(
        mo, x="Month", y="Messages", title="Messages per Month",
        color_discrete_sequence=[COLORS["accent"]],
    )
    fig.update_layout(**PL)
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

# Day-of-week vs Month-name
dw_col, mn_col = st.columns(2)

with dw_col:
    wd = weekday_counts(df, who)
    if not wd.empty:
        fig = px.bar(
            x=wd.index, y=wd.values, title="By Day of Week",
            color=wd.values, color_continuous_scale="Purples",
        )
        fig.update_layout(**PL, coloraxis_showscale=False,
                          xaxis_title="", yaxis_title="Messages")
        st.plotly_chart(fig, use_container_width=True)

with mn_col:
    mc = month_name_counts(df, who)
    if not mc.empty:
        fig = px.bar(
            x=mc.index, y=mc.values, title="By Month Name",
            color=mc.values, color_continuous_scale="Reds",
        )
        fig.update_layout(**PL, coloraxis_showscale=False,
                          xaxis_title="", yaxis_title="Messages")
        st.plotly_chart(fig, use_container_width=True)

# Heatmap
hm = hourly_heatmap(df, who)
if not hm.empty:
    fig = px.imshow(
        hm, title="Hourly Activity Heatmap",
        color_continuous_scale="Purples", aspect="auto",
    )
    fig.update_layout(**PL)
    fig.update_xaxes(title="Hour of Day")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6 · Most Active Users (Everyone only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if chosen == "Everyone":
    st.markdown(
        '<div class="sec-hdr">👑 Most Active Participants</div>',
        unsafe_allow_html=True,
    )
    active = most_active_senders(df, top_n=10)
    fig = px.bar(
        active, x=active.index, y="Messages",
        title="Top Participants",
        color="Share %", color_continuous_scale="Viridis",
    )
    fig.update_layout(**PL, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7 · Word Frequency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="sec-hdr">💬 Most Used Words</div>', unsafe_allow_html=True)

freq = word_frequencies(df, who, top_k=20)
if freq:
    fdf = pd.DataFrame(freq, columns=["Word", "Count"])
    fig = px.bar(
        fdf[::-1], x="Count", y="Word", orientation="h",
        title="Top 20 Words (stopwords removed)",
        color="Count", color_continuous_scale="Tealgrn",
    )
    fig.update_layout(**PL, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No word data available.")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8 · Emoji Insights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="sec-hdr">😀 Emoji Insights</div>', unsafe_allow_html=True)

edf, w_e, wo_e = emoji_stats(df, who)
e1, e2 = st.columns([2, 1])

with e1:
    if not edf.empty:
        fig = px.bar(
            edf.head(15), x="Emoji", y="Count",
            title="Top 15 Emojis",
            color="Count", color_continuous_scale="Sunset",
        )
        fig.update_layout(**PL, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emojis found.")

with e2:
    if w_e + wo_e > 0:
        fig = go.Figure(go.Pie(
            labels=["With Emoji", "Without Emoji"],
            values=[w_e, wo_e],
            marker_colors=[COLORS["primary"], COLORS["neutral"]],
            hole=.55, textinfo="percent+label", textfont_size=12,
        ))
        fig.update_layout(title="Emoji Usage", showlegend=False, **PL)
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Built with **{APP_NAME}** {APP_ICON}  ·  "
    "Powered by VADER · LDA · scikit-learn · Plotly"
)