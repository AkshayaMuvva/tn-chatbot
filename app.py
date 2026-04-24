"""
app.py — Streamlit Chat Frontend
Tamil Nadu Engineering College Admissions RAG Chatbot
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# ── Load .env FIRST (before anything else reads env vars) ─────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TN College Advisor | Admissions 2026-27",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

* { box-sizing: border-box; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome (keep header for sidebar toggle) ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999 !important;
}
.block-container { padding-top: 0rem !important; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(90deg, #7c3aed, #4f46e5, #0ea5e9);
    padding: 1.2rem 2rem;
    border-radius: 0 0 16px 16px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 24px rgba(124,58,237,0.4);
}
.header-banner h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
    margin: 0;
}
.header-banner p {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}

/* ── Chat container ── */
.chat-area {
    max-height: 62vh;
    overflow-y: auto;
    padding: 0.5rem 0.5rem 1rem 0.5rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(124,58,237,0.4) transparent;
}

/* ── Message bubbles ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 12px rgba(124,58,237,0.3);
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 0.6rem 0;
    gap: 0.6rem;
}
/* ════════════════════════════════════════
   STREAMLIT CHAT MESSAGES — full override
   ════════════════════════════════════════ */

/* The chat message container background */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    margin: 0.5rem 0 !important;
    padding: 0.8rem 1rem !important;
}

/* All text inside chat messages → bright white */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] code {
    color: #f1f5f9 !important;
}

/* Markdown rendered text inside chat */
[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] .stMarkdown * {
    color: #f1f5f9 !important;
}

/* Code blocks inside chat */
[data-testid="stChatMessage"] code {
    background: rgba(124, 58, 237, 0.2) !important;
    color: #c4b5fd !important;
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
}

/* Tables inside chat */
[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
}
[data-testid="stChatMessage"] th {
    background: rgba(124, 58, 237, 0.25) !important;
    color: #e2e8f0 !important;
    padding: 0.4rem 0.6rem;
    border: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stChatMessage"] td {
    color: #e2e8f0 !important;
    padding: 0.35rem 0.6rem;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Bold text in chat messages */
[data-testid="stChatMessage"] strong {
    color: #ffffff !important;
    font-weight: 700;
}

/* ════════════════════════════════════════
   BUTTONS — fix dark text on light bg
   ════════════════════════════════════════ */
.stButton > button {
    background: rgba(124, 58, 237, 0.15) !important;
    border: 1px solid rgba(124, 58, 237, 0.4) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(124, 58, 237, 0.35) !important;
    border-color: rgba(124, 58, 237, 0.7) !important;
    color: #ffffff !important;
}

/* ════════════════════════════════════════
   CHAT INPUT — white placeholder & text
   ════════════════════════════════════════ */
[data-testid="stChatInput"] > div {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(124, 58, 237, 0.4) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea::placeholder {
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255, 255, 255, 0.45) !important;
}

/* ════════════════════════════════════════
   SIDEBAR
   ════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: rgba(10, 8, 35, 0.95) !important;
    border-right: 1px solid rgba(124,58,237,0.2);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
.sidebar-section {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #a78bfa !important;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.profile-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.82rem;
}
.profile-key { color: rgba(255,255,255,0.5); }
.profile-val { color: #e2e8f0; font-weight: 500; }

/* ── College cards ── */
.college-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s;
}
.college-card:hover { border-color: rgba(124,58,237,0.5); }
.college-card-name {
    font-weight: 600;
    color: #c4b5fd;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.college-card-meta {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55);
    display: flex; gap: 1rem; flex-wrap: wrap;
}
.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-govt { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-private { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-deemed { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

/* ── Input area ── */
.stChatInput > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
    color: white !important;
}
.stChatInput textarea {
    color: white !important;
}

/* ── Suggestion chips ── */
.suggestion-chip {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    color: #c4b5fd;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    margin: 0.2rem;
    transition: all 0.2s;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34d399;
    margin-right: 6px;
    box-shadow: 0 0 6px #34d399;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }

/* Markdown inside bubbles */
.bubble p { margin: 0.3rem 0; }
.bubble ul { margin: 0.4rem 0; padding-left: 1.2rem; }
.bubble li { margin: 0.2rem 0; }
.bubble strong { color: #e2e8f0; }
.bubble h3, .bubble h4 { color: #c4b5fd; margin: 0.5rem 0 0.2rem; }
.bubble table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 0.5rem 0; }
.bubble th { background: rgba(124,58,237,0.2); color: #c4b5fd; padding: 0.4rem 0.6rem; }
.bubble td { border-top: 1px solid rgba(255,255,255,0.08); padding: 0.35rem 0.6rem; color: rgba(255,255,255,0.8); }
</style>
""", unsafe_allow_html=True)


# ── Init session state ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "history": [],
        "student_profile": {},
        "colleges_shown": [],
        "rag_context": "",
        "rag_ready": False,
        "graph_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Load RAG index and graph (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Load ChromaDB index and LangGraph (cached across reruns)."""
    from rag.embedder import is_index_ready, build_index
    from rag.chunker import create_chunks
    from graph.supervisor import get_graph

    if not is_index_ready():
        chunks = create_chunks()
        build_index(chunks, force_rebuild=False)

    graph = get_graph()
    return graph


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem; text-align: center;'>
        <div style='font-size: 2.5rem;'>🎓</div>
        <div style='font-family: Outfit; font-size: 1.1rem; font-weight: 700;
                    background: linear-gradient(90deg,#a78bfa,#60a5fa);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin-top: 0.3rem;'>
            TN College Advisor
        </div>
        <div style='font-size: 0.75rem; color: rgba(255,255,255,0.55); margin-top: 0.2rem;'>
            <span class='status-dot'></span>Powered by Groq + RAG
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Student Profile — toggle between card view and edit form ──────────
    st.markdown("---")
    st.markdown(
        "<div class='sidebar-title'>📋 Your Profile</div>",
        unsafe_allow_html=True,
    )

    # Track edit mode in session state
    if "profile_editing" not in st.session_state:
        st.session_state.profile_editing = True  # start in edit mode if no profile

    profile = st.session_state.student_profile
    has_profile = bool(profile)

    # If profile exists and not in edit mode → show compact card
    if has_profile and not st.session_state.profile_editing:
        LABELS = {
            "tnea_rank":       ("TNEA Rank",  lambda v: f"#{int(v):,}"),
            "jee_rank":        ("JEE Rank",   lambda v: f"#{int(v):,}"),
            "twelfth_percent": ("12th %",     lambda v: f"{float(v):.1f}%"),
            "community":       ("Community",  lambda v: v.upper()),
            "preferred_city":  ("City",       lambda v: str(v)),
            "preferred_branch":("Branch",     lambda v: str(v)[:26] + "…" if len(str(v)) > 26 else str(v)),
            "max_fee":         ("Max Fee",    lambda v: f"Rs.{int(v):,}/yr"),
        }
        rows_html = ""
        for key, (label, fmt) in LABELS.items():
            val = profile.get(key)
            if val:
                try:
                    dval = fmt(val)
                except Exception:
                    dval = str(val)
                rows_html += (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.07);"
                    f"font-size:0.82rem;'>"
                    f"<span style='color:rgba(255,255,255,0.55);'>{label}</span>"
                    f"<span style='color:#f1f5f9;font-weight:600;'>{dval}</span>"
                    f"</div>"
                )
        if rows_html:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);"
                f"border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.5rem;'>"
                f"{rows_html}</div>",
                unsafe_allow_html=True,
            )
        if st.button("✏️ Edit Profile", use_container_width=True, key="edit_profile_btn"):
            st.session_state.profile_editing = True
            st.rerun()

    else:
        # ── Show the input FORM ────────────────────────────────────────────
        if has_profile:
            st.markdown(
                "<div style='font-size:0.75rem;color:rgba(255,255,255,0.5);margin-bottom:0.4rem;'>"
                "Update your details below and save.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:rgba(255,255,255,0.6);margin-bottom:0.5rem;'>"
                "Fill in your details to get personalised college recommendations.</div>",
                unsafe_allow_html=True,
            )

        with st.form(key="profile_form", border=False):
            col1, col2 = st.columns(2)
            with col1:
                tnea_rank_inp = st.number_input(
                    "TNEA Rank",
                    min_value=1, max_value=250000,
                    value=int(profile.get("tnea_rank", 1)),
                    step=1,
                    help="Your TNEA rank (1 = best). Set 1 if not applicable.",
                )
            with col2:
                pct_inp = st.number_input(
                    "12th % *",
                    min_value=0.0, max_value=100.0,
                    value=float(profile.get("twelfth_percent", 0.0)),
                    step=0.1,
                    format="%.1f",
                )

            # Pre-select saved community
            comm_options = ["-- Select --", "General / OC", "OBC / BC", "MBC", "SC", "ST", "PWD", "EWS"]
            comm_reverse = {"general": "General / OC", "obc": "OBC / BC", "mbc": "MBC",
                            "sc": "SC", "st": "ST", "pwd": "PWD", "ews": "EWS"}
            saved_comm = comm_reverse.get(profile.get("community", ""), "-- Select --")
            comm_idx = comm_options.index(saved_comm) if saved_comm in comm_options else 0
            community_inp = st.selectbox("Community *", options=comm_options, index=comm_idx)

            branch_options = [
                "-- Any Branch --",
                "Computer Science and Engineering",
                "Electronics and Communication Engineering",
                "Electrical and Electronics Engineering",
                "Mechanical Engineering",
                "Information Technology",
                "Artificial Intelligence and Machine Learning",
                "Artificial Intelligence and Data Science",
                "Civil Engineering", "Chemical Engineering",
                "Biotechnology", "Data Science",
            ]
            saved_branch = profile.get("preferred_branch", "-- Any Branch --")
            branch_idx = branch_options.index(saved_branch) if saved_branch in branch_options else 0
            branch_inp = st.selectbox("Preferred Branch", options=branch_options, index=branch_idx)

            city_options = [
                "-- Any City --", "Chennai", "Coimbatore", "Madurai",
                "Tiruchirappalli", "Vellore", "Salem", "Thanjavur",
                "Erode", "Tirunelveli", "Namakkal", "Karur",
            ]
            saved_city = profile.get("preferred_city", "-- Any City --")
            city_idx = city_options.index(saved_city) if saved_city in city_options else 0
            city_inp = st.selectbox("Preferred City", options=city_options, index=city_idx)

            max_fee_inp = st.number_input(
                "Max Annual Fee (Rs.)",
                min_value=0, max_value=2000000,
                value=int(profile.get("max_fee", 0)),
                step=10000,
                help="0 = no limit",
            )

            submitted = st.form_submit_button("✅ Save Profile", use_container_width=True)

        if submitted:
            new_profile = {}
            if tnea_rank_inp and tnea_rank_inp > 1:
                new_profile["tnea_rank"] = int(tnea_rank_inp)
            if pct_inp and pct_inp > 0:
                new_profile["twelfth_percent"] = float(pct_inp)
            if community_inp != "-- Select --":
                comm_map = {
                    "General / OC": "general", "OBC / BC": "obc", "MBC": "mbc",
                    "SC": "sc", "ST": "st", "PWD": "pwd", "EWS": "ews",
                }
                new_profile["community"] = comm_map.get(community_inp, "general")
            if branch_inp != "-- Any Branch --":
                new_profile["preferred_branch"] = branch_inp
            if city_inp != "-- Any City --":
                new_profile["preferred_city"] = city_inp
            if max_fee_inp and max_fee_inp > 0:
                new_profile["max_fee"] = int(max_fee_inp)

            st.session_state.student_profile = new_profile
            st.session_state.profile_editing = False  # collapse form, show card
            st.rerun()


    # ── Colleges explored ──────────────────────────────────────────────────
    if st.session_state.colleges_shown:
        st.markdown("---")
        st.markdown("<div class='sidebar-title'>🏛️ Colleges Explored</div>", unsafe_allow_html=True)
        for c in st.session_state.colleges_shown[-6:]:
            st.markdown(
                f"<div style='font-size:0.8rem;color:#c4b5fd;padding:0.25rem 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.06);'>• {c}</div>",
                unsafe_allow_html=True
            )

    # ── Reset ──────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button(" Start New Chat", use_container_width=True):
        for key in ["messages", "history", "student_profile", "colleges_shown", "rag_context"]:
            st.session_state[key] = [] if key in ("messages", "history", "colleges_shown") else (
                {} if key == "student_profile" else ""
            )
        st.rerun()

    # ── Quick links ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='sidebar-title'>🔗 Quick Links</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem;'>
    <a href='https://tneaonline.org' target='_blank'
       style='color:#60a5fa;text-decoration:none;display:block;padding:0.2rem 0;'>
       📌 TNEA Official Portal
    </a>
    <a href='https://www.nirfindia.org/Rankings/2024/EngineeringRanking.html' target='_blank'
       style='color:#60a5fa;text-decoration:none;display:block;padding:0.2rem 0;'>
       📊 NIRF Engineering Rankings
    </a>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class='header-banner'>
    <div style='font-size:2rem;'>🎓</div>
    <div>
        <h1>Tamil Nadu Engineering College Advisor</h1>
        <p>RAG-powered admissions assistant · 100+ colleges · 2026-27 cycle · Verified data</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load system (with spinner) ────────────────────────────────────────────────
if not st.session_state.graph_loaded:
    with st.spinner("🔄 Loading AI system and vector index... (first load ~30s)"):
        try:
            _graph = load_rag_system()
            st.session_state.graph_loaded = True
            st.session_state.rag_ready = True
        except Exception as e:
            st.error(f"❌ Failed to load system: {e}\n\nMake sure GROQ_API_KEY is set and run `python setup_rag.py` first.")
            st.stop()

# ── Welcome message ───────────────────────────────────────────────────────────
WELCOME = (
    "👋 **Welcome! I'm your Tamil Nadu Engineering College Admissions Advisor.**\n\n"
    "I can help you:\n"
    "- 🔍 Find colleges you're eligible for (based on your rank/marks)\n"
    "- 📋 Get full admission details, documents, and deadlines\n"
    "- ⚖️ Compare colleges side by side\n"
    "- 🏛️ Search colleges by branch, city, or budget\n\n"
    "**To get started, tell me:**\n"
    "• Your **TNEA rank** or **JEE Main rank**\n"
    "• Your **12th percentage**\n"
    "• Your **community** (General/OBC/BC/MBC/SC/ST/PWD)\n\n"
    "*All data is from our verified database of 100+ Tamil Nadu engineering colleges.*"
)

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": WELCOME})

# ── Render chat history ────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='msg-user'><div class='bubble'>{msg['content']}</div></div>",
                unsafe_allow_html=True,
            )
        else:
            content = msg["content"]
            # Render markdown inside bot bubble via st.chat_message for safety
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(content)

# ── Suggestion chips ──────────────────────────────────────────────────────────
SUGGESTIONS = [
    "Show me Government colleges in Chennai",
    "What documents are needed for admission?",
    "Compare VIT Vellore and NIT Trichy",
    "What are the TNEA 2026 deadlines?",
    "Find CSE colleges under ₹1 lakh/year",
    "Colleges with good placements in Coimbatore",
]

if len(st.session_state.messages) < 3:
    st.markdown("<div style='margin-top:0.8rem;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.78rem;color:rgba(255,255,255,0.4);margin-bottom:0.4rem;'>"
        "💡 Try asking:</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, suggestion in enumerate(SUGGESTIONS[:6]):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state._pending_input = suggestion
    st.markdown("</div>", unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
pending = getattr(st.session_state, "_pending_input", None)
if pending:
    del st.session_state._pending_input
    user_input = pending
else:
    user_input = st.chat_input(
        "Ask about colleges, eligibility, deadlines, documents…",
        key="chat_input",
    )

if user_input and st.session_state.rag_ready:
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🤔 Searching verified college database…"):
        try:
            from graph.supervisor import chat as graph_chat

            ai_response, new_history, new_profile, new_colleges, new_context = graph_chat(
                user_input=user_input,
                history=st.session_state.history,
                profile=st.session_state.student_profile,
                colleges_shown=st.session_state.colleges_shown,
                rag_context=st.session_state.rag_context,
            )

            st.session_state.history = new_history
            st.session_state.student_profile = new_profile
            st.session_state.colleges_shown = new_colleges
            st.session_state.rag_context = new_context
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            err_text = str(e)
            err_lower = err_text.lower()
            if "groq_api_key" in err_lower or "api key" in err_lower or "permission denied" in err_lower:
                hint = "Please check your GROQ_API_KEY and try again."
            elif "429" in err_lower or "quota" in err_lower or "rate" in err_lower:
                hint = "Rate limit reached. Wait 30-60 seconds and retry, or switch to a key/project with higher quota."
            elif "deadline" in err_lower or "timeout" in err_lower or "timed out" in err_lower:
                hint = "Request timed out. Please retry with a shorter query."
            else:
                hint = "Please try again. If this keeps happening, share this error text so it can be fixed quickly."
            err_msg = f"⚠️ Something went wrong: `{type(e).__name__}: {e}`\n\n{hint}"
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:2rem;font-size:0.72rem;color:rgba(255,255,255,0.25);'>
    Data from verified TN Engineering College Database · 2026-27 Admissions
    · Powered by Groq Llama 3.1 8B + ChromaDB + LangGraph
</div>
""", unsafe_allow_html=True)
