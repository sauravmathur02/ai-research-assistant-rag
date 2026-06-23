import streamlit as st
from src.document_loader import extract_text
from src.ingest import ingest_document
from src.rag import ask_question
from google.api_core.exceptions import ResourceExhausted
from src.vector_store import (
    reset_collection,
    delete_document,
    get_uploaded_documents,
    get_stored_chunks_count,
)

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Tokens ── */
:root {
    --bg-0:   #08080f;
    --bg-1:   #0d0d1a;
    --bg-2:   #12121f;
    --bg-3:   #1a1a2e;
    --glass:  rgba(255,255,255,0.035);
    --glass-h:rgba(255,255,255,0.06);
    --accent: #6366f1;
    --accent2:#8b5cf6;
    --cyan:   #06b6d4;
    --green:  #10b981;
    --amber:  #f59e0b;
    --red:    #ef4444;
    --a-dim:  rgba(99,102,241,0.14);
    --a-glow: rgba(99,102,241,0.35);
    --t1: #f1f5f9;
    --t2: #94a3b8;
    --t3: #64748b;
    --t4: #475569;
    --br: rgba(255,255,255,0.07);
    --br-a:rgba(99,102,241,0.28);
    --r: 12px;
    --r2:16px;
    --sh: 0 4px 24px rgba(0,0,0,0.5);
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-0) !important;
}
[data-testid="stHeader"] { background: transparent !important; border-bottom: 1px solid var(--br); }
[data-testid="stMainBlockContainer"], .main .block-container {
    padding: 1.25rem 2rem 3rem 2rem !important;
    max-width: 100% !important;
}
section[data-testid="stMain"] { background: var(--bg-0) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--br) !important;
}
[data-testid="stSidebar"] section { padding: 1.25rem 1rem !important; }

/* ── Typography ── */
/* NOTE: do NOT apply font-family to span/div globally — it breaks Streamlit's
   Material Icons font (causing "keyboard_arrow_right" to appear as raw text) */
h1,h2,h3,h4,h5,h6 { color: var(--t1) !important; font-family: 'Inter', sans-serif !important; }
p, li { color: var(--t2) !important; font-family: 'Inter', sans-serif !important; }
.stMarkdown p { color: var(--t2) !important; margin-bottom: 0.4rem !important; }
label { color: var(--t2) !important; font-family: 'Inter', sans-serif !important; }
/* Apply Inter broadly but exclude icon-bearing elements */
[data-testid], .stButton, .stTextInput, .stMarkdown, .stMetric {
    font-family: 'Inter', sans-serif;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--glass) !important;
    border: 1px solid var(--br) !important;
    border-radius: var(--r2) !important;
    padding: 4px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: var(--t3) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.1rem !important;
    border: none !important;
    transition: all 0.18s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--t1) !important; background: var(--glass-h) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    box-shadow: 0 2px 14px var(--a-glow) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 14px var(--a-glow) !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px var(--a-glow) !important;
    filter: brightness(1.08) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--glass) !important;
    border: 1px solid var(--br) !important;
    color: var(--t2) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--glass-h) !important;
    border-color: var(--br-a) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--glass) !important;
    border: 1px solid var(--br) !important;
    border-radius: 10px !important;
    color: var(--t1) !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
    caret-color: var(--accent) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--a-dim) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: var(--t4) !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
/* Outer drop zone container */
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.025) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: var(--r2) !important;
    transition: all 0.22s !important;
}
[data-testid="stFileUploader"] > section:hover,
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
    background: rgba(99,102,241,0.06) !important;
}
/* The inner label/section Streamlit wraps drop zone content in */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
/* "Browse files" button — dark themed */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 8px !important;
    color: var(--t2) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    box-shadow: none !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(99,102,241,0.14) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: var(--t1) !important;
    transform: none !important;
}
/* Drop zone instruction text */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--t3) !important;
}
[data-testid="stFileUploader"] svg { fill: var(--t3) !important; }
[data-testid="stFileUploaderFileName"] { color: var(--t2) !important; }

/* ── Remove (✕) buttons — dark background + red X ── */
/* We apply .st-danger-btn via JS since Streamlit doesn't expose button keys to CSS */
.st-danger-btn,
.st-danger-btn:hover {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.22) !important;
    color: #ef4444 !important;
    box-shadow: none !important;
    transform: none !important;
    border-radius: 8px !important;
}
.st-danger-btn:hover {
    background: rgba(239,68,68,0.15) !important;
    border-color: rgba(239,68,68,0.4) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--glass) !important;
    border: 1px solid var(--br) !important;
    border-radius: var(--r) !important;
    padding: 0.9rem 1.1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover { border-color: var(--br-a) !important; }
[data-testid="stMetric"] label {
    color: var(--t3) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] { color: var(--t1) !important; font-size: 1.45rem !important; font-weight: 700 !important; }

/* ── Expanders — target Streamlit's expander only, not bare summary ── */
[data-testid="stExpander"] {
    border-radius: var(--r) !important;
    overflow: hidden !important;
    border: 1px solid var(--br) !important;
}
[data-testid="stExpander"] summary {
    background: var(--glass) !important;
    border-bottom: 1px solid var(--br) !important;
    padding: 0.6rem 1rem !important;
    color: var(--t2) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.18s, border-color 0.18s !important;
    list-style: none !important;
}
[data-testid="stExpander"] summary:hover { background: var(--glass-h) !important; }
[data-testid="stExpander"][open] summary { border-bottom-color: transparent !important; }
[data-testid="stExpander"] > div {
    background: var(--bg-2) !important;
    padding: 1rem !important;
}
/* Custom HTML details/summary (file preview) */
.custom-details {
    background: var(--glass);
    border: 1px solid var(--br);
    border-radius: var(--r);
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.custom-details summary {
    padding: 0.55rem 0.9rem;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--t2);
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.15s;
}
.custom-details summary:hover { background: var(--glass-h); }
.custom-details[open] summary { border-bottom: 1px solid var(--br); }
.custom-details .detail-body {
    padding: 0.75rem 0.9rem;
    font-size: 0.8rem;
    color: var(--t2);
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: var(--glass) !important;
    border: 1px solid var(--br) !important;
    border-radius: 4px 14px 14px 14px !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg,rgba(99,102,241,0.22),rgba(139,92,246,0.18)) !important;
    border-color: var(--br-a) !important;
    border-radius: 14px 4px 14px 14px !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
    border-radius: 50% !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--bg-3) !important;
    border: 1px solid var(--br-a) !important;
    border-radius: 50% !important;
}

/* ── Alerts ── */
[data-testid="stAlertContainer"][data-baseweb="notification"][kind="success"] {
    background: rgba(16,185,129,0.1) !important;
    border-color: var(--green) !important;
    border-radius: var(--r) !important;
}
[data-testid="stAlertContainer"] { border-radius: var(--r) !important; }
.stSuccess, .element-container .stSuccess { background: rgba(16,185,129,0.1) !important; }
.stInfo { background: var(--a-dim) !important; border-color: var(--accent) !important; border-radius: var(--r) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border-radius: var(--r) !important; }
.stError { background: rgba(239,68,68,0.1) !important; border-radius: var(--r) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.55); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden !important; height: 0 !important; }

/* ── Divider ── */
hr { border-color: var(--br) !important; margin: 1rem 0 !important; }

/* ══════════════════════════════════════════════════════════════════
   SIDEBAR — always visible, can't be accidentally collapsed
   ══════════════════════════════════════════════════════════════════ */

/* Force the sidebar to stay on screen regardless of Streamlit's state machine.
   Streamlit uses transform:translateX(-100%) to slide it off; we override that. */
[data-testid="stSidebar"] {
    transform: translateX(0px) !important;
    visibility: visible !important;
    display: flex !important;
    min-width: 240px !important;
    width: 240px !important;
    transition: none !important;
}

/* Drag handle for resizable sidebar */
.sidebar-resizer {
    position: absolute !important;
    top: 0 !important;
    right: 0 !important;
    width: 6px !important;
    height: 100% !important;
    cursor: col-resize !important;
    background: transparent !important;
    z-index: 999999 !important;
    transition: background 0.15s ease !important;
}
.sidebar-resizer:hover, .sidebar-resizer.dragging {
    background: rgba(99, 102, 241, 0.3) !important;
    border-right: 2px solid var(--accent) !important;
}

body.dragging-sidebar, body.dragging-sidebar * {
    cursor: col-resize !important;
    user-select: none !important;
}

/* Hide ALL collapse/close buttons inside the sidebar so it can't be closed */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNavCollapseButton"],
button[title="Close sidebar"],
button[title="Collapse sidebar"],
[data-testid="stSidebar"] > div > div > button,
[data-testid="stSidebarContent"] button[kind="header"] { display: none !important; }

/* The expand tab that appears on the left edge when sidebar is collapsed —
   keep it styled as a visible accent pill so there's always a way back. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 50% !important;
    left: 0 !important;
    transform: translateY(-50%) !important;
    z-index: 9999 !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border-radius: 0 10px 10px 0 !important;
    width: 30px !important;
    height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: 4px 0 18px rgba(99,102,241,0.5) !important;
    border: none !important;
    transition: width 0.18s ease !important;
}
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover { width: 38px !important; }
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    color: white !important; fill: white !important;
}
[data-testid="stSidebarCollapsedControl"] kbd,
[data-testid="collapsedControl"] kbd { display: none !important; }

/* ── Custom components ── */

/* Sidebar brand */
.sb-brand { display:flex; align-items:center; gap:10px; margin-bottom:0.25rem; }
.sb-logo {
    width:38px; height:38px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.1rem;
    box-shadow: 0 0 18px rgba(99,102,241,0.45);
    flex-shrink:0;
}
.sb-name { font-size:0.95rem; font-weight:700; color:var(--t1) !important; line-height:1.2; }
.sb-sub  { font-size:0.62rem; color:var(--t3) !important; font-weight:500; letter-spacing:0.07em; text-transform:uppercase; }

/* Section label */
.sec-label {
    font-size:0.65rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:var(--t3) !important;
    margin:1.1rem 0 0.5rem 0;
    padding-bottom:5px;
    border-bottom:1px solid var(--br);
}

/* Stat row */
.stat-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:5px 0;
    border-bottom:1px solid rgba(255,255,255,0.03);
}
.stat-lbl { font-size:0.76rem; color:var(--t3) !important; }
.stat-val { font-size:0.8rem; font-weight:600; color:var(--t2) !important; }

/* Status dot */
.dot { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; }
.dot-on  { background:var(--green); box-shadow:0 0 6px var(--green); }
.dot-off { background:var(--red); }

/* Doc pill */
.doc-pill {
    display:flex; align-items:center; gap:7px;
    background:rgba(99,102,241,0.08);
    border:1px solid rgba(99,102,241,0.16);
    border-radius:8px;
    padding:5px 9px; margin-bottom:4px;
    font-size:0.77rem; color:#a5b4fc !important;
    word-break:break-all;
}

/* KPI card */
.kpi {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.04));
    border: 1px solid rgba(99,102,241,0.18);
    border-radius:14px;
    padding:1rem 1.15rem;
    position:relative; overflow:hidden;
    transition: all 0.22s ease;
    height:100%;
}
.kpi::before {
    content:'';
    position:absolute; top:0; left:0;
    width:100%; height:2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--cyan));
}
.kpi:hover { border-color:rgba(99,102,241,0.38); transform:translateY(-2px); box-shadow:0 8px 28px rgba(99,102,241,0.14); }
.kpi-icon { font-size:1.3rem; margin-bottom:6px; }
.kpi-label { font-size:0.67rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--t3) !important; margin-bottom:4px; }
.kpi-val   { font-size:1.55rem; font-weight:800; color:var(--t1) !important; line-height:1; }
.kpi-unit  { font-size:0.72rem; color:var(--t3) !important; margin-top:3px; }

/* File card */
.fc {
    background:var(--glass);
    border:1px solid var(--br);
    border-radius:var(--r2);
    padding:0.9rem 1.1rem;
    margin-bottom:0.7rem;
    display:flex; align-items:flex-start; gap:11px;
    transition:all 0.18s;
    cursor:default;
}
.fc:hover { background:var(--glass-h); border-color:var(--br-a); }
.fc-icon { width:38px; height:38px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0; }
.fc-icon-pdf  { background:rgba(239,68,68,0.14); }
.fc-icon-docx { background:rgba(59,130,246,0.14); }
.fc-icon-txt  { background:rgba(16,185,129,0.14); }
.fc-name  { font-size:0.88rem; font-weight:600; color:var(--t1) !important; }
.fc-meta  { font-size:0.75rem; color:var(--t3) !important; margin-top:2px; }
.badge {
    display:inline-block; font-size:0.62rem; font-weight:700;
    padding:2px 7px; border-radius:20px; margin-left:6px;
    text-transform:uppercase; letter-spacing:0.05em;
}
.badge-pdf  { background:rgba(239,68,68,0.18);  color:#f87171 !important; }
.badge-docx { background:rgba(59,130,246,0.18); color:#60a5fa !important; }
.badge-txt  { background:rgba(16,185,129,0.18); color:#34d399 !important; }

/* Source card */
.src-card {
    background:rgba(99,102,241,0.05);
    border:1px solid rgba(99,102,241,0.14);
    border-radius:var(--r);
    padding:0.8rem 0.95rem;
    margin-bottom:0.55rem;
    transition:border-color 0.18s;
}
.src-card:hover { border-color:rgba(99,102,241,0.3); }
.src-head { display:flex; align-items:center; gap:7px; margin-bottom:5px; flex-wrap:wrap; }
.src-file  { font-size:0.8rem; font-weight:600; color:#a5b4fc !important; }
.src-chunk { font-size:0.7rem; color:var(--t3) !important; }
.src-text  { font-size:0.79rem; color:var(--t2) !important; line-height:1.55; }
.sim-badge { margin-left:auto; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:20px; }
.sim-h { background:rgba(16,185,129,0.18); color:#34d399 !important; }
.sim-m { background:rgba(245,158,11,0.18); color:#fbbf24 !important; }
.sim-l { background:rgba(239,68,68,0.18);  color:#f87171 !important; }

/* Empty state */
.empty {
    text-align:center; padding:3.5rem 1rem;
}
.empty-icon  { font-size:2.8rem; margin-bottom:0.8rem; }
.empty-title { font-size:1rem; font-weight:600; color:var(--t3) !important; margin-bottom:0.3rem; }
.empty-desc  { font-size:0.82rem; color:var(--t4) !important; max-width:320px; margin:0 auto; }

/* Comparison banner */
.cmp-banner {
    background:linear-gradient(135deg,rgba(245,158,11,0.1),rgba(239,68,68,0.06));
    border:1px solid rgba(245,158,11,0.25);
    border-radius:var(--r);
    padding:0.75rem 1rem;
    margin-bottom:0.9rem;
    font-size:0.82rem;
    color:#fbbf24 !important;
    display:flex; align-items:center; gap:8px;
}

/* Main page header */
.page-hdr {
    padding:0.25rem 0 1.25rem 0;
    border-bottom:1px solid var(--br);
    margin-bottom:1.5rem;
    display:flex; align-items:center; gap:14px;
}
.page-hdr-logo {
    width:46px; height:46px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.4rem;
    box-shadow:0 0 24px rgba(99,102,241,0.4);
    flex-shrink:0;
}
.page-hdr-t1 {
    font-size:1.6rem; font-weight:800; line-height:1.1;
    background:linear-gradient(135deg,#6366f1,#8b5cf6,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.page-hdr-t2 { font-size:0.8rem; color:var(--t3) !important; margin-top:3px; letter-spacing:0.02em; }

/* Comparison feature card */
.feat-card {
    background:rgba(99,102,241,0.06);
    border:1px solid rgba(99,102,241,0.18);
    border-radius:var(--r2);
    padding:1.2rem;
}
.feat-title { font-weight:700; color:#a5b4fc !important; margin-bottom:0.75rem; font-size:0.9rem; }

/* Keyword chip */
.kw-chip {
    display:inline-block;
    background:rgba(99,102,241,0.14);
    color:#a5b4fc !important;
    padding:3px 10px; border-radius:20px;
    font-size:0.73rem; font-weight:500;
    margin:2px;
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-up { animation: fadeUp 0.28s ease; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<script>
(function applyDynamicStyles() {
    'use strict';

    function fixSidebar() {
        // CSS override handles the transform, but also try clicking the expand btn
        const expandBtn = document.querySelector(
            '[data-testid="stSidebarCollapsedControl"] button, ' +
            '[data-testid="collapsedControl"] button'
        );
        if (expandBtn) expandBtn.click();

        // Direct style override as belt-and-suspenders
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.setProperty('transform', 'translateX(0px)', 'important');
            sidebar.style.setProperty('visibility', 'visible', 'important');
            sidebar.style.setProperty('display', 'flex', 'important');
        }
    }

    function makeSidebarResizable() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return;

        if (sidebar.style.position !== 'relative') {
            sidebar.style.setProperty('position', 'relative', 'important');
        }

        const resizer = sidebar.querySelector('.sidebar-resizer');
        
        // Don't overwrite width while actively dragging to avoid jitter
        if (!resizer || !resizer.classList.contains('dragging')) {
            const savedWidth = localStorage.getItem('sidebarWidth');
            if (savedWidth) {
                const w = parseInt(savedWidth, 10);
                if (w >= 200 && w <= 600) {
                    sidebar.style.setProperty('width', w + 'px', 'important');
                    sidebar.style.setProperty('min-width', w + 'px', 'important');
                    sidebar.style.setProperty('max-width', w + 'px', 'important');
                }
            }
        }

        if (resizer) return;

        // Create resizer element
        const newResizer = document.createElement('div');
        newResizer.className = 'sidebar-resizer';
        sidebar.appendChild(newResizer);

        let startX, startWidth;

        newResizer.addEventListener('mousedown', function(e) {
            startX = e.clientX;
            startWidth = parseInt(document.defaultView.getComputedStyle(sidebar).width, 10);
            newResizer.classList.add('dragging');
            document.body.classList.add('dragging-sidebar');
            
            document.documentElement.addEventListener('mousemove', doDrag, false);
            document.documentElement.addEventListener('mouseup', stopDrag, false);
            e.preventDefault();
        });

        function doDrag(e) {
            let newWidth = startWidth + (e.clientX - startX);
            if (newWidth < 200) newWidth = 200;
            if (newWidth > 600) newWidth = 600;
            
            sidebar.style.setProperty('width', newWidth + 'px', 'important');
            sidebar.style.setProperty('min-width', newWidth + 'px', 'important');
            sidebar.style.setProperty('max-width', newWidth + 'px', 'important');
            localStorage.setItem('sidebarWidth', newWidth);
        }

        function stopDrag(e) {
            newResizer.classList.remove('dragging');
            document.body.classList.remove('dragging-sidebar');
            document.documentElement.removeEventListener('mousemove', doDrag, false);
            document.documentElement.removeEventListener('mouseup', stopDrag, false);
        }
    }

    function styleRemoveButtons() {
        // Tag every button whose text content is exactly '✕' with our danger class
        document.querySelectorAll('button').forEach(function(btn) {
            if (btn.textContent.trim() === '\u2715') {
                btn.classList.add('st-danger-btn');
            }
        });
    }

    function runAll() {
        fixSidebar();
        makeSidebarResizable();
        styleRemoveButtons();
    }

    // Run once immediately (catches elements already in DOM)
    runAll();

    // Watch for DOM changes (Streamlit reruns add new buttons / re-render sidebar)
    const observer = new MutationObserver(function(mutations) {
        // Debounce to avoid hammering on every tiny mutation
        clearTimeout(observer._t);
        observer._t = setTimeout(runAll, 80);
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""",
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

total_chunks_stored = get_stored_chunks_count()
st.session_state.documents_processed = total_chunks_stored > 0
uploaded_docs = get_uploaded_documents()


def _file_meta(name: str):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
    if ext == "pdf":
        return "📄", "fc-icon-pdf", "badge-pdf", "PDF"
    if ext == "docx":
        return "📝", "fc-icon-docx", "badge-docx", "DOCX"
    return "📃", "fc-icon-txt", "badge-txt", "TXT"


def _file_card_html(name: str, size_bytes: int = 0) -> str:
    icon, icon_cls, badge_cls, label = _file_meta(name)
    size_str = f"{round(size_bytes / 1024, 1)} KB" if size_bytes else ""
    return f"""
<div class="fc fade-up">
  <div class="fc-icon {icon_cls}">{icon}</div>
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
      <span class="fc-name">{name}</span>
      <span class="badge {badge_cls}">{label}</span>
    </div>
    <div class="fc-meta">{size_str}</div>
  </div>
</div>"""


def _sim_cls(sim: int) -> str:
    if sim >= 75: return "sim-h"
    if sim >= 45: return "sim-m"
    return "sim-l"


def _kpi(icon: str, label: str, value: str, unit: str = "") -> str:
    return f"""
<div class="kpi fade-up">
  <div class="kpi-icon">{icon}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-val">{value}</div>
  {'<div class="kpi-unit">' + unit + '</div>' if unit else ''}
</div>"""


def _source_card(src: dict) -> str:
    sim = src.get("similarity", 0)
    cls = _sim_cls(sim)
    excerpt = src["content"][:320].replace("<", "&lt;").replace(">", "&gt;")
    if len(src["content"]) > 320:
        excerpt += "…"
    return f"""
<div class="src-card fade-up">
  <div class="src-head">
    <span class="src-file">📄 {src['source']}</span>
    <span class="src-chunk">Chunk #{src['chunk_index']}</span>
    <span class="sim-badge {cls}">{sim}% match</span>
  </div>
  <div class="src-text">{excerpt}</div>
</div>"""


with st.sidebar:

    st.markdown(
        """
<div class="sb-brand">
  <div class="sb-logo">🔬</div>
  <div>
    <div class="sb-name">AI Research</div>
    <div class="sb-sub">AI-Powered Research Assistant</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown('<div class="sec-label">System Status</div>', unsafe_allow_html=True)
    db_status = "Connected" if total_chunks_stored > 0 else "Ready"
    dot_cls = "dot-on" if total_chunks_stored > 0 else "dot-off"
    st.markdown(
        f"""
<div class="stat-row">
  <span class="stat-lbl">🗄️ Database</span>
  <span class="stat-val"><span class="dot {dot_cls}"></span>{db_status}</span>
</div>
<div class="stat-row">
  <span class="stat-lbl">🤖 AI Model</span>
  <span class="stat-val">Gemini 2.5 Flash</span>
</div>
<div class="stat-row">
  <span class="stat-lbl">⚡ Embedding Engine</span>
  <span class="stat-val">MiniLM-L6-v2</span>
</div>
<div class="stat-row">
  <span class="stat-lbl">🔍 Search Engine</span>
  <span class="stat-val">Hybrid Retrieval</span>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-label">Your Library</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Documents", len(uploaded_docs))
    c2.metric("Chunks", total_chunks_stored)

    if uploaded_docs:
        st.markdown('<div class="sec-label">Indexed Files</div>', unsafe_allow_html=True)
        for doc in uploaded_docs:
            icon, _, _, _ = _file_meta(doc)
            st.markdown(
                f'<div class="doc-pill">{icon} <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{doc}</span></div>',
                unsafe_allow_html=True,
            )

    if st.session_state.get("last_retrieval_time") is not None:
        st.markdown('<div class="sec-label">Last Query Metrics</div>', unsafe_allow_html=True)
        ret_t = st.session_state.last_retrieval_time
        gen_t = st.session_state.last_generation_time
        tot_t = st.session_state.last_total_response_time
        n_src = st.session_state.last_sources_count
        stats  = st.session_state.get("last_retrieval_stats", {})
        st.markdown(
            f"""
<div class="stat-row"><span class="stat-lbl">Retrieval</span><span class="stat-val">{ret_t:.3f}s</span></div>
<div class="stat-row"><span class="stat-lbl">Generation</span><span class="stat-val">{gen_t:.3f}s</span></div>
<div class="stat-row"><span class="stat-lbl">Total</span><span class="stat-val">{tot_t:.3f}s</span></div>
<div class="stat-row"><span class="stat-lbl">Chunks Used</span><span class="stat-val">{n_src}</span></div>
<div class="stat-row"><span class="stat-lbl">Docs Involved</span><span class="stat-val">{stats.get('docs_involved_count', 0)}</span></div>""",
            unsafe_allow_html=True,
        )
        if stats.get("comparison_mode"):
            st.markdown(
                '<div class="cmp-banner" style="margin-top:0.6rem;font-size:0.76rem;">⚖️ Comparison mode was active</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Database",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state.processing,
    ):
        reset_collection()
        st.session_state.documents_processed = False
        st.session_state.chat_history = []
        for key in [
            "last_retrieval_time", "last_generation_time",
            "last_total_response_time", "last_sources_count",
            "last_standalone_query", "last_retrieval_stats",
        ]:
            st.session_state.pop(key, None)
        st.success("Database and chat history cleared.")
        st.rerun()

    st.markdown(
        '<p style="font-size:0.65rem;color:#334155;text-align:center;margin-top:1.5rem;">AI Research Assistant · RAG Platform</p>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div class="page-hdr">
  <div class="page-hdr-logo">🔬</div>
  <div>
    <div class="page-hdr-t1">AI Research Assistant</div>
    <div class="page-hdr-t2">AI-Powered Research Assistant &nbsp;·&nbsp; Upload · Analyze · Compare · Summarize</div>
  </div>
</div>""",
    unsafe_allow_html=True,
)


tab_docs, tab_chat, tab_analytics, tab_compare = st.tabs(
    ["📁  Documents", "💬  Research Chat", "📊  Analytics", "⚖️  Compare"]
)


with tab_docs:

    col_up, col_prev = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("### 1. Upload Documents")
        st.markdown(
            '<p style="color:var(--t3);font-size:0.82rem;margin-bottom:1rem;">Upload your source knowledge base (PDF, DOCX, TXT).</p>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "drop",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            disabled=st.session_state.processing,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown(
                f'<p style="color:var(--t2);font-size:0.84rem;margin-bottom:0.75rem;">📎 <b>{len(uploaded_files)}</b> file(s) ready for ingestion</p>',
                unsafe_allow_html=True,
            )
            if st.button(
                "🚀  Index Documents",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing,
            ):
                with st.spinner("Processing documents…"):
                    total_indexed = 0
                    errors = []
                    for f in uploaded_files:
                        try:
                            cnt = ingest_document(f)
                            total_indexed += cnt
                        except Exception as exc:
                            errors.append(f"{f.name}: {exc}")
                    for err in errors:
                        st.error(f"⚠️ {err}")
                    if total_indexed > 0:
                        st.session_state.documents_processed = True
                        st.success(
                            f"✅ Successfully indexed **{total_indexed}** chunks"
                        )
                        st.rerun()

    with col_prev:
        if uploaded_files:
            st.markdown("### 2. Preview")
            for f in uploaded_files:
                st.markdown(_file_card_html(f.name, f.size), unsafe_allow_html=True)
                with st.expander(f"📄 Preview — {f.name}"):
                    try:
                        raw = extract_text(f)
                        st.text_area(
                            "",
                            raw[:2000] if raw else "⚠️ No text could be extracted.",
                            height=160,
                            key=f"prev_{f.name}",
                            label_visibility="collapsed",
                        )
                    except Exception as exc:
                        st.error(f"Preview error: {exc}")
        else:
            st.markdown(
                """
<div class="empty">
  <div class="empty-icon">📂</div>
  <div class="empty-title">No documents selected</div>
  <div class="empty-desc">Upload files via the uploader to view content previews and proceed to ingestion.</div>
</div>""",
                unsafe_allow_html=True,
            )

    if uploaded_docs:
        st.markdown("---")
        st.markdown("### Your Library")
        for doc in uploaded_docs:
            icon, icon_cls, badge_cls, label = _file_meta(doc)
            fc_col, rm_col = st.columns([10, 1])
            with fc_col:
                st.markdown(
                    f"""
<div class="fc" style="margin-bottom:0;">
  <div class="fc-icon {icon_cls}">{icon}</div>
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
      <span class="fc-name">{doc}</span>
      <span class="badge {badge_cls}">{label}</span>
    </div>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )
            with rm_col:
                if st.button(
                    "✕",
                    key=f"rm_{doc}",
                    help=f"Remove {doc} from the index",
                    type="secondary",
                ):
                    delete_document(doc)
                    st.toast(f"🗑️ Removed **{doc}** from the index.")
                    st.rerun()


with tab_chat:

    if not st.session_state.documents_processed:
        st.markdown(
            """
<div class="empty">
  <div class="empty-icon">📚</div>
  <div class="empty-title">Library is empty</div>
  <div class="empty-desc">Ingest your documents in the <b>Documents</b> tab to enable research capabilities.</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        chat_area = st.container(height=490)
        with chat_area:
            if not st.session_state.chat_history:
                st.markdown(
                    """
<div class="empty">
  <div class="empty-icon">💬</div>
  <div class="empty-title">Ready to assist with your research</div>
  <div class="empty-desc">Your documents are indexed and ready. Ask a question, request a summary, or compare files.</div>
</div>""",
                    unsafe_allow_html=True,
                )
            else:
                for chat in st.session_state.chat_history:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(chat["question"])
                    with st.chat_message("assistant", avatar="🔬"):
                        st.markdown(chat["answer"])
                        if chat.get("sources"):
                            st.markdown(
                                f'<p style="font-size:0.75rem;color:var(--t3);margin:0.8rem 0 0.4rem 0;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;">📚 {len(chat["sources"])} Retrieved Sources</p>',
                                unsafe_allow_html=True,
                            )
                            src_html = ""
                            for src in chat["sources"]:
                                sim = src.get("similarity", 0)
                                sim_cls = _sim_cls(sim)
                                full_text = src["content"].replace("<", "&lt;").replace(">", "&gt;")
                                src_html += f"""
<details class="custom-details">
  <summary>📄 <b>{src['source']}</b>&nbsp; Chunk #{src['chunk_index']} &nbsp;<span class="sim-badge {sim_cls}">{sim}% match</span></summary>
  <div class="detail-body">{full_text}</div>
</details>"""
                            st.markdown(src_html, unsafe_allow_html=True)
                        if chat.get("stats", {}).get("comparison_mode"):
                            st.markdown(
                                '<div class="cmp-banner">⚖️ Comparison mode was active</div>',
                                unsafe_allow_html=True,
                            )
                    st.markdown("<div style='margin-bottom:0.3rem;'></div>", unsafe_allow_html=True)

        st.markdown("---")

        if not st.session_state.chat_history:
            st.markdown(
                '<p style="font-size:0.75rem;color:var(--t3);margin-bottom:0.45rem;font-weight:600;">Try asking:</p>',
                unsafe_allow_html=True,
            )
            doc_names = [d.replace(".pdf", "").replace(".docx", "").replace(".txt", "") for d in uploaded_docs]
            EXAMPLE_QS = [
                f"Summarize {doc_names[0]}" if doc_names else "Summarize the main document",
                "What are the key concepts in this document?",
                f"Compare {doc_names[0]} and {doc_names[1]}" if len(doc_names) >= 2 else "What are the main topics covered?",
                "Explain the methodology discussed in the documents",
            ]
            eg_c1, eg_c2 = st.columns(2)
            for idx, eq in enumerate(EXAMPLE_QS):
                with (eg_c1 if idx % 2 == 0 else eg_c2):
                    if st.button(f"💡 {eq}", key=f"eq_{idx}", use_container_width=True):
                        st.session_state["chat_query_input"] = eq
                        st.rerun()
            st.markdown("<div style='margin-bottom:0.3rem;'></div>", unsafe_allow_html=True)

        q_col, btn_col = st.columns([6, 1])
        with q_col:
            question = st.text_input(
                "query",
                placeholder="Ask a question, request a summary, or compare documents…",
                disabled=st.session_state.processing,
                label_visibility="collapsed",
                key="chat_query_input",
            )
        with btn_col:
            send_btn = st.button(
                "Send →",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing,
            )

        if send_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                st.session_state.processing = True
                with st.spinner("Analyzing your documents and generating a response…"):
                    try:
                        answer, sources, ret_t, gen_t, standalone, stats = ask_question(
                            question, st.session_state.chat_history
                        )
                        st.session_state.last_retrieval_time       = ret_t
                        st.session_state.last_generation_time      = gen_t
                        st.session_state.last_total_response_time  = ret_t + gen_t
                        st.session_state.last_sources_count        = len(sources)
                        st.session_state.last_standalone_query     = standalone
                        st.session_state.last_retrieval_stats      = stats
                        st.session_state.chat_history.append(
                            {"question": question, "answer": answer, "sources": sources, "stats": stats}
                        )
                    except ResourceExhausted:
                        st.error("⚠️ API quota exceeded. Please wait a moment and retry.")
                    except Exception as exc:
                        st.error(f"⚠️ Unexpected error: {exc}")
                    finally:
                        st.session_state.processing = False
                        st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()




with tab_analytics:

    st.markdown("### Insights Dashboard")
    st.markdown(
        '<p style="color:var(--t3);font-size:0.83rem;margin-bottom:1.25rem;margin-top:-0.4rem;">Track performance and understand how your queries are processed.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("last_retrieval_time") is None:
        st.markdown(
            """
<div class="empty">
  <div class="empty-icon">📊</div>
  <div class="empty-title">No activity yet</div>
  <div class="empty-desc">Ask a question in the Research Chat tab and your performance insights will appear here.</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        ret_t  = st.session_state.last_retrieval_time
        gen_t  = st.session_state.last_generation_time
        tot_t  = st.session_state.last_total_response_time
        n_src  = st.session_state.last_sources_count
        stats  = st.session_state.get("last_retrieval_stats", {})

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(_kpi("⚡", "Search Speed", f"{ret_t:.3f}", "seconds"), unsafe_allow_html=True)
        with k2:
            st.markdown(_kpi("🤖", "Response Time", f"{gen_t:.3f}", "seconds"), unsafe_allow_html=True)
        with k3:
            st.markdown(_kpi("🕐", "Total Time", f"{tot_t:.3f}", "seconds"), unsafe_allow_html=True)
        with k4:
            st.markdown(_kpi("📝", "Passages Found", str(n_src), "results"), unsafe_allow_html=True)
        with k5:
            st.markdown(_kpi("📂", "Documents Used", str(stats.get("docs_involved_count", 0)), "files"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Query Breakdown")

            if stats.get("comparison_mode"):
                st.markdown(
                    '<div class="cmp-banner">⚖️ Comparison analysis was active for this query.</div>',
                    unsafe_allow_html=True,
                )

            if st.session_state.get("last_standalone_query"):
                st.markdown("**Resolved Question**")
                st.info(st.session_state.last_standalone_query)

            st.markdown("**Passages Used Per Document**")
            cpd = stats.get("chunks_per_doc", {})
            if cpd:
                for doc, cnt in cpd.items():
                    pct = int(cnt / n_src * 100) if n_src else 0
                    st.markdown(
                        f'<div class="stat-row"><span class="stat-lbl">{doc}</span><span class="stat-val">{cnt} passage(s) &nbsp;<span style="color:var(--t3);font-weight:400;">({pct}%)</span></span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<p style="color:var(--t4);font-size:0.82rem;">No breakdown available for this query.</p>', unsafe_allow_html=True)

        with right:
            st.markdown("#### Sources Referenced")
            contrib = stats.get("contributing_docs", [])
            if contrib:
                for doc in contrib:
                    st.markdown(f'<div class="doc-pill">✅ {doc}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:var(--t4);font-size:0.82rem;">No specific document citations in the last response.</p>', unsafe_allow_html=True)

            st.markdown("#### Session Summary")
            st.markdown(
                f"""
<div class="stat-row"><span class="stat-lbl">Questions Asked</span><span class="stat-val">{len(st.session_state.chat_history)}</span></div>
<div class="stat-row"><span class="stat-lbl">Documents in Library</span><span class="stat-val">{len(uploaded_docs)}</span></div>
<div class="stat-row"><span class="stat-lbl">Total Passages Indexed</span><span class="stat-val">{total_chunks_stored}</span></div>
<div class="stat-row"><span class="stat-lbl">Embedding Model</span><span class="stat-val">MiniLM-L6-v2</span></div>
<div class="stat-row"><span class="stat-lbl">Search Strategy</span><span class="stat-val">Hybrid Retrieval</span></div>""",
                unsafe_allow_html=True,
            )


with tab_compare:

    st.markdown("### ⚖️ Compare Documents")
    st.markdown(
        '<p style="color:var(--t2);font-size:0.85rem;margin-bottom:1.25rem;">Easily compare multiple documents to understand key similarities, differences, and insights.</p>',
        unsafe_allow_html=True,
    )

    info_col, query_col = st.columns([1, 1], gap="large")

    with info_col:
        st.markdown(
            """
<div class="feat-card">
  <div class="feat-title">Comparison Features</div>
  <div style="margin-top:0.5rem;">
    <div class="stat-row"><span class="stat-lbl">✅ Compare key findings</span></div>
    <div class="stat-row"><span class="stat-lbl">✅ Compare methodologies</span></div>
    <div class="stat-row"><span class="stat-lbl">✅ Compare conclusions</span></div>
    <div class="stat-row"><span class="stat-lbl">✅ Identify similarities</span></div>
    <div class="stat-row"><span class="stat-lbl">✅ Identify differences</span></div>
    <div class="stat-row"><span class="stat-lbl">✅ Generate structured summaries</span></div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    with query_col:
        st.markdown("**Ask a Comparison Question**")

        if not st.session_state.documents_processed:
            st.markdown(
                """
<div class="empty" style="padding:1.5rem;">
  <div class="empty-icon">📚</div>
  <div class="empty-title">Add documents first</div>
  <div class="empty-desc">Upload at least 2 documents in the Documents tab to start comparing them.</div>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:var(--t3);font-size:0.8rem;margin-bottom:0.5rem;">Examples — click to use:</p>',
                unsafe_allow_html=True,
            )

            SAMPLES = [
                "Compare the key differences between the uploaded documents.",
                "What are the similarities and differences across all files?",
                "How do the methodologies in these documents relate to each other?",
            ]
            for sq in SAMPLES:
                if st.button(f"↗ {sq}", key=f"smp_{sq[:18]}", use_container_width=True):
                    st.session_state["cmp_query_area"] = sq
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            cmp_q = st.text_area(
                "comparison_query",
                placeholder="e.g. Compare the methodologies in Document A vs Document B",
                height=90,
                key="cmp_query_area",
                label_visibility="collapsed",
            )

            if st.button(
                "🔍  Run Comparison",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing,
            ):
                if not cmp_q.strip():
                    st.warning("Please enter a question to compare your documents.")
                else:
                    st.session_state.processing = True
                    with st.spinner("Comparing your documents…"):
                        try:
                            answer, sources, ret_t, gen_t, standalone, stats = ask_question(
                                cmp_q, st.session_state.chat_history
                            )
                            st.session_state.last_retrieval_time      = ret_t
                            st.session_state.last_generation_time     = gen_t
                            st.session_state.last_total_response_time = ret_t + gen_t
                            st.session_state.last_sources_count       = len(sources)
                            st.session_state.last_standalone_query    = standalone
                            st.session_state.last_retrieval_stats     = stats
                            st.session_state.chat_history.append(
                                {"question": cmp_q, "answer": answer, "sources": sources, "stats": stats}
                            )

                            if stats.get("comparison_mode"):
                                st.markdown(
                                    '<div class="cmp-banner fade-up">⚖️ Comparison Mode activated — partitioned multi-document retrieval used</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.info("ℹ️ Comparison mode was not detected. Try adding 'compare', 'difference', or 'versus' to your query.")

                            st.markdown("---")
                            st.markdown("#### Comparison Result")
                            st.markdown(answer)

                            if sources:
                                st.markdown(
                                    f'<p style="font-size:0.75rem;color:var(--t3);margin:0.9rem 0 0.4rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;">📚 {len(sources)} Retrieved Sources</p>',
                                    unsafe_allow_html=True,
                                )
                                for src in sources:
                                    st.markdown(_source_card(src), unsafe_allow_html=True)
                                    with st.expander(f"Full text — {src['source']} (Chunk #{src['chunk_index']})"):
                                        st.markdown(src["content"])

                        except ResourceExhausted:
                            st.error("⚠️ Gemini API quota exceeded. Please wait a few seconds and retry.")
                        except Exception as exc:
                            st.error(f"⚠️ Error: {exc}")
                        finally:
                            st.session_state.processing = False
                            st.session_state.pop("cmp_prefill", None)