"""
Universal Document Q&A Assistant — Agentic RAG
Powered by Groq LLaMA 3.1 + ChromaDB + SentenceTransformers
"""

import os
import streamlit as st
from rag_engine import (
    extract_text_from_pdf,
    chunk_documents,
    get_embedder,
    store_chunks,
    collection_exists,
    get_collection_stats,
    run_rag_agent,
)

# Page config
st.set_page_config(
    page_title="DocuMind — Agentic Document Intelligence",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium Editorial CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --bg-deep:      #0e0c0a;
    --bg-surface:   #15120f;
    --bg-raised:    #1d1915;
    --bg-hover:     #252018;
    --border:       rgba(212,168,96,0.14);
    --border-glow:  rgba(212,168,96,0.38);
    --amber:        #d4a860;
    --amber-bright: #f0c878;
    --amber-dim:    rgba(212,168,96,0.55);
    --cream:        #ede8df;
    --cream-muted:  #9e9488;
    --teal:         #4db8a0;
    --teal-dim:     rgba(77,184,160,0.18);
    --rose:         #e07070;
    --rose-dim:     rgba(224,112,112,0.15);
    --violet:       #a78bfa;
    --violet-dim:   rgba(167,139,250,0.15);
    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'Sora', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Global Reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-deep) !important;
    color: var(--cream) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141008 0%, #0e0c0a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}

/* ── Sidebar brand mark ── */
.brand-mark {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    padding: 0 0.5rem 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.brand-glyph {
    font-family: var(--font-display);
    font-size: 2.2rem;
    line-height: 1;
    color: var(--amber);
    letter-spacing: -1px;
    text-shadow: 0 0 20px rgba(212,168,96,0.3);
}
.brand-text { display: flex; flex-direction: column; line-height: 1.1; }
.brand-name { font-family: var(--font-display); font-size: 1.35rem; color: var(--cream); letter-spacing: 0.02em; }
.brand-sub  { font-size: 0.63rem; color: var(--amber-dim); letter-spacing: 0.18em; text-transform: uppercase; font-weight: 500; }

/* ── Section label ── */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--amber-dim);
    font-weight: 600;
    margin: 1.2rem 0 0.5rem;
    padding-left: 2px;
}

/* ── Status pill ── */
.status-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg-raised);
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.green { background: var(--teal); box-shadow: 0 0 6px var(--teal); animation: pulse-dot 2s ease infinite; }
.status-dot.red   { background: var(--rose); box-shadow: 0 0 6px var(--rose); }
.stat-block { display: flex; flex-direction: column; line-height: 1.45; }
.stat-key { color: var(--cream-muted); font-size: 0.72rem; }
.stat-val { color: var(--cream); font-weight: 500; }

/* ── Sidebar footer ── */
.sidebar-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.68rem;
    color: var(--cream-muted);
    letter-spacing: 0.06em;
    text-align: center;
    line-height: 2;
}
.sidebar-footer span { color: var(--amber); }

/* ── Main hero ── */
.hero-eyebrow {
    font-size: 0.68rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--amber);
    font-weight: 600;
    margin-bottom: 0.75rem;
    animation: fadeUp 0.6s ease both;
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.12;
    color: var(--cream);
    margin-bottom: 0.85rem;
    animation: fadeUp 0.7s 0.08s ease both;
    letter-spacing: -0.02em;
}
.hero-title em { color: var(--amber); font-style: italic; }
.hero-sub {
    font-size: 0.95rem;
    color: var(--cream-muted);
    max-width: 520px;
    line-height: 1.7;
    animation: fadeUp 0.7s 0.16s ease both;
    margin-bottom: 1.5rem;
}

/* ── Ornamental divider ── */
.ornament {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0.5rem 0 1.5rem;
    animation: fadeUp 0.7s 0.24s ease both;
}
.ornament-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
}
.ornament-diamond { width: 6px; height: 6px; background: var(--amber); transform: rotate(45deg); opacity: 0.7; }

/* ── Tag pills ── */
.tag-base {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    font-family: var(--font-mono);
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}
.tag-source { background: rgba(212,168,96,0.1); border: 1px solid rgba(212,168,96,0.28); color: var(--amber-bright); }
.tag-page   { background: var(--teal-dim); border: 1px solid rgba(77,184,160,0.3); color: var(--teal); }
.tag-score  { background: var(--rose-dim); border: 1px solid rgba(224,112,112,0.3); color: var(--rose); }
.tag-agent  { background: var(--violet-dim); border: 1px solid rgba(167,139,250,0.3); color: var(--violet); }

/* ── Source card ── */
.source-card {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 12px 14px;
    margin: 10px 0;
    font-size: 0.82rem;
    color: var(--cream-muted);
    line-height: 1.65;
    font-style: italic;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c49240, #a07030) !important;
    color: #0e0c0a !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(212,168,96,0.2) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(212,168,96,0.35) !important;
}
[data-testid="stTextInput"] input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    border-radius: 7px !important;
    font-family: var(--font-body) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 2px rgba(212,168,96,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    font-family: var(--font-body) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px rgba(212,168,96,0.1) !important;
}
[data-testid="stExpander"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    color: var(--cream-muted) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 4px var(--teal); }
    50%       { box-shadow: 0 0 10px var(--teal); }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-mark">
        <div class="brand-glyph">D</div>
        <div class="brand-text">
            <span class="brand-name">DocuMind</span>
            <span class="brand-sub">Agentic Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db_ready = collection_exists()
    stats = (
        get_collection_stats() if db_ready
        else {"document_count": 0, "chunk_count": 0, "sources": []}
    )

    st.markdown('<div class="section-label">📂 Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Policies, resumes, manuals, reports, research papers — any PDF.",
        label_visibility="collapsed"
    )

    show_steps = st.toggle(
        "Show agent thinking",
        value=True,
        help="Reveals the agent's iteration steps while searching"
    )

    if st.button("⟳  Index Documents", use_container_width=True):
        if not uploaded_files:
            st.error("Upload at least one PDF first.")
        else:
            with st.spinner("Indexing…"):
                try:
                    data_dir = "./data/sample_hr_docs"
                    os.makedirs(data_dir, exist_ok=True)
                    for f in os.listdir(data_dir):
                        fp = os.path.join(data_dir, f)
                        if os.path.isfile(fp) and not f.endswith(".gitkeep"):
                            os.remove(fp)
                    pdf_paths = []
                    for uf in uploaded_files:
                        target = os.path.join(data_dir, uf.name)
                        with open(target, "wb") as fh:
                            fh.write(uf.getbuffer())
                        pdf_paths.append(target)
                    all_pages = []
                    for path in pdf_paths:
                        all_pages.extend(extract_text_from_pdf(path))
                    if not all_pages:
                        st.warning("No extractable text found in PDFs.")
                    else:
                        chunks = chunk_documents(all_pages)
                        embedder = get_embedder()
                        store_chunks(chunks, embedder)
                        st.success(f"Indexed {len(pdf_paths)} doc(s) · {len(chunks)} chunks")
                        st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {str(e)}")

    # API Key
    groq_key = bool(os.getenv("GROQ_API_KEY"))
    if not groq_key:
        st.markdown('<div class="section-label">🔑 API Key</div>', unsafe_allow_html=True)
        ui_key = st.text_input("Groq API Key", type="password",
                               help="Get free key at console.groq.com",
                               label_visibility="collapsed",
                               placeholder="gsk_…")
        if ui_key:
            os.environ["GROQ_API_KEY"] = ui_key
            st.success("Key applied for this session")
            st.rerun()
        else:
            st.caption("No key found — [get one free ↗](https://console.groq.com)")

    # DB Status
    st.markdown('<div class="section-label">📊 Index Status</div>', unsafe_allow_html=True)
    if db_ready:
        st.markdown(
            f"""<div class="status-pill">
                <div class="status-dot green"></div>
                <div class="stat-block">
                    <span class="stat-val">{stats['document_count']} doc(s) &nbsp;·&nbsp; {stats['chunk_count']} chunks</span>
                    <span class="stat-key">Vector store ready</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )
        if stats["sources"]:
            with st.expander("Indexed files", expanded=False):
                for src in stats["sources"]:
                    st.caption(f"✓ {src}")
    else:
        st.markdown(
            """<div class="status-pill">
                <div class="status-dot red"></div>
                <div class="stat-block">
                    <span class="stat-val">No index yet</span>
                    <span class="stat-key">Upload PDFs and index them</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        """<div class="sidebar-footer">
            Powered by<br>
            <span>Groq LLaMA 3.1</span> · <span>ChromaDB</span> · <span>Agentic RAG</span>
        </div>""",
        unsafe_allow_html=True
    )

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-eyebrow">Document Intelligence · Agentic RAG</div>
<div class="hero-title">Ask anything about<br><em>your documents</em></div>
<div class="hero-sub">
    Upload any PDF — HR policies, research papers, contracts, manuals, resumes —
    and get precise, cited answers. The agent searches, reflects, and rewrites queries
    until it finds the truth.
</div>
<div class="ornament">
    <div class="ornament-line"></div>
    <div class="ornament-diamond"></div>
    <div class="ornament-line"></div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("↳ Sources & Agent trace"):
                if "iterations" in message:
                    st.markdown(
                        f"<span class='tag-base tag-agent'>🔄 {message['iterations']} iteration(s)</span>",
                        unsafe_allow_html=True
                    )
                if "sub_queries" in message and len(message["sub_queries"]) > 1:
                    for q in message["sub_queries"]:
                        st.markdown(
                            f"<span class='tag-base tag-agent'>🔍 {q}</span>",
                            unsafe_allow_html=True
                        )
                st.markdown("<hr style='border-color:var(--border);margin:10px 0;'>",
                            unsafe_allow_html=True)
                for src in message["sources"]:
                    st.markdown(
                        f"<span class='tag-base tag-source'>📄 {src['source']}</span>"
                        f"<span class='tag-base tag-page'>pg. {src['page']}</span>"
                        f"<span class='tag-base tag-score'>{src['score']:.2f}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='source-card'>{src['text'][:220]}…</div>",
                        unsafe_allow_html=True
                    )

# Chat input & agent pipeline
if user_question := st.chat_input("Ask anything about your uploaded documents…"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if not db_ready:
        with st.chat_message("assistant"):
            st.warning("No documents indexed yet. Upload PDFs and click Index Documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Agent searching…"):
                try:
                    result = run_rag_agent(
                        question=user_question,
                        embedder=get_embedder(),
                        show_agent_steps=show_steps
                    )

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("↳ Sources & Agent trace"):
                            st.markdown(
                                f"<span class='tag-base tag-agent'>🔄 {result['iterations']} iteration(s)</span>",
                                unsafe_allow_html=True
                            )
                            if len(result["sub_queries"]) > 1:
                                for q in result["sub_queries"]:
                                    st.markdown(
                                        f"<span class='tag-base tag-agent'>🔍 {q}</span>",
                                        unsafe_allow_html=True
                                    )
                            st.markdown("<hr style='border-color:var(--border);margin:10px 0;'>",
                                        unsafe_allow_html=True)
                            for src in result["sources"]:
                                st.markdown(
                                    f"<span class='tag-base tag-source'>📄 {src['source']}</span>"
                                    f"<span class='tag-base tag-page'>pg. {src['page']}</span>"
                                    f"<span class='tag-base tag-score'>{src['score']:.2f}</span>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"<div class='source-card'>{src['text'][:220]}…</div>",
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "iterations": result["iterations"],
                        "sub_queries": result["sub_queries"]
                    })

                except Exception as e:
                    st.error(f"Agent error: {str(e)}")
