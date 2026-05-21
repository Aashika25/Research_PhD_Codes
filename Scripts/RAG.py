import os
import re
import textwrap
import tempfile
from pathlib import Path
import streamlit as st
import chromadb
from chromadb.config import Settings
import fitz                        
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / "Config" / ".env"
load_dotenv(dotenv_path=env_path)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LUMEN",
    page_icon="🔍",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS  — dark industrial aesthetic
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #d4d4d8;
}

/* Header */
.rag-header {
    border-left: 4px solid #38bdf8;
    padding: 0.4rem 1rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(90deg, #0f1923 0%, transparent 100%);
}
.rag-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #f0f9ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: #64748b;
    font-size: 0.82rem;
    margin: 0.2rem 0 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* Chunk cards */
.chunk-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.84rem;
    line-height: 1.6;
}
.chunk-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    margin-bottom: 0.4rem;
    letter-spacing: 0.3px;
}
.chunk-score {
    color: #facc15;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
}
.chunk-text {
    color: #cbd5e1;
    font-size: 0.83rem;
}

/* Answer box */
.answer-box {
    background: #0f2039;
    border: 1px solid #1d4ed8;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e0f2fe;
    margin-top: 0.5rem;
}

/* Status pills */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 4px;
}
.pill-pdf  { background:#1e3a5f; color:#7dd3fc; border:1px solid #2563eb; }
.pill-txt  { background:#1a3327; color:#6ee7b7; border:1px solid #059669; }

/* Streamlit overrides */
div[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #334155;
    border-radius: 8px;
    padding: 0.5rem;
}
.stTextInput > div > div > input {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 0.4rem 1.2rem;
    letter-spacing: 0.3px;
}
.stButton > button:hover { background: #2563eb; }

section[data-testid="stSidebar"] {
    background: #080c14;
    border-right: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="rag-header">
  <h1>🔍 LUMEN - Illuminate your documents</h1>
  <p>ChromaDB · all-MiniLM-L6-v2 · Groq(Llama-3.1) &nbsp;|&nbsp; Upload docs → Ask questions</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS  (cached so they load once)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading Llama3 (Groq)…")
def load_llm():
    return Groq(api_key=os.getenv("API_KEY"))

@st.cache_resource(show_spinner="Initialising ChromaDB…")
def get_chroma_client():
    return chromadb.Client()          # in-memory; swap to PersistentClient for disk

embedder = load_embedder()
llm      = load_llm()
client   = get_chroma_client()

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 400   # characters per chunk
CHUNK_OVERLAP = 80


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 60]   # drop tiny tail chunks


def extract_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    doc  = fitz.open(tmp_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    os.unlink(tmp_path)
    return text


def ingest_document(collection, file_name: str, text: str, doc_type: str):
    chunks = chunk_text(text)
    ids    = [f"{file_name}__chunk_{i}" for i in range(len(chunks))]
    embeds = embedder.encode(chunks).tolist()
    metas  = [{"source": file_name, "type": doc_type, "chunk_idx": i}
              for i in range(len(chunks))]
    # Upsert so re-uploads don't error
    collection.upsert(documents=chunks, embeddings=embeds, metadatas=metas, ids=ids)
    return len(chunks)


def query_rag(collection, question: str, top_k: int = 4):
    q_embed = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=top_k)
    docs     = results["documents"][0]
    metas    = results["metadatas"][0]
    dists    = results["distances"][0]

    context  = "\n\n".join(f"[{m['source']}]\n{d}" for d, m in zip(docs, metas))
    prompt = (
    f"Answer the following question using only the context provided. "
    f"Be detailed and complete. Do not ask follow-up questions.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n\nAnswer:"
    )
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer only from the given context. Be detailed. Do not ask follow-up questions."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3,
    )
    raw    = response.choices[0].message.content
    answer = raw.split("Question:")[0].strip()
    return answer, docs, metas, dists


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — UPLOAD & INGEST
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Document Loader")
    st.caption("Upload one PDF and one technical doc (.txt or .md)")

    pdf_file = st.file_uploader("PDF document", type=["pdf"])
    txt_file = st.file_uploader("Technical doc (.txt / .md)", type=["txt", "md"])

    ingest_btn = st.button("⚡ Ingest Documents", use_container_width=True)

    st.markdown("---")
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#475569;line-height:1.8'>
    <b style='color:#64748b'>STACK</b><br>
    Vector DB &nbsp;→ ChromaDB<br>
    Embeddings → MiniLM-L6<br>
    LLM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Llama3-8.1B<br>
    Chunking &nbsp;&nbsp;→ 400 chars / 80 overlap
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "collection" not in st.session_state:
    st.session_state.collection   = None
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
if "history" not in st.session_state:
    st.session_state.history = []

# ═══════════════════════════════════════════════════════════════════
# INGEST
# ═══════════════════════════════════════════════════════════════════
if ingest_btn:
    if not pdf_file and not txt_file:
        st.sidebar.error("Upload at least one document.")
    else:
        # Fresh collection on each ingest
        try:
            client.delete_collection("rag_docs")
        except Exception:
            pass
        collection = client.create_collection("rag_docs")
        ingested   = []

        if pdf_file:
            with st.sidebar:
                with st.spinner(f"Parsing {pdf_file.name}…"):
                    text   = extract_pdf(pdf_file.read())
                    n      = ingest_document(collection, pdf_file.name, text, "PDF")
                    ingested.append((pdf_file.name, "PDF", n))

        if txt_file:
            with st.sidebar:
                with st.spinner(f"Parsing {txt_file.name}…"):
                    text = txt_file.read().decode("utf-8", errors="ignore")
                    n    = ingest_document(collection, txt_file.name, text, "TechDoc")
                    ingested.append((txt_file.name, "TechDoc", n))

        st.session_state.collection    = collection
        st.session_state.ingested_docs = ingested
        st.session_state.history       = []
        st.sidebar.success("✓ Ingestion complete")

# ═══════════════════════════════════════════════════════════════════
# INGESTED DOC STATUS
# ═══════════════════════════════════════════════════════════════════
if st.session_state.ingested_docs:
    cols = st.columns(len(st.session_state.ingested_docs))
    for col, (name, dtype, n) in zip(cols, st.session_state.ingested_docs):
        pill_cls = "pill-pdf" if dtype == "PDF" else "pill-txt"
        col.markdown(
            f'<span class="pill {pill_cls}">{dtype}</span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">{name}</span><br>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#64748b">'
            f'{n} chunks indexed</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# PRESET QUERIES
# ═══════════════════════════════════════════════════════════════════
PRESET_QUERIES = [
    "What are the main topics covered in these documents?",
    "Summarise the key rules or guidelines mentioned.",
    "What technical processes or procedures are described?",
    "What risks or limitations are highlighted?",
]

if st.session_state.collection:
    st.markdown("#### 💬 Ask a Question")
    preset_col, _ = st.columns([3, 1])
    with preset_col:
        preset = st.selectbox("Quick queries (or type your own below)",
                              ["— select —"] + PRESET_QUERIES)

    query_col, btn_col = st.columns([5, 1])
    with query_col:
        user_q = st.text_input(
            "Your question",
            value=preset if preset != "— select —" else "",
            placeholder="Type your question here…",
            label_visibility="collapsed",
        )
    with btn_col:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and user_q.strip():
        with st.spinner("Retrieving chunks & generating answer…"):
            answer, chunks, metas, dists = query_rag(
                st.session_state.collection, user_q, top_k=top_k
            )
        st.session_state.history.append({
            "question": user_q,
            "answer":   answer,
            "chunks":   chunks,
            "metas":    metas,
            "dists":    dists,
        })

# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════
for entry in reversed(st.session_state.history):
    st.markdown(f"**Q: {entry['question']}**")

    st.markdown("---")
    st.write(entry["answer"])

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# EMPTY STATE
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.collection:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#334155">
      <div style="font-size:3rem">📂</div>
      <div style="font-family:IBM Plex Mono,monospace;font-size:1rem;color:#475569;margin-top:1rem">
        Upload a PDF + a technical doc in the sidebar,<br>then click <b>Ingest Documents</b> to begin.
      </div>
      <div style="font-size:0.78rem;color:#334155;margin-top:1rem">
        Suggested: GDPR PDF + PostgreSQL docs .txt
      </div>
    </div>
    """, unsafe_allow_html=True)