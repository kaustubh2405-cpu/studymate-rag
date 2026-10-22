import html
import re

import streamlit as st
from dotenv import load_dotenv

from rag import FALLBACK, MIN_SCORE, TOP_K, StudyMateRAG, get_llm, load_embedder

load_dotenv()
st.set_page_config(page_title="StudyMate RAG", page_icon="📚")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp textarea,
.stApp input, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Public Sans', sans-serif; }

.hero h1 { font-size: 2.7rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.3rem 0 0.4rem; }
.hero p { font-size: 1.1rem; color: #52606D; max-width: 34rem; margin-bottom: 1.6rem; }
.step { border-left: 3px solid #0E6B5C; padding: 2px 0 2px 14px; }
.step .num { font-weight: 700; color: #0E6B5C; }
.step b { display: block; margin: 2px 0; }
.step p { color: #52606D; font-size: 0.92rem; margin: 0; }
.note { color: #52606D; font-size: 0.9rem; margin-top: 1.6rem; }

.pill { display: inline-block; padding: 2px 10px; margin: 0 6px 4px 0; border-radius: 999px;
        background: #E3F1EE; color: #0E6B5C; font-size: 0.82rem; font-weight: 600; }
.src-row { margin-top: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
.src-label { color: #52606D; font-size: 0.85rem; margin-right: 6px; }
.snippet { background: #F7F9FB; border: 1px solid #E1E7ED; border-radius: 6px;
           padding: 10px 12px; font-size: 0.9rem; line-height: 1.6; }
.snippet mark { background: #FFE58A; color: inherit; padding: 0 2px; border-radius: 2px; }

.doc { padding: 10px 12px; border-left: 3px solid #0E6B5C; background: #F1F6F5;
       border-radius: 0 6px 6px 0; margin: 6px 0 10px; }
.doc-name { font-weight: 600; word-break: break-word; }
.doc-meta { color: #52606D; font-size: 0.85rem; }

[data-testid="stChatMessage"] { background: #FFFFFF; border: 1px solid #E1E7ED; border-radius: 10px; }
.st-key-suggestions button { justify-content: flex-start; text-align: left; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_SUGGESTIONS = [
    "What is this document about?",
    "What are the key terms I should know?",
]
STOP = {
    "what", "which", "when", "where", "does", "with", "that", "this", "from", "have", "about",
    "into", "your", "their", "there", "these", "those", "explain", "describe", "tell", "give",
    "define", "mean", "means", "quot", "x27",
}


def highlight(text, query):
    """Escape the passage and wrap words from the question in <mark> like a highlighter."""
    safe = html.escape(text)
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", query)} - STOP
    if words:
        alt = "|".join(sorted(map(re.escape, words), key=len, reverse=True))
        safe = re.sub(rf"\b({alt})\b", r"<mark>\1</mark>", safe, flags=re.IGNORECASE)
    return safe.replace("\n", "<br>")


def chat_to_markdown(messages, doc_name):
    lines = [f"# StudyMate chat: {doc_name}", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"**Q:** {m['content']}", ""]
        else:
            lines.append(f"**A:** {m['content']}")
            pages = sorted({s["page"] for s in m.get("sources", [])})
            if pages:
                lines.append(f"*Sources: pages {', '.join(map(str, pages))}*")
            lines.append("")
    return "\n".join(lines)


def show_sources(chunks, query):
    if not chunks:
        return
    pages = sorted({c["page"] for c in chunks})
    pills = "".join(f'<span class="pill">Page {p}</span>' for p in pages)
    st.markdown(
        f'<div class="src-row"><span class="src-label">Sources</span>{pills}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View the passages used"):
        for c in chunks:
            st.markdown(f"**Page {c['page']}** &nbsp; match {max(c['score'], 0):.0%}")
            st.progress(min(max(c["score"], 0.0), 1.0))
            st.markdown(
                f'<div class="snippet">{highlight(c["text"], query)}</div>',
                unsafe_allow_html=True,
            )


@st.cache_resource(show_spinner="Loading embedding model...")
def embedder():
    return load_embedder()


messages = st.session_state.setdefault("messages", [])
top_k, min_score = TOP_K, MIN_SCORE

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📚 StudyMate")
    provider = st.selectbox("LLM provider", ["groq", "openai", "gemini", "mistral"])
    pdf = st.file_uploader("Upload a PDF", type="pdf")

    if st.button("Process PDF", type="primary", disabled=pdf is None, use_container_width=True):
        with st.status("Processing your PDF...", expanded=True) as status:
            try:
                st.write("Loading the embedding model...")
                embed = embedder()
                st.write("Reading pages and splitting into passages...")
                new_rag = StudyMateRAG(embed, get_llm(provider))
                n = new_rag.ingest(pdf)
                st.write("Preparing suggested questions...")
                try:
                    suggestions = new_rag.suggest_questions(3)
                except Exception:
                    suggestions = []
                st.session_state.update(
                    rag=new_rag,
                    messages=[],
                    suggestions=suggestions,
                    doc={"name": pdf.name, "pages": new_rag.pages, "chunks": n},
                )
                messages = st.session_state.messages
                status.update(label="Ready to answer questions", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error(str(e))

    rag = st.session_state.get("rag")
    if rag:
        doc = st.session_state.doc
        st.markdown(
            f'<div class="doc"><div class="doc-name">{html.escape(doc["name"])}</div>'
            f'<div class="doc-meta">{doc["pages"]} pages, {doc["chunks"]} passages indexed</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Settings"):
            top_k = st.slider("Passages to retrieve", 1, 8, TOP_K)
            min_score = st.slider(
                "Strictness",
                0.10,
                0.60,
                float(MIN_SCORE),
                0.05,
                help="Higher means StudyMate only answers when the document clearly matches your question.",
            )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            messages = st.session_state.messages
        st.download_button(
            "Download chat",
            data=chat_to_markdown(messages, doc["name"]),
            file_name="studymate-chat.md",
            mime="text/markdown",
            disabled=not messages,
            use_container_width=True,
        )

# ---------------------------------------------------------------- main
typed = st.chat_input("Ask a question about your document", disabled=rag is None)
question = st.session_state.pop("pending", None) or typed

if rag is None:
    st.markdown(
        '<div class="hero"><h1>StudyMate</h1>'
        "<p>Ask questions about any PDF and get answers backed by the exact pages they came from.</p></div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Upload", "Add lecture notes, a textbook chapter, or a paper in the sidebar."),
        ("2", "Process", "StudyMate splits it into passages and indexes them for search."),
        ("3", "Ask", "Get answers with page citations and the passages they used."),
    ]
    for col, (num, title, text) in zip(st.columns(3), steps):
        col.markdown(
            f'<div class="step"><span class="num">{num}</span><b>{title}</b><p>{text}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="note">Works best with text-based PDFs. Scanned pages need OCR first.</p>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(f"#### Chatting with **{html.escape(st.session_state.doc['name'])}**")

    for m in messages:
        with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "📚"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m.get("sources", []), m.get("query", ""))

    if not messages and not question:
        with st.container(key="suggestions"):
            st.markdown("**Try asking**")
            for i, q in enumerate(st.session_state.get("suggestions") or DEFAULT_SUGGESTIONS):
                if st.button(q, key=f"sug{i}", use_container_width=True):
                    st.session_state.pending = q
                    st.rerun()

    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📚"):
            try:
                with st.spinner("Searching your document..."):
                    chunks, stream = rag.stream_answer(question, k=top_k, min_score=min_score)
                text = str(st.write_stream(stream)).strip()
                if not text or FALLBACK.lower() in text.lower():
                    text, chunks = FALLBACK, []
                show_sources(chunks, question)
                messages.append(
                    {"role": "assistant", "content": text, "sources": chunks, "query": question}
                )
            except Exception as e:
                messages.pop()
                st.error(f"Could not get an answer: {e}")

    st.caption("Answers come from an AI model reading your document. Check important details against the cited pages.")