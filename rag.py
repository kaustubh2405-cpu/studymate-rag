"""StudyMate RAG core: PDF -> chunks -> embeddings -> ChromaDB -> grounded answer."""
import os
import re
import uuid

import chromadb
from pypdf import PdfReader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
FALLBACK = "I couldn't find the answer to that in the document."
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
# If even the best chunk is less similar than this, we skip the LLM and use the fallback.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))

PROMPT = ChatPromptTemplate.from_template(
    """You are StudyMate, a study assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, reply with exactly: {fallback}
Mention page numbers like (p. 3) when you use them.

Context:
{context}

Question: {question}
Answer:"""
)


SUGGEST_PROMPT = ChatPromptTemplate.from_template(
    """Below are excerpts from a document. Write {n} short, specific study questions that these excerpts can answer.
Return exactly one question per line, with no numbering and no extra text.

Excerpts:
{context}"""
)


def parse_questions(text, n=3):
    """Turn the model's reply into a clean list of questions."""
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if len(line) > 8 and line.endswith("?"):
            questions.append(line)
    return questions[:n]


def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def get_llm(provider=None):
    """Pick an LLM API. Set the matching API key in .env. Override the model with LLM_MODEL."""
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    model = os.getenv("LLM_MODEL")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "llama-3.1-8b-instant", temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash", temperature=0)
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model or "mistral-small-latest", temperature=0)
    raise ValueError(f"Unknown provider: {provider}")


class StudyMateRAG:
    def __init__(self, embedder, llm):
        self.embedder = embedder
        self.llm = llm
        self.pages = 0
        self.n_chunks = 0
        self.chain = PROMPT | llm | StrOutputParser()
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name=f"doc_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
        )

    def ingest(self, pdf_file) -> int:
        """Read the PDF page by page, split into overlapping chunks, embed, store. Returns chunk count."""
        reader = PdfReader(pdf_file)
        self.pages = len(reader.pages)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        texts, metas = [], []
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for chunk in splitter.split_text(text):
                texts.append(chunk)
                metas.append({"page": page_no})
        if not texts:
            raise ValueError("No extractable text found. Scanned PDFs need OCR first.")

        embeddings = self.embedder.encode(
            texts, batch_size=32, normalize_embeddings=True
        ).tolist()
        for i in range(0, len(texts), 500):  # Chroma has a max batch size
            self.collection.add(
                ids=[f"c{j}" for j in range(i, min(i + 500, len(texts)))],
                documents=texts[i : i + 500],
                embeddings=embeddings[i : i + 500],
                metadatas=metas[i : i + 500],
            )
        self.n_chunks = len(texts)
        return len(texts)

    def retrieve(self, question, k=TOP_K):
        """Semantic search: embed the question and return the k most similar chunks."""
        q = self.embedder.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=q, n_results=min(k, self.collection.count())
        )
        return [
            {"text": d, "page": m["page"], "score": 1 - dist}
            for d, m, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def _build_inputs(self, question, k, min_score):
        """Retrieve chunks and build the prompt inputs. Returns (None, []) when retrieval is too weak."""
        chunks = self.retrieve(question, k)
        if not chunks or chunks[0]["score"] < min_score:
            return None, []
        context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
        return {"context": context, "question": question, "fallback": FALLBACK}, chunks

    def answer(self, question, k=TOP_K, min_score=MIN_SCORE):
        """Returns (answer, source_chunks). Sources are empty when the fallback is used."""
        inputs, chunks = self._build_inputs(question, k, min_score)
        if inputs is None:
            return FALLBACK, []
        reply = self.chain.invoke(inputs).strip()
        if FALLBACK.lower() in reply.lower():
            return FALLBACK, []
        return reply, chunks

    def stream_answer(self, question, k=TOP_K, min_score=MIN_SCORE):
        """Returns (source_chunks, token_iterator) so the UI can stream the answer as it is generated."""
        inputs, chunks = self._build_inputs(question, k, min_score)
        if inputs is None:
            return [], iter([FALLBACK])
        return chunks, self.chain.stream(inputs)

    def suggest_questions(self, n=3):
        """Ask the LLM for starter questions based on a spread of passages from the document."""
        docs = self.collection.get(include=["documents"])["documents"]
        if not docs:
            return []
        step = max(1, len(docs) // 6)
        context = "\n\n".join(f"- {d[:500]}" for d in docs[::step][:6])
        reply = (SUGGEST_PROMPT | self.llm | StrOutputParser()).invoke({"n": n, "context": context})
        return parse_questions(reply, n)