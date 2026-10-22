# StudyMate RAG

A PDF question-answering app. Upload notes or a paper, ask questions, and get answers grounded in the document with source-page citations. If the document doesn't contain the answer, it says so instead of guessing.

**Stack:** Python, LangChain, ChromaDB, sentence-transformers, Streamlit, LLM API (Groq / OpenAI / Gemini / Mistral)

## Run it
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key
streamlit run app.py
```

## How it works
1. **Ingest:** `pypdf` extracts text page by page. `RecursiveCharacterTextSplitter` cuts it into 800-character chunks with 150 overlap, keeping each chunk's page number.
2. **Embed:** `all-MiniLM-L6-v2` (sentence-transformers) turns each chunk into a vector.
3. **Store:** vectors go into a ChromaDB collection using cosine distance.
4. **Retrieve:** the question is embedded the same way, and the top 4 most similar chunks are fetched (semantic search).
5. **Generate:** the chunks plus the question go to the LLM through a LangChain prompt that says "answer only from this context."
6. **Fallback:** if the best chunk's similarity is below `MIN_SCORE`, or the LLM says the answer isn't there, the app returns a "not found" message and shows no sources.

## Measure accuracy (for your resume)
Edit `questions.json` with ~10 questions about your PDF (a few answerable, a few not), then:
```bash
python evaluate.py your_notes.pdf questions.json
```
It prints a result like `8/10 correct`.

## Interview talking points
- **Why chunk?** Embedding a whole PDF blurs meaning, and LLMs have context limits. Small chunks retrieve precisely.
- **Why overlap?** So a sentence cut at a chunk boundary still appears whole in one chunk.
- **Why cosine similarity?** It compares the direction of vectors (meaning), not their length.
- **How do you reduce hallucination?** Restrict the prompt to retrieved context, use temperature 0, and fall back when retrieval is weak.
- **Limitations:** scanned PDFs need OCR, tables extract poorly, and the vector store is in-memory so it resets on restart.
