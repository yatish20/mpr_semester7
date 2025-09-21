# ingest_and_query.py
import os
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import openai

# config: set env vars OPENAI_API_KEY, CHROMA_DIR
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 1) simple PDF -> text
def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

# 2) simple splitting: chunk by ~500 tokens (approx 400-800 chars)
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return [c.strip() for c in chunks if c.strip()]

# 3) embeddings: use sentence-transformers locally (you can swap to OpenAI embeddings)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast
def embed_texts(texts: List[str]):
    return embed_model.encode(texts, convert_to_numpy=True)

# 4) Chroma client
client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"
))
collection = None
def get_collection(name="docs"):
    global collection
    if collection is None:
        try:
            collection = client.get_collection(name)
        except Exception:
            collection = client.create_collection(name)
    return collection

# 5) ingest function
def ingest_file(path, doc_id=None):
    text = pdf_to_text(path)
    chunks = chunk_text(text)
    embs = embed_texts(chunks)
    coll = get_collection()
    ids = [f"{os.path.basename(path)}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(path), "chunk_index": i} for i in range(len(chunks))]
    coll.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embs.tolist())

# 6) retrieve + call LLM
def retrieve(query, k=4):
    q_emb = embed_model.encode([query])
    coll = get_collection()
    results = coll.query(query_embeddings=q_emb.tolist(), n_results=k)
    # results: dict with ids, documents, metadatas, distances
    return results

def answer_with_openai(query, retrieved_docs):
    # build prompt
    context = "\n\n---\n\n".join(retrieved_docs)
    prompt = f"""You are an assistant that answers questions only using the context given. If answer is not in context, say "I don't know from the document."
Context:
{context}

Question: {query}

Answer concisely and cite the source chunks (provide chunk index)."""
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # example — swap as needed
        messages=[{"role":"user", "content":prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"]

# example usage
if __name__ == "__main__":
    # ingest a PDF:
    ingest_file("example.pdf")

    # query:
    q = "What is the company's return policy?"
    r = retrieve(q, k=4)
    docs = r["documents"][0]  # list of doc strings
    print("Retrieved docs:")
    for i,d in enumerate(docs):
        print(i, d[:200].replace("\n"," "), "...")
    ans = answer_with_openai(q, docs[:4])
    print("Answer:\n", ans)
