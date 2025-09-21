import os
import io
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import openai
import chromadb
import numpy as np
from utils import chunk_text

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")# PUT YOUR OPENAI API KEY HERE (BHARATH,LAVANYA,NATHAN)
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection_name = "pdf_docs"
try:
    collection = client.get_collection(collection_name)
except Exception:
    collection = client.create_collection(collection_name)

app = FastAPI(title="RAG PDF Q&A Prototype")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    query: str
    k: int = 4


@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF uploads supported in this prototype')

    contents = await file.read()
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        pages = [p.extract_text() or '' for p in pdf.pages]
        text = "\n\n".join(pages).strip()

    if not text:
        raise HTTPException(status_code=400, detail='No extractable text found in PDF (maybe scanned image PDF)')

    chunks = chunk_text(text, chunk_size=1200, overlap=200)

    # Get embeddings via OpenAI (batching)
    embeddings = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        resp = openai.Embeddings.create(input=batch, model="text-embedding-3-small")
        batch_embs = [r['embedding'] for r in resp['data']]
        embeddings.extend(batch_embs)

    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]

    # Add to Chroma (if ids exist already, we replace)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    client.persist()

    return {"status": "ok", "file": file.filename, "chunks_indexed": len(chunks)}


@app.post('/query')
async def query_document(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail='Query is required')

    # embed query
    q_emb = openai.Embeddings.create(
        input=[req.query],
        model="text-embedding-3-small"
    )['data'][0]['embedding']

    # query chroma
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=req.k,
        include=['documents', 'metadatas', 'distances']
    )

    docs = res['documents'][0]
    metadatas = res['metadatas'][0]

    # Build prompt with retrieved context
    context_sections = []
    for m, d in zip(metadatas, docs):
        idx = m.get('chunk_index')
        src = m.get('source')
        header = f"[source: {src} | chunk: {idx}]"
        context_sections.append(header + "\n" + d)

    context = "\n\n---\n\n".join(context_sections)

    system_msg = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "If the answer is not present in the context, reply: 'I don't know from the document.' "
        "Cite sources using the bracketed source tags present in the context."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}\n\nAnswer concisely and cite the source chunk(s)."

    chat_resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )

    answer = chat_resp['choices'][0]['message']['content']

    return {
        "answer": answer,
        "retrieved": [
            {
                "source": m.get('source'),
                "chunk_index": m.get('chunk_index'),
                "snippet": (d[:300] + '...') if len(d) > 300 else d
            }
            for m, d in zip(metadatas, docs)
        ]
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
