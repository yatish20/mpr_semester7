
# 📄 RAG PDF Q&A Prototype

This project is a **Retrieval-Augmented Generation (RAG) system** built with **FastAPI**, **ChromaDB**, and **OpenAI GPT models**.  
It allows you to upload PDFs, index their contents into a vector database, and then ask questions against the documents with accurate, source-cited answers.

---

## ✨ Features

- 📂 Upload PDF documents
- 🔎 Automatic text extraction & chunking
- 📚 Store embeddings in **ChromaDB** (persistent vector DB)
- 🤖 Answer user questions with **OpenAI GPT** using retrieved context
- 🔗 Source citation from document chunks
- ⚡ REST API powered by **FastAPI**

---

## 🛠️ Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) - Web API framework  
- [ChromaDB](https://docs.trychroma.com/) - Vector database for embeddings  
- [OpenAI API](https://platform.openai.com/) - Embeddings + LLM for Q&A  
- [pdfplumber](https://github.com/jsvine/pdfplumber) - Extract text from PDFs  
- [Uvicorn](https://www.uvicorn.org/) - ASGI server  

---

## 📂 Project Structure

```

RAG\_MPR/
│── app.py              # Main FastAPI app
│── utils.py            # Helper for text chunking
│── requirements.txt    # Python dependencies
│── chroma\_db/          # Persistent ChromaDB storage
│── README.md           # Documentation

````

---

## ⚡ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file (or export in shell):

```bash
# .env
OPENAI_API_KEY=your_openai_api_key # add your API key here , either on your os.env or hardcode it
CHROMA_DIR=./chroma_db
```

Make sure your key is **never hardcoded** in the codebase.

On Linux/Mac:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

On Windows (Powershell):

```bash
$env:OPENAI_API_KEY="your_openai_api_key"
```

### 5. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server will start at: [http://localhost:8000](http://localhost:8000)

---

## 📡 API Endpoints

### 🔼 Upload PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/your/document.pdf"
```

Response:

```json
{
  "status": "ok",
  "file": "document.pdf",
  "chunks_indexed": 12
}
```

---

### ❓ Query Document

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is sustainable farming?", "k": 4}'
```

Response:

```json
{
  "answer": "Sustainable farming is ... [source: agriculture.pdf | chunk: 2]",
  "retrieved": [
    {
      "source": "agriculture.pdf",
      "chunk_index": 2,
      "snippet": "Sustainable farming is ..."
    }
  ]
}
```

---

## 📖 How It Works

1. **Upload a PDF** → Extracts text using `pdfplumber`
2. **Chunk text** → Breaks into overlapping segments for embedding
3. **Generate embeddings** → Uses `text-embedding-3-small` from OpenAI
4. **Store in ChromaDB** → Persistent vector storage
5. **Query** → User query embedded & compared with DB
6. **Answer generated** → GPT model produces context-aware response with citations

---

## 🛡️ Security Notes

* Do **not** hardcode your API key in code. Always use environment variables.
* Add `__pycache__/` and `.env` to `.gitignore`.
* GitHub will block pushes if secrets are detected.

---

## 🚀 Future Improvements

* Web UI for uploading & querying documents
* Support for scanned PDFs via OCR (e.g., `pytesseract`)
* Multi-file query support
* User authentication & access control

---

## 👨‍💻 Contributors

* **Yatish Shah** (project owner)
* Your teammates (Bharath, Lavanya, Nathan, …)

---

## 📜 License

MIT License

```
