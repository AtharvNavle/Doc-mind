# DocMind API — Complete Technical Documentation

This document explains everything about the DocMind API project in depth: what it does, what every technology is and why it was chosen, how every file works, and how all the pieces connect together. It is written to be understandable whether you are revisiting the codebase after weeks or explaining it to someone new.

---

## Table of Contents

1. [What Is DocMind?](#1-what-is-docmind)
2. [What Is RAG (Retrieval-Augmented Generation)?](#2-what-is-rag-retrieval-augmented-generation)
3. [Why RAG Instead of Just Asking the LLM?](#3-why-rag-instead-of-just-asking-the-llm)
4. [Technology Stack — What Each Tool Is and Why We Use It](#4-technology-stack--what-each-tool-is-and-why-we-use-it)
   - [FastAPI](#fastapi)
   - [PostgreSQL](#postgresql)
   - [SQLAlchemy (Async)](#sqlalchemy-async)
   - [Pydantic v2](#pydantic-v2)
   - [ChromaDB](#chromadb)
   - [PyMuPDF (fitz)](#pymupdf-fitz)
   - [Gemini (via OpenAI-compatible SDK)](#gemini-via-openai-compatible-sdk)
   - [pydantic-settings](#pydantic-settings)
5. [The Full Processing Pipeline — Step by Step](#5-the-full-processing-pipeline--step-by-step)
   - [Upload Pipeline (Ingestion)](#upload-pipeline-ingestion)
   - [Query Pipeline (RAG)](#query-pipeline-rag)
6. [What Are Embeddings?](#6-what-are-embeddings)
7. [What Is a Vector Store?](#7-what-is-a-vector-store)
8. [Project Folder Structure](#8-project-folder-structure)
9. [File-by-File Explanation](#9-file-by-file-explanation)
   - [app/config.py](#appconfigpy)
   - [app/database.py](#appdatabasepy)
   - [app/main.py](#appmainpy)
   - [app/models/document.py](#appmodelsdocumentpy)
   - [app/schemas/document.py](#appschemasdocumentpy)
   - [app/utils/file_handler.py](#apputilsfile_handlerpy)
   - [app/services/parser.py](#appservicesparserpy)
   - [app/services/chunker.py](#appserviceschunkerpy)
   - [app/services/embedder.py](#appservicesembedderpy)
   - [app/services/vector_store.py](#appservicesvector_storepy)
   - [app/services/rag.py](#appservicesragpy)
   - [app/routers/documents.py](#approutersdocumentspy)
   - [app/routers/query.py](#approutersquerypy)
10. [API Endpoints Reference](#10-api-endpoints-reference)
11. [Environment Variables (.env)](#11-environment-variables-env)
12. [Data Flow Diagram](#12-data-flow-diagram)
13. [Two Databases — Why We Use Both](#13-two-databases--why-we-use-both)
14. [Background Tasks — Why Processing Is Async](#14-background-tasks--why-processing-is-async)
15. [Error Handling Strategy](#15-error-handling-strategy)
16. [Running the Project Locally](#16-running-the-project-locally)

---

## 1. What Is DocMind?

DocMind is a backend API that allows users to:

1. **Upload a document** — either a PDF file or a plain `.txt` file.
2. **Ask natural language questions** about that document — and receive accurate, context-grounded answers.

For example:
- You upload a 50-page legal contract.
- You ask: "What is the termination clause?"
- DocMind reads only the relevant sections of the contract, constructs a prompt with those sections as context, and sends it to an LLM (Large Language Model) to generate an answer.

This is useful because raw LLMs (like Gemini or GPT) cannot read files you own. They only know what was in their training data. DocMind bridges that gap by retrieving the right context from your own documents and feeding it to the LLM at query time.

---

## 2. What Is RAG (Retrieval-Augmented Generation)?

RAG stands for **Retrieval-Augmented Generation**. It is a design pattern for building AI-powered question answering systems. It has three stages:

### Stage 1: Indexing (done at upload time)
- Take the raw document text.
- Split it into smaller chunks.
- Convert each chunk into a vector (a list of numbers that represents the meaning of that text).
- Store those vectors in a vector database.

### Stage 2: Retrieval (done at query time)
- Take the user's question.
- Convert it into a vector using the same embedding model.
- Find the chunks whose vectors are most similar to the question vector.
- Those chunks are the most semantically relevant parts of the document.

### Stage 3: Generation (done at query time)
- Take those retrieved chunks and assemble them into a context block.
- Pass that context + the original question to an LLM.
- The LLM generates an answer grounded only in that context — not hallucinated from thin air.

The key insight: **the LLM never reads the full document**. It only reads the 3–5 most relevant chunks. This is faster, cheaper, and avoids context-length limits.

---

## 3. Why RAG Instead of Just Asking the LLM?

You might wonder: why not just paste the whole document into the LLM's chat and ask it directly?

There are several hard limitations:

| Problem | Why RAG Solves It |
|---|---|
| LLMs have a token limit (e.g. 128k tokens) | RAG only sends relevant chunks, not the whole doc |
| Larger context = more expensive API calls | RAG sends only 4 chunks, not 100 pages |
| LLMs hallucinate when they don't know | RAG constrains the LLM to only use provided context |
| You own private documents the LLM has never seen | RAG retrieves from your own stored embeddings |
| Latency increases with document size | RAG retrieval + generation is fast regardless of doc size |

So RAG is not just a workaround — it is the correct architecture for document Q&A at scale.

---

## 4. Technology Stack — What Each Tool Is and Why We Use It

### FastAPI

FastAPI is a Python web framework for building APIs. It is chosen here because:

- **Async-first**: FastAPI supports `async def` route handlers, which means the server does not block while waiting for database queries or API calls. It can handle many concurrent requests efficiently.
- **Automatic documentation**: FastAPI auto-generates an interactive `/docs` page (Swagger UI) from your code, so you can test every endpoint from the browser.
- **Type safety via Pydantic**: FastAPI uses Pydantic models for request/response validation. If a request body is malformed, FastAPI returns a clean `422` error automatically.
- **Dependency injection**: FastAPI has a built-in `Depends()` system that makes it easy to inject shared resources like database sessions into route handlers without global state.

### PostgreSQL

PostgreSQL is a relational database. In this project, it stores **document metadata** — information about each uploaded file:

- What is the file called?
- What type is it (PDF or TXT)?
- How large is it?
- What is its current processing status?
- How many chunks was it split into?

Why PostgreSQL and not SQLite or MongoDB?

- **ACID compliance**: PostgreSQL guarantees that writes are consistent. If a background task crashes halfway through updating a status, PostgreSQL will not leave the record in a corrupt state.
- **UUID support**: PostgreSQL has a native UUID column type, which we use as primary keys. UUIDs are safe to generate client-side without risk of collision.
- **Production-ready**: SQLite is not suitable for concurrent writes. MongoDB is schemaless, which makes it harder to enforce the strict column types we need.
- **asyncpg driver**: PostgreSQL works with `asyncpg`, a fast async driver that integrates with SQLAlchemy's async engine for non-blocking database access.

PostgreSQL does **NOT** store the actual document text or vectors. It only stores the metadata (filename, status, chunk count, etc.).

### SQLAlchemy (Async)

SQLAlchemy is a Python ORM (Object-Relational Mapper). An ORM lets you write Python classes and methods instead of raw SQL.

For example, instead of writing:
```sql
SELECT * FROM documents WHERE id = '...';
```
You write:
```python
result = await session.execute(select(Document).where(Document.id == doc_id))
```

We use SQLAlchemy 2.0 with its **async engine** (`create_async_engine`) and **async session** (`AsyncSession`). This is important because synchronous database calls would block the entire event loop in an async FastAPI app, eliminating all the concurrency benefits.

Key concepts used in this project:
- `create_async_engine`: Creates a connection pool to PostgreSQL using `asyncpg`.
- `async_sessionmaker`: A factory that produces async sessions on demand.
- `Mapped` and `mapped_column`: SQLAlchemy 2.0's typed ORM syntax that plays nicely with Python type hints.
- `get_db()` dependency: A FastAPI dependency that yields a session per request and closes it cleanly when the request finishes.

### Pydantic v2

Pydantic is a data validation library. In this project, every API request body and response body is defined as a Pydantic model.

When a request arrives at a FastAPI endpoint, Pydantic:
1. Parses the raw JSON body.
2. Validates each field against the declared types.
3. Raises a `422 Unprocessable Entity` error with detailed field-level messages if anything is wrong.

We use `ConfigDict(from_attributes=True)` on all response schemas. This tells Pydantic: "You can construct this model from a SQLAlchemy ORM object — read its attributes directly instead of expecting a plain dict." This is required for the `DocumentRead` schema, which is built from a `Document` ORM model returned by the database.

### ChromaDB

ChromaDB is an open-source **vector database**. Its entire purpose is to store and search vectors (embeddings).

In this project, ChromaDB stores:
- The text of each chunk from a document.
- The embedding vector for each chunk.
- Metadata (chunk index number).

ChromaDB is run in **persistent local mode** — it saves data to the `./chroma_store/` directory on disk. There is no separate server to run.

Why ChromaDB over other vector stores (Pinecone, Weaviate, FAISS)?

- **Zero infrastructure**: ChromaDB runs in-process as a Python library. No Docker container, no cloud account, no API key needed.
- **Persistent storage**: Unlike FAISS (which is in-memory only), ChromaDB saves embeddings to disk and reloads them on restart.
- **Collection-per-document model**: ChromaDB supports named collections. We create one collection per document, named by the document's UUID. This perfectly isolates each document's chunks and makes deletion clean — delete the collection and all its embeddings are gone.
- **Similarity search built-in**: ChromaDB's `.query()` method takes an embedding vector and returns the N most similar stored vectors, along with distance scores. No manual math needed.

### PyMuPDF (fitz)

PyMuPDF is a Python library for reading PDF files. It is imported as `fitz`.

When a PDF is uploaded, PyMuPDF:
1. Opens the file.
2. Iterates over every page.
3. Calls `page.get_text()` to extract all the text from that page.
4. Returns the combined text as a plain string.

Why PyMuPDF over other PDF libraries (pdfplumber, PyPDF2)?

- **Speed**: PyMuPDF is written in C and is significantly faster than pure-Python alternatives.
- **Accuracy**: It handles complex PDF layouts (multi-column, tables, footnotes) better than simpler parsers.
- **Minimal dependencies**: One install, no system-level dependencies required.

For `.txt` files, we simply call `Path.read_text(encoding="utf-8")` — no library needed.

### Gemini (via OpenAI-compatible SDK)

Google's Gemini models are used for two tasks:

1. **Embedding generation** (`gemini-embedding-001`): Converts a piece of text into a vector of floating point numbers that represents its semantic meaning.
2. **Chat/completion** (`gemini-2.0-flash-lite`): Takes a system prompt + user message and returns a generated text answer.

Importantly, we do **not** use the official Google Gemini Python SDK. Instead, we use the **OpenAI Python SDK** (`from openai import AsyncOpenAI`) but point it at Gemini's OpenAI-compatible API endpoint:

```
https://generativelanguage.googleapis.com/v1beta/openai
```

Google provides this endpoint so that applications built for the OpenAI API can switch to Gemini with minimal code changes. The benefit for us is that we can swap Gemini for any other OpenAI-compatible provider (OpenAI itself, Ollama, Groq, etc.) by just changing two environment variables: `GEMINI_BASE_URL` and the model names.

### pydantic-settings

`pydantic-settings` is an extension of Pydantic that reads configuration from environment variables and `.env` files.

The `Settings` class inherits from `BaseSettings`. When it is instantiated, it automatically:
1. Reads the `.env` file in the project root.
2. Maps environment variable names to class fields (case-insensitive).
3. Validates and type-casts each value.

We wrap `Settings()` in an `@lru_cache` decorator, so the settings object is constructed only once per process and reused on every call to `get_settings()`. This is important in an async app where `get_settings()` may be called thousands of times per second.

---

## 5. The Full Processing Pipeline — Step by Step

### Upload Pipeline (Ingestion)

This happens when you call `POST /documents/upload`.

```
User uploads file
      │
      ▼
validate_upload_file()        ← checks type (pdf/txt) and size limit
      │
      ▼
save_upload_file()            ← saves file as {uuid}.{ext} inside ./uploads/
      │
      ▼
Document() created in DB      ← status = "processing", no chunk_count yet
      │
      ▼
HTTP 201 returned to user     ← { document_id, status: "processing" }
      │
      ▼  (background task starts AFTER response is sent)
      │
extract_text_from_file()      ← PyMuPDF for PDF, read_text() for TXT
      │
      ▼
chunk_text()                  ← splits text into 500-word chunks, 50-word overlap
      │
      ▼
embed_texts()                 ← sends all chunks to Gemini embedding API in one batch
      │
      ▼
add_document_chunks()         ← stores chunks + vectors in ChromaDB collection
      │
      ▼
document.status = "ready"     ← updates PostgreSQL record
document.chunk_count = N
```

The key design decision: the HTTP response is sent **before** processing starts. This is called a non-blocking background task. The user gets the `document_id` immediately and can poll `GET /documents/{id}` to check the status.

### Query Pipeline (RAG)

This happens when you call `POST /query/`.

```
User sends { document_id, question }
      │
      ▼
Validate document exists in PostgreSQL
      │
      ▼
Check status == "ready"       ← returns 422 if still processing or failed
      │
      ▼
embed_text(question)          ← converts the question to a vector
      │
      ▼
search_document_chunks()      ← queries ChromaDB: "find top 4 chunks most similar to this vector"
      │
      ▼
_build_context()              ← formats chunks into: [Chunk 0] text... [Chunk 2] text...
      │
      ▼
Build prompt:
  system: "Answer ONLY from context. Say you don't know if not in context."
  user:   "Context:\n{chunks}\n\nQuestion: {question}"
      │
      ▼
client.chat.completions.create()   ← calls Gemini LLM
      │
      ▼
Return { answer, sources: [{ chunk_text, chunk_index, similarity_score }] }
```

The `sources` array in the response tells the user exactly which parts of the document the answer was derived from. This is crucial for trust and auditability — the user can verify the answer against the original chunks.

---

## 6. What Are Embeddings?

An embedding is a way of turning text into a list of numbers (a vector) such that **similar text produces similar numbers**.

For example:
- "The cat sat on the mat" → `[0.12, -0.34, 0.89, ...]` (768 or 1536 numbers)
- "A feline rested on the rug" → `[0.11, -0.33, 0.91, ...]` (very close numbers)
- "The stock market crashed" → `[-0.54, 0.22, -0.61, ...]` (very different numbers)

The embedding model (`gemini-embedding-001`) has learned from massive amounts of text how to produce these numerical representations such that semantic similarity maps to mathematical closeness.

This is powerful because computers cannot naturally understand that "cat" and "feline" mean the same thing. But after embedding, their vectors will be close in space, so a vector search will treat them as similar.

In DocMind:
- Every document chunk is embedded once at upload time and stored.
- Every user question is embedded at query time.
- We then find which stored chunk vectors are closest to the question vector — those are the most relevant chunks.

The similarity score returned in the response is derived from the distance between the question vector and each chunk vector. A score close to 1.0 means very similar. A score close to 0 means not similar at all.

---

## 7. What Is a Vector Store?

A vector store (or vector database) is a database optimized for storing and searching vectors.

A normal relational database (like PostgreSQL) can find rows by exact matches (`WHERE id = '...'`) or range comparisons (`WHERE size > 1000`). But it cannot efficiently answer: "Which rows have vectors closest to this query vector?"

ChromaDB answers that question efficiently using approximate nearest-neighbor search. When you call `.query(query_embeddings=[...], n_results=4)`, it returns the 4 stored vectors that are most similar to your query vector — even if you have millions of stored vectors.

In DocMind, ChromaDB is organized into **collections** — one per document. Each collection stores:
- A unique ID per chunk (e.g. `{doc_id}-0`, `{doc_id}-1`)
- The original chunk text (as document content)
- The embedding vector for that chunk
- Metadata: the `chunk_index` number

When a document is deleted, we delete its entire ChromaDB collection — this removes all its chunks and vectors in one operation.

---

## 8. Project Folder Structure

```
docmind-api/
├── app/                         # All application source code
│   ├── __init__.py              # Marks app as a Python package
│   ├── main.py                  # FastAPI app creation, middleware, router wiring
│   ├── config.py                # Settings loaded from .env via pydantic-settings
│   ├── database.py              # Async SQLAlchemy engine and session factory
│   │
│   ├── models/                  # SQLAlchemy ORM table definitions
│   │   ├── __init__.py
│   │   └── document.py          # Document table, DocumentStatus enum
│   │
│   ├── schemas/                 # Pydantic request/response models
│   │   ├── __init__.py
│   │   └── document.py          # DocumentRead, QueryRequest, QueryResponse, etc.
│   │
│   ├── routers/                 # FastAPI route handlers (thin layer, no business logic)
│   │   ├── __init__.py
│   │   ├── documents.py         # /documents/* endpoints + background processing
│   │   └── query.py             # /query/ endpoint
│   │
│   ├── services/                # Business logic — one concern per file
│   │   ├── __init__.py
│   │   ├── parser.py            # PDF/TXT text extraction
│   │   ├── chunker.py           # Text splitting with overlap
│   │   ├── embedder.py          # Embedding generation via Gemini
│   │   ├── vector_store.py      # ChromaDB operations (add, search, delete)
│   │   └── rag.py               # Full RAG pipeline orchestration
│   │
│   └── utils/                   # Stateless helper functions
│       ├── __init__.py
│       └── file_handler.py      # Upload validation + disk persistence
│
├── tests/
│   ├── __init__.py
│   └── test_api.py              # (to be built) async integration tests
│
├── uploads/                     # Uploaded files stored here (gitignored)
├── chroma_store/                # ChromaDB persistent data (gitignored)
├── .env                         # Your real secrets (gitignored)
├── .env.example                 # Template showing what .env should contain
├── .gitignore
├── requirements.txt             # Python dependencies
├── Dockerfile                   # (to be built) container definition
├── docker-compose.yml           # (to be built) multi-container orchestration
└── DOCUMENTATION.md             # This file
```

The structure follows a strict separation of concerns:
- **Routers** only handle HTTP: parse input, call a service, return output.
- **Services** contain all the logic: no HTTP knowledge, no direct DB access in most cases.
- **Models** define what PostgreSQL tables look like.
- **Schemas** define what the API's JSON looks like.
- **Utils** are pure helper functions with no side effects tied to the domain.

---

## 9. File-by-File Explanation

### app/config.py

This file defines all runtime configuration for the application.

```python
class Settings(BaseSettings):
    database_url: str
    gemini_api_key: str
    gemini_base_url: str
    embedding_model: str
    chat_model: str
    chroma_persist_dir: str
    upload_dir: str
    max_file_size_mb: int
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    app_env: str
    app_version: str
```

Each field maps to an environment variable of the same name (case-insensitive). For example, the `database_url` field is populated by the `DATABASE_URL` environment variable in `.env`.

The `get_settings()` function is decorated with `@lru_cache(maxsize=1)`. This means the `Settings()` object is created exactly once when the application starts. Every subsequent call to `get_settings()` returns the same cached object — this is important for performance since configuration is accessed in almost every service call.

Default values are provided for all settings, so the app can start even without a `.env` file (though it won't connect to a real database or API).

---

### app/database.py

This file sets up the connection to PostgreSQL using SQLAlchemy's async interface.

```python
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)
```

- `create_async_engine`: Creates an async connection pool. The pool manages multiple database connections so the app does not have to reconnect on every request.
- `echo=False`: Disables SQL query logging to stdout. Set to `True` during debugging to see every query.
- `pool_pre_ping=True`: Before using a connection from the pool, SQLAlchemy sends a lightweight "ping" to verify the connection is still alive. This prevents errors caused by stale connections.

```python
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

- `async_sessionmaker`: A factory that creates `AsyncSession` objects on demand.
- `expire_on_commit=False`: By default, SQLAlchemy marks all ORM objects as "expired" after a commit, meaning accessing their attributes would trigger a new database query. We disable this because FastAPI routers often access model fields after committing — we want to avoid those extra queries.

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
```

This is a FastAPI dependency. Route handlers declare `db: AsyncSession = Depends(get_db)` to receive a fresh session. The `async with` block ensures the session is always closed (and any pending transaction is rolled back) when the request finishes, even if an exception is raised.

---

### app/main.py

This is the entry point of the FastAPI application. It:

1. **Configures logging** globally so all modules can use Python's standard `logging` module and output formatted log lines.
2. **Creates the `FastAPI` app** instance with a title and version from settings.
3. **Adds CORS middleware** — Cross-Origin Resource Sharing allows web browsers on different domains to call the API. The current configuration allows all origins (`"*"`), which is fine for development. In production this should be restricted to your frontend's domain.
4. **Registers routers** — `app.include_router(documents_router)` and `app.include_router(query_router)` attach all the endpoints defined in the router files to the main app.
5. **Defines the `/health` endpoint** — a simple endpoint that returns `{"status": "ok"}`. This is used by load balancers and monitoring systems to check if the service is alive.

---

### app/models/document.py

This file defines the `documents` table in PostgreSQL using SQLAlchemy's ORM.

```python
class DocumentStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
```

`DocumentStatus` is a Python enum. By inheriting from `str`, its values are plain strings. SQLAlchemy stores it in PostgreSQL as an `ENUM` type — a column that only accepts one of these three values. This prevents invalid status strings from ever being written to the database.

```python
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
```

- `id` is a UUID primary key. UUIDs are generated in Python (`default=uuid.uuid4`) rather than the database. This means we know the ID before inserting the row, which simplifies code that needs to reference the ID immediately after creation.
- `filename`: The UUID-based name we assign when saving the file (e.g. `abc123.pdf`). We rename the file to avoid collisions and special characters.
- `original_name`: The file's original name as uploaded by the user (e.g. `My Contract.pdf`). Stored for display purposes.
- `file_type`: Either `"pdf"` or `"txt"`.
- `file_size`: Size in bytes, stored as an integer.
- `status`: The `DocumentStatus` enum value.
- `chunk_count`: Nullable — it is `NULL` until processing completes, then set to the number of chunks created.
- `created_at` / `updated_at`: Timestamps set and maintained by the database server (`server_default=func.now()`, `onupdate=func.now()`).

---

### app/schemas/document.py

This file defines all Pydantic models used for API input and output.

**`DocumentCreate`** — used internally when constructing a document before saving to the DB. Not exposed directly as an endpoint body.

**`DocumentRead`** — the shape of data returned whenever a document is read from the API. Note it includes `id`, `status`, `chunk_count`, and timestamps — things the user cannot set, only read.

The `model_config = ConfigDict(from_attributes=True)` setting on every schema tells Pydantic: "you may receive a SQLAlchemy ORM object instead of a dict — read its Python attributes directly." Without this, trying to construct `DocumentRead` from a `Document` ORM instance would fail.

**`QueryRequest`** — the body sent by the user to `POST /query/`:
```json
{
  "document_id": "uuid-here",
  "question": "What is the main topic?"
}
```
The `question` field has `Field(min_length=1)` — an empty string question is rejected with 422 before the handler even runs.

**`QuerySource`** — represents a single retrieved chunk returned alongside the answer:
```json
{
  "chunk_text": "The defendant agrees to...",
  "chunk_index": 3,
  "similarity_score": 0.9123
}
```

**`QueryResponse`** — the full response from `POST /query/`:
```json
{
  "answer": "The termination clause states...",
  "sources": [...]
}
```

---

### app/utils/file_handler.py

This utility module handles everything related to uploaded files before they are processed.

**`validate_upload_file(upload_file)`**:
- Checks that the file has a name.
- Checks that the file extension is `pdf` or `txt`.
- Reads the file content to measure its byte size.
- Rewinds the file cursor back to the start (`await upload_file.seek(0)`) so subsequent reads work correctly.
- Checks the size against `MAX_FILE_SIZE_MB` from settings.
- Returns `(file_type, file_size)`.

**`save_upload_file(upload_file)`**:
- Generates a new UUID-based filename (e.g. `550e8400.txt`) to avoid collisions and sanitize paths.
- Creates the `uploads/` directory if it does not exist (`mkdir(parents=True, exist_ok=True)`).
- Reads the file content and writes it to disk as bytes.
- Returns `(stored_filename, full_file_path)`.

Note that `validate_upload_file` is always called before `save_upload_file` in the router. The file is only saved to disk if it passes all validation checks.

---

### app/services/parser.py

This service handles extracting raw text from a saved document file.

**`extract_text_from_file(file_path, file_type)`**:
- Checks the file exists on disk.
- For `pdf`: calls `_extract_text_from_pdf()` which opens the file with PyMuPDF, iterates every page, calls `page.get_text()`, and joins all page texts with newlines.
- For `txt`: calls `_extract_text_from_txt()` which reads the file as UTF-8 text.
- Raises `ValueError` if no text was extracted (e.g. a scanned image-only PDF).

This function is `async` even though it performs synchronous I/O (file reading). This is intentional for consistency — in a future version, the PDF parsing could be moved to a thread pool executor to avoid blocking the event loop on large files.

---

### app/services/chunker.py

This service splits extracted text into overlapping chunks.

**`chunk_text(text, chunk_size, chunk_overlap)`**:

The text is split by whitespace into a list of "tokens" (words). Then a sliding window moves through this list:

- Window size: `chunk_size` (default 500 words)
- Step size: `chunk_size - chunk_overlap` (default 500 - 50 = 450 words)

So chunk 0 contains words 0–499, chunk 1 contains words 450–949, chunk 2 contains words 900–1399, and so on.

The 50-word overlap ensures that content near a chunk boundary is not lost. If a sentence spans the boundary between chunk 1 and chunk 2, the overlapping region means both chunks contain enough context for the sentence to be meaningful.

Why word-based rather than character-based or token-based?
- Word splitting is simple, fast, and language-agnostic.
- Tokens (in the LLM sense) average about 0.75 words each, so 500 words ≈ 375 tokens — well within embedding model limits.

---

### app/services/embedder.py

This service generates vector embeddings for text using the Gemini embedding model.

```python
client = AsyncOpenAI(
    api_key=settings.gemini_api_key,
    base_url=settings.gemini_base_url,
)
```

The OpenAI SDK client is initialized at module level (once, when the module is imported). The `base_url` points to Gemini's OpenAI-compatible endpoint, so every API call is routed to Google's servers instead of OpenAI's.

**`embed_text(text)`**: Embeds a single string. Used for embedding the user's question at query time.

**`embed_texts(texts)`**: Embeds a list of strings in one API call. Used for embedding all chunks at upload time. Batching is important — sending 50 chunks in one request is far more efficient than making 50 separate API calls.

Both functions include a `NotFoundError` fallback: if the configured embedding model name is rejected by the API, they retry with a hardcoded fallback model (`gemini-embedding-001`). This prevents the entire background task from failing just because of a wrong model name in the config.

---

### app/services/vector_store.py

This service manages all interactions with ChromaDB.

```python
client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
```

`PersistentClient` saves all data to the `./chroma_store/` directory. When the app restarts, all previously stored collections and embeddings are available immediately without re-processing.

**`get_or_create_document_collection(document_id)`**: Returns the ChromaDB collection named after the document's UUID. If the collection does not exist, it is created. This is safe to call multiple times — it will not duplicate the collection.

**`add_document_chunks(document_id, chunks, embeddings)`**:
- Takes the list of chunk texts and their corresponding embedding vectors.
- Assigns each chunk a unique ID: `{document_id}-0`, `{document_id}-1`, etc.
- Stores metadata `{"chunk_index": idx}` alongside each chunk so we can return `chunk_index` in query results.
- Calls `collection.add()` to persist everything.

**`search_document_chunks(document_id, query_embedding, top_k)`**:
- Calls `collection.query()` with the question's embedding vector.
- ChromaDB returns the `top_k` most similar stored chunks, along with their distances.
- We convert the raw `distance` value into a `similarity_score` using `1.0 / (1.0 + distance)`. This maps distance 0 (identical) to score 1.0, and large distances to scores near 0.

**`delete_document_collection(document_id)`**:
- Checks if the collection exists before deleting (to avoid errors on collections that never got created due to processing failure).
- Deletes the entire collection, removing all stored chunks and embeddings for that document.

---

### app/services/rag.py

This is the orchestration service that ties together the query pipeline. It is the most important service.

```python
CHAT_MODEL_FALLBACKS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
```

A fallback list of models is tried in order if the configured model is not available on the API endpoint. This makes the service resilient to model name changes.

**`_build_context(chunks)`**: Formats the retrieved chunks into a numbered context block:
```
[Chunk 0] The defendant shall have 30 days to...

[Chunk 3] The termination clause applies when...
```

**`run_rag_query(document_id, question)`**:
1. Embeds the question using `embed_text()`.
2. Searches ChromaDB for the top-K most similar chunks using `search_document_chunks()`.
3. Builds the context string.
4. Constructs a two-part prompt:
   - **System message**: Instructs the model to answer only from context and admit when it doesn't know.
   - **User message**: The context block followed by the question.
5. Calls the Gemini chat API with `temperature=0` — zero temperature means the model always picks the most likely next token, producing deterministic and factual (rather than creative) responses.
6. Returns a `QueryResponse` with the answer text and the source chunks.

The system prompt `"Answer ONLY using the provided context"` is what prevents hallucination — the LLM is explicitly told not to draw on its training data.

---

### app/routers/documents.py

This is the HTTP layer for document management. It handles four endpoints.

**`process_document_background(document_id)`** — this is the background task function, not an endpoint. It:
- Opens its **own** database session using `AsyncSessionLocal()` directly (not `get_db()`). This is necessary because background tasks run after the HTTP response has been sent, so the request-scoped session from `get_db()` has already been closed.
- Fetches the document from the DB.
- Calls `extract_text_from_file()`, `chunk_text()`, `embed_texts()`, `add_document_chunks()` in sequence.
- Sets `status = "ready"` and `chunk_count` on success.
- Sets `status = "failed"` on any exception, so the user can see something went wrong by polling the status endpoint.

**`POST /documents/upload`**:
- Calls `validate_upload_file()` and `save_upload_file()`.
- Creates a `Document` ORM object and commits it to the DB.
- Registers `process_document_background` as a background task using `background_tasks.add_task()`.
- Returns `201 Created` with `{ document_id, status: "processing" }` immediately.

**`GET /documents/`**: Lists all documents ordered by `created_at` descending (newest first).

**`GET /documents/{doc_id}`**: Fetches one document by UUID. Returns `404` if not found.

**`DELETE /documents/{doc_id}`**:
1. Finds the document or returns `404`.
2. Deletes the file from disk if it exists.
3. Calls `delete_document_collection()` to remove its ChromaDB vectors.
4. Deletes the PostgreSQL record.
5. Returns `{ "detail": "Document deleted successfully." }`.

The deletion order matters: disk first, then ChromaDB, then PostgreSQL. This ensures that even if a step fails midway, there is no orphaned PostgreSQL record pointing to a non-existent file.

---

### app/routers/query.py

This is the HTTP layer for RAG queries. It has one endpoint.

**`POST /query/`**:
- Receives `{ document_id, question }` as JSON body (validated by `QueryRequest`).
- Fetches the document from PostgreSQL.
- Returns `404` if the document does not exist.
- Returns `422` if the document's status is not `"ready"` — you cannot query a document that is still being processed or that failed.
- Calls `run_rag_query()` and returns the result directly.

Keeping this endpoint lean (just validation + delegation) is intentional. All the actual logic is in `rag.py` where it can be tested independently.

---

## 10. API Endpoints Reference

| Method | Path | Description | Success Code |
|--------|------|-------------|--------------|
| `GET` | `/health` | Liveness check | `200` |
| `POST` | `/documents/upload` | Upload a PDF or TXT file | `201` |
| `GET` | `/documents/` | List all documents | `200` |
| `GET` | `/documents/{doc_id}` | Get one document by UUID | `200` |
| `DELETE` | `/documents/{doc_id}` | Delete document + vectors + file | `200` |
| `POST` | `/query/` | Ask a question about a document | `200` |

### POST /documents/upload

**Request**: `multipart/form-data` with a `file` field.

**Response (201)**:
```json
{
  "document_id": "a7503799-8985-4ba0-9ff2-a4e52d966550",
  "status": "processing"
}
```

### GET /documents/{doc_id}

**Response (200)**:
```json
{
  "id": "a7503799-8985-4ba0-9ff2-a4e52d966550",
  "filename": "4a99603c-4270-46a4-b420-491d95ffd788.txt",
  "original_name": "my_document.txt",
  "file_type": "txt",
  "file_size": 4096,
  "status": "ready",
  "chunk_count": 12,
  "created_at": "2026-04-23T22:55:50.463Z",
  "updated_at": "2026-04-23T22:55:52.170Z"
}
```

### POST /query/

**Request**:
```json
{
  "document_id": "a7503799-8985-4ba0-9ff2-a4e52d966550",
  "question": "What is the termination clause?"
}
```

**Response (200)**:
```json
{
  "answer": "The termination clause states that either party may terminate the agreement with 30 days written notice.",
  "sources": [
    {
      "chunk_text": "Either party may terminate this agreement upon 30 days written notice to the other party...",
      "chunk_index": 3,
      "similarity_score": 0.9341
    },
    {
      "chunk_text": "Termination shall become effective upon the expiration of the notice period...",
      "chunk_index": 4,
      "similarity_score": 0.8912
    }
  ]
}
```

---

## 11. Environment Variables (.env)

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string with asyncpg driver | `postgresql+asyncpg://postgres:password@localhost:5432/docmind` |
| `GEMINI_API_KEY` | Your Google AI Studio API key | `AIzaSy...` |
| `GEMINI_BASE_URL` | OpenAI-compatible Gemini endpoint | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `EMBEDDING_MODEL` | Model name for generating embeddings | `gemini-embedding-001` |
| `CHAT_MODEL` | Model name for generating answers | `gemini-2.0-flash-lite` |
| `CHROMA_PERSIST_DIR` | Directory where ChromaDB saves data | `./chroma_store` |
| `UPLOAD_DIR` | Directory where uploaded files are saved | `./uploads` |
| `MAX_FILE_SIZE_MB` | Maximum allowed upload size in megabytes | `20` |
| `CHUNK_SIZE` | Number of words per chunk | `500` |
| `CHUNK_OVERLAP` | Number of overlapping words between chunks | `50` |
| `TOP_K_RESULTS` | Number of chunks to retrieve per query | `4` |
| `APP_ENV` | Environment name for logging/behavior | `development` |

---

## 12. Data Flow Diagram

```
                          UPLOAD FLOW
┌─────────┐    POST /upload    ┌─────────────────┐
│  User   │ ─────────────────► │   FastAPI App   │
└─────────┘                    └────────┬────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │     file_handler.py        │
                          │   validate + save to disk  │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │       PostgreSQL            │
                          │   INSERT document row       │
                          │   status = "processing"     │
                          └─────────────────────────────┘
                                        │ (background task)
                          ┌─────────────▼──────────────┐
                          │        parser.py            │
                          │   extract text from file    │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │        chunker.py           │
                          │   split into 500-word chunks│
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │        embedder.py          │
                          │   call Gemini Embeddings API│
                          │   get vectors for all chunks│
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │      vector_store.py        │
                          │   store chunks + vectors    │
                          │   in ChromaDB collection    │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │       PostgreSQL            │
                          │   UPDATE status = "ready"   │
                          │   UPDATE chunk_count = N    │
                          └─────────────────────────────┘


                           QUERY FLOW
┌─────────┐    POST /query/    ┌─────────────────┐
│  User   │ ─────────────────► │   FastAPI App   │
└─────────┘                    └────────┬────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │       PostgreSQL            │
                          │   check document is "ready" │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │        embedder.py          │
                          │   embed the question        │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │      vector_store.py        │
                          │   search ChromaDB for       │
                          │   top 4 similar chunks      │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │          rag.py             │
                          │   build context prompt      │
                          │   call Gemini Chat API      │
                          │   return answer + sources   │
                          └─────────────┬──────────────┘
                                        │
┌─────────┐                ┌────────────▼───────────────┐
│  User   │ ◄────────────── │  { answer, sources }      │
└─────────┘                └────────────────────────────┘
```

---

## 13. Two Databases — Why We Use Both

It is a common question: why use PostgreSQL *and* ChromaDB? Why not just one?

They serve fundamentally different purposes:

| | PostgreSQL | ChromaDB |
|---|---|---|
| **What it stores** | Structured metadata (filename, status, size, timestamps) | Unstructured vectors + raw text chunks |
| **How it searches** | Exact matches, filters, ranges, joins | Semantic similarity (nearest neighbor) |
| **What it is good at** | "Find the document with ID X" | "Find the 4 chunks most similar to this question" |
| **Persistence** | Disk (full ACID transactions) | Disk (append-only, optimized for vectors) |
| **When it is written** | At upload time (metadata) and after processing (status update) | During background processing (chunk storage) |
| **When it is read** | On every list/get/query request | On every query request |
| **When it is deleted** | On DELETE endpoint | On DELETE endpoint |

PostgreSQL and ChromaDB are always kept in sync:
- When a document is created in PostgreSQL, a ChromaDB collection will be created during processing.
- When a document is deleted from PostgreSQL, its ChromaDB collection is deleted first.

If you tried to store the chunks in PostgreSQL, you would need to manually implement vector similarity search — which is possible with the `pgvector` extension but adds complexity. ChromaDB handles this out of the box with zero configuration.

---

## 14. Background Tasks — Why Processing Is Async

When a user uploads a document, the processing pipeline (parse → chunk → embed → store) could take anywhere from 1 second (small TXT) to 60+ seconds (large PDF with many chunks, each requiring an API call).

If we processed the document synchronously inside the upload endpoint, the user would have to wait the full duration before receiving a response. The HTTP connection would stay open. If the client timed out, the process would abort halfway.

Instead, we use FastAPI's `BackgroundTasks`:

```python
background_tasks.add_task(process_document_background, str(document.id))
return {"document_id": ..., "status": "processing"}
```

The response is sent **immediately**. The background task starts running **after** the response is sent. The user receives their `document_id` and can start polling `GET /documents/{id}` to check when `status` changes from `"processing"` to `"ready"` or `"failed"`.

This is also why `process_document_background` creates its own database session directly (`async with AsyncSessionLocal() as session`) instead of using the `get_db()` dependency. By the time the background task runs, the request context (and its injected session) is already closed.

---

## 15. Error Handling Strategy

| Scenario | HTTP Code | Response |
|---|---|---|
| File has wrong extension | `422` | `{"detail": "Only PDF and TXT files are supported."}` |
| File exceeds size limit | `422` | `{"detail": "File exceeds maximum size of 20 MB."}` |
| Document not found | `404` | `{"detail": "Document not found."}` |
| Document still processing | `422` | `{"detail": "Document is not ready for querying."}` |
| Empty question | `422` | Pydantic field validation error |
| Background processing crash | DB status → `"failed"` | Visible via GET /documents/{id} |
| Embedding model not found | Automatic fallback to `gemini-embedding-001` | Logged as warning |
| Chat model not found | Iterates through `CHAT_MODEL_FALLBACKS` | Logged as warning |
| All fallback models fail | `500` | `{"detail": "Internal Server Error"}` |

The general philosophy: surface expected errors as clean JSON with meaningful messages and appropriate HTTP codes. Let unexpected errors bubble up as 500s — they will appear in logs for investigation.

---

## 16. Running the Project Locally

### Prerequisites
- Python 3.11+
- PostgreSQL running locally with a `docmind` database

### Steps

```powershell
# 1. Create the database in PostgreSQL
# (run in psql or your DB client)
CREATE DATABASE docmind;

# 2. Install dependencies
py -m pip install -r requirements.txt

# 3. Create your .env file (copy from .env.example and fill in values)
# At minimum, set:
#   DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@localhost:5432/docmind
#   GEMINI_API_KEY=your_key_here

# 4. Create database tables (run once)
py -c "import asyncio; from app.database import engine; from app.models.document import Base; exec('async def create():\n    async with engine.begin() as conn:\n        await conn.run_sync(Base.metadata.create_all)'); asyncio.run(create())"

# 5. Start the development server
py -m uvicorn app.main:app --reload

# 6. Open the interactive API docs
# http://127.0.0.1:8000/docs
```

### Typical Test Workflow

1. `GET /health` — verify the server is running.
2. `POST /documents/upload` — upload a small `.txt` file.
3. `GET /documents/{doc_id}` — wait until `status` is `"ready"`.
4. `POST /query/` — send a question about the file content.
5. Inspect `answer` and `sources` in the response.
6. `DELETE /documents/{doc_id}` — clean up.

---

*This documentation was written based on the actual source code at the time of writing. All file paths, field names, and model names reflect the live implementation.*
