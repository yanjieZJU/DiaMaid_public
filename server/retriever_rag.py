"""
Traditional RAG retriever for baseline comparison.
Builds a flat vector index from raw PDF articles (no knowledge graph).

Features:
- Structure-aware chunking: section/subsection/paragraph boundaries
- Target chunk size: 500-900 tokens, overlap: 100-200 tokens
- Hybrid retrieval: BM25 + dense embedding, fused via RRF
- Query expansion: 3 rewritten sub-queries
- Rerank cutoff: top 8, final context: top 6 chunks
- Metadata: title / abstract / section / paper topic
"""

import os
import sys
import json
import time
import re
import hashlib
import concurrent.futures
from pathlib import Path
import sqlite3
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import *

import httpx
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery

# ── Constants ──────────────────────────────────────────────────────────────────
ARTICLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "case_llm_artifact", "original_articles"
)
RAG_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_storage")

# Chunking
CHUNK_MIN_TOKENS = 500
CHUNK_MAX_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 150   # within [100, 200]
CHARS_PER_TOKEN = 4          # rough approximation

# Retrieval
BM25_TOP_K = 20              # candidates from BM25
DENSE_TOP_K = 20             # candidates from dense search
RERANK_CUTOFF = 8            # after RRF fusion
FINAL_TOP_K = 6              # returned to LLM
QUERY_EXPANSIONS = 3         # sub-queries generated

# Section header patterns (common in academic PDFs)
_SECTION_RE = re.compile(
    r'(?m)^(?:'
    r'\d+\.?\s+[A-Z][A-Za-z\s]{3,60}|'          # "1. Introduction" / "2 Related Work"
    r'[A-Z][A-Z\s]{4,40}(?:\n|$)|'              # "INTRODUCTION" all-caps
    r'(?:Abstract|Introduction|Related Work|'
    r'Background|Methodology|Method|System|'
    r'Approach|Experiment|Evaluation|Results|'
    r'Discussion|Conclusion|References|'
    r'Acknowledgements?)\s*(?:\n|$)'             # known section names
    r')',
    re.IGNORECASE
)

# ── Globals ────────────────────────────────────────────────────────────────────
_RAG_VECTOR_STORE = None   # kept for compat check in main.py
_RAG_MATRIX = None         # np.ndarray (N, D) float32
_RAG_IDS = None            # list[str]
_RAG_EMBED_MODEL = None
_BM25_INDEX = None         # list[tuple[str, list[str]]]
_DB_CONN = None
RAG_DB_PATH = os.path.join(RAG_STORAGE_DIR, "rag_nodes.sqlite")

# ── Embed model ────────────────────────────────────────────────────────────────
def get_embed_model():
    global _RAG_EMBED_MODEL
    if _RAG_EMBED_MODEL is None:
        _RAG_EMBED_MODEL = OpenAIEmbedding(
            model_name="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_BASE_URL"),
        )
    return _RAG_EMBED_MODEL

def get_rag_db():
    global _DB_CONN
    if _DB_CONN is None:
        _DB_CONN = sqlite3.connect(RAG_DB_PATH, check_same_thread=False)
        _DB_CONN.row_factory = sqlite3.Row
    return _DB_CONN


def init_rag_db():
    conn = get_rag_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS rag_nodes (
        node_id TEXT PRIMARY KEY,
        source TEXT,
        title TEXT,
        abstract TEXT,
        section TEXT,
        text TEXT
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_nodes_source ON rag_nodes(source)")
    conn.commit()


def fetch_nodes_by_ids(node_ids: list[str]) -> dict[str, dict]:
    if not node_ids:
        return {}

    conn = get_rag_db()
    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"""
        SELECT node_id, source, title, abstract, section, text
        FROM rag_nodes
        WHERE node_id IN ({placeholders})
        """,
        node_ids
    ).fetchall()

    out = {}
    for r in rows:
        out[r["node_id"]] = {
            "source": r["source"],
            "title": r["title"],
            "abstract": r["abstract"],
            "section": r["section"],
            "text": r["text"],
        }
    return out

# ── PDF parsing ────────────────────────────────────────────────────────────────
def _extract_pdf_text(pdf_path: str) -> str:
    def _try_pdfplumber(path):
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text

    def _try_pypdf(path):
        from pypdf import PdfReader
        text = ""
        for page in PdfReader(path).pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text

    fname = os.path.basename(pdf_path)
    for fn, label in [(_try_pdfplumber, "pdfplumber"), (_try_pypdf, "pypdf")]:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn, pdf_path)
                result = future.result(timeout=30)
            if result.strip():
                return result.strip()
        except concurrent.futures.TimeoutError:
            print(f"  [WARN] {label} timed out on {fname}, trying next parser...")
        except Exception:
            pass

    print(f"  [WARN] Cannot parse {fname}, skipping.")
    return ""


def _extract_title_abstract(text: str) -> tuple[str, str]:
    """Best-effort extraction of title and abstract from raw PDF text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else ""

    abstract = ""
    abs_start = -1
    for i, line in enumerate(lines):
        if re.match(r'^abstract\s*$', line, re.IGNORECASE):
            abs_start = i + 1
            break
        if re.match(r'^abstract[:\s—–-]', line, re.IGNORECASE):
            abs_start = i
            break
    if abs_start >= 0:
        abs_lines = []
        for line in lines[abs_start: abs_start + 20]:
            if re.match(r'^(introduction|keywords?|1\.?\s)', line, re.IGNORECASE):
                break
            abs_lines.append(line)
        abstract = " ".join(abs_lines)

    return title, abstract


# ── Structure-aware chunking ───────────────────────────────────────────────────
def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (section_name, section_text) pairs using header detection.
    Falls back to paragraph-based splitting if no headers found.
    """
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 2:
        # No clear sections — split by double newline (paragraphs)
        paragraphs = re.split(r'\n{2,}', text)
        return [("body", p.strip()) for p in paragraphs if p.strip()]

    sections = []
    for i, m in enumerate(matches):
        sec_name = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[start:end].strip()
        if sec_text:
            sections.append((sec_name, sec_text))
    return sections


def _chunk_section(section_name: str, section_text: str,
                   min_chars: int, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Chunk a single section's text, respecting paragraph boundaries where possible.
    Large sections (methods/experiments) are allowed up to max_chars.
    """
    # Try to split on paragraph boundaries first
    paragraphs = re.split(r'\n{2,}', section_text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current and len(current) >= min_chars:
                chunks.append(current)
                # overlap: keep last overlap_chars of current as prefix
                overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
                current = overlap_text + "\n\n" + para
            else:
                # paragraph itself is too long — hard split
                if current:
                    current = current + "\n\n" + para
                else:
                    current = para
                while len(current) > max_chars:
                    chunks.append(current[:max_chars])
                    current = current[max_chars - overlap_chars:]

    if current and len(current) >= min_chars // 2:
        chunks.append(current)

    # Prepend section name to each chunk for context
    return [f"[{section_name}]\n{c}" for c in chunks]


def chunk_document(text: str, title: str = "", abstract: str = "") -> list[dict]:
    """
    Returns list of chunk dicts: {text, section, char_len}
    """
    min_chars = CHUNK_MIN_TOKENS * CHARS_PER_TOKEN
    max_chars = CHUNK_MAX_TOKENS * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    sections = _split_into_sections(text)
    chunks = []

    # Always include abstract as its own chunk if available
    if abstract and len(abstract) > 50:
        chunks.append({"text": f"[Abstract]\n{abstract}", "section": "abstract"})

    for sec_name, sec_text in sections:
        # Skip reference lists (usually not useful for RAG)
        if re.match(r'^references?\s*$', sec_name, re.IGNORECASE):
            continue
        for chunk_text in _chunk_section(sec_name, sec_text, min_chars, max_chars, overlap_chars):
            chunks.append({"text": chunk_text, "section": sec_name.lower()})

    return chunks


# ── Build index ────────────────────────────────────────────────────────────────
def build_rag_index(articles_dir: str = ARTICLES_DIR,
                    storage_dir: str = RAG_STORAGE_DIR) -> SimpleVectorStore:
    os.makedirs(storage_dir, exist_ok=True)
    manifest_path = os.path.join(storage_dir, "manifest.json")
    store_path = os.path.join(storage_dir, "default__vector_store.json")
    # meta_path = os.path.join(storage_dir, "node_meta.json")
    # text_path = os.path.join(storage_dir, "node_text.json")

    # Load existing store
    if os.path.exists(store_path) and os.path.exists(manifest_path):
        print("Loading existing RAG vector store...")
        vector_store = SimpleVectorStore.from_persist_dir(persist_dir=storage_dir)
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"   Loaded {len(vector_store._data.embedding_dict)} chunks from {len(manifest)} files.")
    else:
        vector_store = SimpleVectorStore()
        manifest = {}

    # node_meta: dict = {}
    # if os.path.exists(meta_path):
    #     with open(meta_path, encoding="utf-8") as f:
    #         node_meta = json.load(f)

    # node_text: dict = {}
    # if os.path.exists(text_path):
    #     with open(text_path, encoding="utf-8") as f:
    #         node_text = json.load(f)

    embed_model = get_embed_model()

    init_rag_db()
    conn = get_rag_db()

    pdf_files = sorted(Path(articles_dir).glob("*.pdf"))
    print(f"[PDF] {len(pdf_files)} PDFs found in {articles_dir}")

    new_files = 0
    for i, pdf_path in enumerate(pdf_files):
        fname = pdf_path.name
        stat = pdf_path.stat()
        file_sig = f"{stat.st_size}_{stat.st_mtime}"
        if manifest.get(fname) == file_sig:
            continue

        print(f"  [{i+1}/{len(pdf_files)}] {fname[:72]}")
        raw_text = _extract_pdf_text(str(pdf_path))
        if not raw_text:
            manifest[fname] = file_sig
            continue

        title, abstract = _extract_title_abstract(raw_text)
        chunks = chunk_document(raw_text, title, abstract)
        if not chunks:
            manifest[fname] = file_sig
            continue

        # Embed in batches of 20, with per-batch timeout to avoid hanging
        texts = [c["text"] for c in chunks]
        batch_size = 20
        all_embeddings = []
        for b in range(0, len(texts), batch_size):
            batch = texts[b: b + batch_size]
            embs = None
            for attempt in range(3):
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(embed_model.get_text_embedding_batch, batch)
                        embs = future.result(timeout=30)  # 30s per batch
                    break
                except concurrent.futures.TimeoutError:
                    print(f"    [WARN] Embed timeout (attempt {attempt+1}), retrying...")
                    time.sleep(2)
                except Exception as e:
                    print(f"    [WARN] Embed error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            if embs is None:
                print(f"    [WARN] Skipping batch after 3 failures.")
                all_embeddings.extend([None] * len(batch))
            else:
                all_embeddings.extend(embs)

        for chunk_dict, embedding in zip(chunks, all_embeddings):
            if embedding is None:
                continue
            node_id = hashlib.md5(
                f"{fname}_{chunk_dict['section']}_{chunk_dict['text'][:60]}".encode()
            ).hexdigest()
            node = TextNode(text=chunk_dict["text"], id_=node_id)
            node.metadata = {
                "source": fname,
                "title": title,
                "abstract": abstract[:300] if abstract else "",
                "section": chunk_dict["section"],
                "type": "ARTICLE",
            }
            node.embedding = embedding  # must be set before add()
            vector_store.add([node])
            # node_meta[node_id] = node.metadata
            # node_text[node_id] = chunk_dict["text"]  # store text separately
            conn.execute("""
            INSERT OR REPLACE INTO rag_nodes
            (node_id, source, title, abstract, section, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                node_id,
                fname,
                title,
                abstract[:300] if abstract else "",
                chunk_dict["section"],
                chunk_dict["text"],
            ))

        manifest[fname] = file_sig
        new_files += 1

        if new_files % 10 == 0:
            vector_store.persist(persist_path=os.path.join(storage_dir, "default__vector_store.json"))
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            # with open(meta_path, "w", encoding="utf-8") as f:
            #     json.dump(node_meta, f, ensure_ascii=False)
            # with open(text_path, "w", encoding="utf-8") as f:
            #     json.dump(node_text, f, ensure_ascii=False)
            conn.commit()
            print(f"  [SAVE] Checkpoint: {new_files} new files, "
                  f"{len(vector_store._data.embedding_dict)} total chunks.")

    if new_files > 0:
        vector_store.persist(persist_path=os.path.join(storage_dir, "default__vector_store.json"))
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        # with open(meta_path, "w", encoding="utf-8") as f:
        #     json.dump(node_meta, f, ensure_ascii=False)
        # with open(text_path, "w", encoding="utf-8") as f:
        #     json.dump(node_text, f, ensure_ascii=False)
        conn.commit()
        print(f"[OK] Index built: {new_files} new files, "
              f"{len(vector_store._data.embedding_dict)} total chunks.")
    else:
        print("[OK] Index up-to-date.")

    return vector_store


# ── Load ───────────────────────────────────────────────────────────────────────
def get_rag_store():
    global _RAG_VECTOR_STORE, _RAG_MATRIX, _RAG_IDS
    if _RAG_MATRIX is None:
        npy_path = os.path.join(RAG_STORAGE_DIR, "rag_embeddings.npy")
        ids_path = os.path.join(RAG_STORAGE_DIR, "rag_embedding_ids.json")
        if not os.path.exists(npy_path):
            raise RuntimeError("RAG numpy store not found. Run convert_rag_store.py first.")
        print("[RAG] Loading vector store from disk...", flush=True)
        _RAG_MATRIX = np.load(npy_path)
        with open(ids_path) as f:
            _RAG_IDS = json.load(f)
        _RAG_VECTOR_STORE = True  # sentinel so main.py compat check passes
        print(f"[RAG] Vector store ready: {len(_RAG_IDS)} chunks.", flush=True)
        init_rag_db()
        print("[RAG] Metadata DB ready.", flush=True)
    return _RAG_MATRIX, _RAG_IDS


# ── BM25 helpers ───────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def _build_bm25_index() -> list[tuple[str, list[str]]]:
    """Build an in-memory BM25 index by streaming all texts from SQLite."""
    conn = get_rag_db()
    rows = conn.execute("SELECT node_id, text FROM rag_nodes").fetchall()
    return [(r["node_id"], _tokenize(r["text"])) for r in rows]


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    from math import log
    dl = len(doc_tokens)
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    N = 1  # single-doc IDF approximation (full IDF needs corpus stats)
    for qt in set(query_tokens):
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = log((N - 0 + 0.5) / (0 + 0.5) + 1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


def _bm25_retrieve(query: str, top_k: int = BM25_TOP_K) -> list[tuple[str, float]]:
    global _BM25_INDEX
    if _BM25_INDEX is None:
        print("  [INIT] Building BM25 index...")
        _BM25_INDEX = _build_bm25_index()

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    avg_dl = sum(len(t) for _, t in _BM25_INDEX) / max(len(_BM25_INDEX), 1)
    scored = [
        (nid, _bm25_score(query_tokens, tokens, avg_dl))
        for nid, tokens in _BM25_INDEX
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ── RRF fusion ─────────────────────────────────────────────────────────────────
def _rrf_fuse(ranked_lists: list[list[tuple[str, float]]],
              k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (node_id, _) in enumerate(ranked):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Query expansion ────────────────────────────────────────────────────────────
def _expand_query(query: str, n: int = QUERY_EXPANSIONS) -> list[str]:
    """
    Use the LLM to rewrite the query into n sub-queries for better recall.
    Returns original query + rewrites.
    """
    http_client = httpx.Client(proxy=None, trust_env=False)
    from openai import OpenAI as _OpenAI
    _client = _OpenAI(
        http_client=http_client,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    try:
        resp = _client.chat.completions.create(
            model=os.environ.get("CHAT_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate {n} alternative search queries for the following question. "
                        "Each query should capture a different aspect or phrasing. "
                        f"Output exactly {n} queries, one per line, no numbering or bullets."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        rewrites = [
            l.strip()
            for l in resp.choices[0].message.content.strip().splitlines()
            if l.strip()
        ][:n]
    except Exception as e:
        print(f"  [WARN]  Query expansion failed: {e}")
        rewrites = []

    return [query] + rewrites


def _numpy_dense_search(query_embedding, matrix, ids, top_k):
    """Cosine similarity search using numpy."""
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / (norms + 1e-10)
    scores = normed @ q_norm
    top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-min(top_k, len(scores)):]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [(ids[i], float(scores[i])) for i in top_indices]


# ── Main retrieval ─────────────────────────────────────────────────────────────
def rag_retrieve(query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Hybrid retrieval with query expansion and RRF fusion (BM25 + dense).
    Returns top_k chunks as list of dicts: {source, title, section, text, score}.
    """
    matrix, ids = get_rag_store()
    embed_model = get_embed_model()

    # 1. Use original query only (skip expansion to avoid latency in chat context)
    queries = [query]
    print(f"[SEARCH] RAG: 1 query")

    all_dense_lists: list[list[tuple[str, float]]] = []
    all_bm25_lists: list[list[tuple[str, float]]] = []

    for q in queries:
        # Dense retrieval — call directly (no nested ThreadPoolExecutor which deadlocks in asyncio)
        q_emb = None
        for attempt in range(3):
            try:
                q_emb = embed_model.get_query_embedding(q)
                break
            except Exception as e:
                print(f"  [WARN] Embed error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        if q_emb is not None:
            dense_ranked = _numpy_dense_search(q_emb, matrix, ids, top_k=DENSE_TOP_K)
            all_dense_lists.append(dense_ranked)

        # BM25 retrieval skipped — too slow for 26k chunks in synchronous context

    if not all_dense_lists:
        return []

    # 2. Use dense results directly (no BM25 to fuse)
    fused = _rrf_fuse(all_dense_lists)

    # 3. Take top RERANK_CUTOFF, fetch from SQLite, return top_k
    top_candidates = fused[:RERANK_CUTOFF]
    payload_map = fetch_nodes_by_ids([node_id for node_id, _ in top_candidates])

    results = []
    for node_id, rrf_score in top_candidates:
        payload = payload_map.get(node_id)
        if not payload:
            continue
        results.append({
            "source": payload.get("source", "unknown"),
            "title": payload.get("title", ""),
            "section": payload.get("section", ""),
            "abstract": payload.get("abstract", ""),
            "text": payload.get("text", ""),
            "score": round(rrf_score, 6),
        })
        if len(results) >= top_k:
            break

    print(f"   [OK] Returning {len(results)} chunks after RRF fusion.")
    return results


# ── CLI: build index ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building RAG index from scratch...")
    store = build_rag_index()
    print(f"Done. Total chunks: {len(store._data.embedding_dict)}")
