"""
Convert RAG SimpleVectorStore JSON to numpy binary format.
Uses ijson streaming + memmap to avoid ever holding all embeddings in RAM.

Pass 1: count nodes and detect embedding dim (reads only first embedding)
Pass 2: stream-write each embedding directly into a pre-allocated memmap file
"""
import ijson
import json
import numpy as np
import os

SRC = os.path.join(os.path.dirname(__file__), "rag_storage", "default__vector_store.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "rag_storage")
NPY_PATH = os.path.join(OUT_DIR, "rag_embeddings.npy")
IDS_PATH = os.path.join(OUT_DIR, "rag_embedding_ids.json")

print(f"File size: {os.path.getsize(SRC) / 1e6:.1f} MB", flush=True)

# ── Pass 1: count nodes and get embedding dim ──────────────────────────────
print("Pass 1: counting nodes...", flush=True)
n_nodes = 0
emb_dim = None
with open(SRC, "rb") as f:
    for node_id, emb in ijson.kvitems(f, "embedding_dict"):
        if emb_dim is None:
            emb_dim = len(emb)
        n_nodes += 1
        if n_nodes % 5000 == 0:
            print(f"  counted {n_nodes}...", flush=True)

print(f"Total nodes: {n_nodes}, dim: {emb_dim}", flush=True)

# ── Pre-allocate memmap (written directly to disk, ~0 RAM) ────────────────
print(f"Pre-allocating {n_nodes * emb_dim * 4 / 1e6:.0f} MB on disk...", flush=True)
mm = np.lib.format.open_memmap(NPY_PATH, mode="w+", dtype=np.float32, shape=(n_nodes, emb_dim))

# ── Pass 2: stream embeddings directly into memmap ────────────────────────
print("Pass 2: writing embeddings to disk...", flush=True)
ids = []
with open(SRC, "rb") as f:
    for i, (node_id, emb) in enumerate(ijson.kvitems(f, "embedding_dict")):
        mm[i] = emb
        ids.append(node_id)
        if (i + 1) % 5000 == 0:
            print(f"  written {i+1}/{n_nodes}...", flush=True)

mm.flush()
del mm

print(f"Saving IDs...", flush=True)
with open(IDS_PATH, "w") as f:
    json.dump(ids, f)

print(f"Done. npy: {os.path.getsize(NPY_PATH) / 1e6:.0f} MB", flush=True)
