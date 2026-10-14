import os
import httpx
import json
import sys
import time
import numpy as np
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import *

# --- 配置 ---
http_client = httpx.Client(proxy=None, trust_env=False)

STORAGE_DIR = "./server/graph_storage"

# --- Globals for caching stores ---
_GLOBAL_VECTOR_MATRIX = None   # np.ndarray (N, D) float32
_GLOBAL_VECTOR_IDS = None      # list[str]
_GLOBAL_GRAPH_STORE = None
_GLOBAL_EMBED_MODEL = None
_GLOBAL_EMBED_MODEL_NAME = None

# Embeddings are DECOUPLED from the chat model. DeepSeek's official API has NO
# embeddings endpoint, so embeddings go to their own provider via EMBED_* env
# (default: Alibaba DashScope text-embedding-v4, OpenAI-compatible).
# IMPORTANT: switching the embed model changes the vector space — the on-disk
# corpus graph_storage/embeddings.npy MUST be re-embedded with the SAME model
# (run server/reembed_graph_store.py). text-embedding-v4 hard-caps batch at 10.
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")

def _build_embed_model():
    return OpenAIEmbedding(
        model_name=EMBED_MODEL_NAME,
        api_key=EMBED_API_KEY,
        api_base=EMBED_BASE_URL,
        embed_batch_size=10,     # DashScope text-embedding-v4 hard-caps batch at 10
        dimensions=EMBED_DIM,
    )

def get_embed_model():
    """Return the active embedding model (lazy singleton)."""
    global _GLOBAL_EMBED_MODEL, _GLOBAL_EMBED_MODEL_NAME
    if _GLOBAL_EMBED_MODEL is None:
        _GLOBAL_EMBED_MODEL = _build_embed_model()
        _GLOBAL_EMBED_MODEL_NAME = EMBED_MODEL_NAME
    return _GLOBAL_EMBED_MODEL

def embed_query_with_fallback(text: str, max_attempts: int = 3):
    """Embed a query string via the configured embedding model (DashScope
    text-embedding-v4 by default). Returns the embedding list, or None if all
    attempts fail."""
    model = get_embed_model()
    last_err = None
    for attempt in range(max_attempts):
        try:
            return model.get_query_embedding(text)
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (2 ** attempt))
    print(f"❌ [KG] Embedding failed: {last_err}", flush=True)
    return None

def embed_via_client_with_fallback(client, text):
    """Back-compat shim for PlannerAgent. The passed chat `client` is IGNORED —
    embeddings use the dedicated EMBED_* endpoint, never the chat (DeepSeek)
    endpoint, which has no embeddings API."""
    return embed_query_with_fallback(text, max_attempts=2)

# def get_global_stores():
#     global _GLOBAL_VECTOR_STORE, _GLOBAL_GRAPH_STORE    
#     if _GLOBAL_VECTOR_STORE is None or _GLOBAL_GRAPH_STORE is None:
#         print("Initializing Global Stores for the first time...")
#         _GLOBAL_VECTOR_STORE, _GLOBAL_GRAPH_STORE = load_stores()
    
#     return _GLOBAL_VECTOR_STORE, _GLOBAL_GRAPH_STORE

def get_graph_store():
    global _GLOBAL_GRAPH_STORE
    if _GLOBAL_GRAPH_STORE is None:
        print("[KG] Connecting Neo4j graph store...", flush=True)
        _GLOBAL_GRAPH_STORE = Neo4jPropertyGraphStore(
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD"),
            url=os.environ.get("NEO4J_URL"),
            liveness_check_timeout=60,
            max_connection_lifetime=120,
            connection_acquisition_timeout=60,
        )
        print("[KG] Neo4j graph store ready.", flush=True)
    return _GLOBAL_GRAPH_STORE

def get_vector_store():
    global _GLOBAL_VECTOR_MATRIX, _GLOBAL_VECTOR_IDS
    if _GLOBAL_VECTOR_MATRIX is None:
        npy_path = os.path.join(STORAGE_DIR, "embeddings.npy")
        ids_path = os.path.join(STORAGE_DIR, "embedding_ids.json")
        print("[KG] Loading vector store from disk...", flush=True)
        _GLOBAL_VECTOR_MATRIX = np.load(npy_path)
        with open(ids_path) as f:
            _GLOBAL_VECTOR_IDS = json.load(f)
        print(f"[KG] Vector store ready: {len(_GLOBAL_VECTOR_IDS)} nodes.", flush=True)
    return _GLOBAL_VECTOR_MATRIX, _GLOBAL_VECTOR_IDS

def get_global_stores():
    return get_vector_store(), get_graph_store()

def run_with_retry(func, *args, max_retries=3, initial_delay=1, **kwargs):
    """
    Executes a function with retry logic for transient Neo4j errors.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (ServiceUnavailable, SessionExpired, TransientError) as e:
            last_exception = e
            print(f"⚠️ Neo4j Connection Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        except Exception as e:
            # For other exceptions, re-raise immediately
            raise e
            
    print(f"❌ Failed after {max_retries} attempts.")
    raise last_exception

def execute_read_with_session(driver, func, *args, **kwargs):
    """Safely execute a read transaction and immediately close the session."""
    with driver.session() as session:
        return session.execute_read(func, *args, **kwargs)

def _numpy_vector_search(query_embedding, matrix, ids, top_k, node_type_filter_ids=None):
    """Cosine similarity search using numpy. Returns (ids, scores) sorted by score desc."""
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / (norms + 1e-10)
    scores = normed @ q_norm  # (N,)

    if node_type_filter_ids is not None:
        # Zero out scores for IDs not in the filter set
        filter_set = set(node_type_filter_ids)
        mask = np.array([1.0 if id_ in filter_set else 0.0 for id_ in ids], dtype=np.float32)
        scores = scores * mask

    top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-min(top_k, len(scores)):]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [ids[i] for i in top_indices], [float(scores[i]) for i in top_indices]


def custom_retrieve(query_text, vector_store, graph_store, top_k, node_type=None, role_filter=None, community_id=None):
    print(f"\n🔍 Executing Custom Retrieval for: '{query_text}'", flush=True)
    matrix, ids = vector_store  # unpack (np.ndarray, list[str])

    query_embedding = embed_query_with_fallback(query_text, max_attempts=3)
    if query_embedding is None:
        print(f"   ❌ Embedding unavailable — skipping retrieval for '{query_text}'", flush=True)
        return []

    found_ids = []
    found_scores = []

    # CASE A: Structure-Based Retrieval (Community ID provided)
    if community_id:
        print(f"   🎯 Targeted Retrieval in Community: {community_id}")

        def _fetch_community_candidates(tx, comm_id, role_filter):
            cypher = """
            MATCH (n:LlamaNode {community_L0: $comm_id})
            WHERE n.type = 'MODULE'
            AND ($role_filter IS NULL OR replace(toLower(n.role), ' ', '') CONTAINS replace(toLower($role_filter), ' ', ''))
            RETURN n.id as id
            """
            result = tx.run(cypher, comm_id=comm_id, role_filter=role_filter)
            return [r["id"] for r in result]

        candidate_ids = run_with_retry(
            lambda: execute_read_with_session(graph_store._driver, _fetch_community_candidates, community_id, role_filter)
        )

        print(f"      Found {len(candidate_ids)} candidates in graph.")

        if candidate_ids:
            # Broad search top-100, then intersect with community candidates
            broad_ids, broad_scores = _numpy_vector_search(query_embedding, matrix, ids, top_k=100)
            id_score_map = dict(zip(broad_ids, broad_scores))

            scored_candidates = []
            for cid in candidate_ids:
                if cid in id_score_map:
                    scored_candidates.append((cid, id_score_map[cid] * 1.5))
                else:
                    scored_candidates.append((cid, 0.4))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            found_ids = [x[0] for x in scored_candidates]
            found_scores = [x[1] for x in scored_candidates]
        else:
            print("   ⚠️ Community empty or role mismatch. Fallback to global search.")
            community_id = None

    # CASE B: Global Vector Search (Fallback or Default)
    if not community_id:
        found_ids, found_scores = _numpy_vector_search(query_embedding, matrix, ids, top_k=top_k)
        print(f"   -> Vector Search found {len(found_ids)} candidate IDs.")

    retrieved_nodes = []

    # Batch fetch all node details in a single Neo4j round-trip
    skip_role = bool(community_id)
    candidate_ids = list(found_ids[:top_k * 2])  # fetch a bit more to account for role filtering
    id_to_score = dict(zip(found_ids, found_scores))

    def _fetch_batch_node_details(tx, node_ids, role_filter, skip_role):
        cypher = """
        UNWIND $ids AS node_id
        MATCH (center:LlamaNode) WHERE center.id = node_id
        AND ($skip_role_check = true OR $role_filter IS NULL OR replace(toLower(center.role), ' ', '') CONTAINS replace(toLower($role_filter), ' ', ''))
        WITH center
        CALL {
            WITH center
            OPTIONAL MATCH (center)-[r:FLOWS_TO]->(next:LlamaNode)
            RETURN collect(distinct {name: next.name, id: next.id, function: next.text, code: next.code, edge_label:r.edge_label, properties: properties(r)}) as downstreams
        }
        CALL {
            WITH center
            OPTIONAL MATCH (prev:LlamaNode)-[r:FLOWS_TO]->(center)
            RETURN collect(distinct {name: prev.name, id: prev.id, function: prev.text, code: prev.code, edge_label:r.edge_label, properties: properties(r)}) as upstreams
        }
        CALL {
            WITH center
            OPTIONAL MATCH (center)-[:DEPENDS_ON]->(dep_wf:LlamaNode)
            RETURN collect(distinct {name: dep_wf.name, id: dep_wf.id, function: dep_wf.text, code: dep_wf.code}) as dependencies
        }
        RETURN
            center,
            center.source as source_file,
            center.community_L0 as comm_id,
            center.summary_L0 as comm_summary,
            upstreams,
            downstreams,
            dependencies
        """
        result = tx.run(cypher, ids=node_ids, role_filter=role_filter, skip_role_check=skip_role)
        return list(result)

    try:
        records = run_with_retry(
            lambda: execute_read_with_session(graph_store._driver, _fetch_batch_node_details, candidate_ids, role_filter, skip_role)
        )
        # Build a map keyed by node id to preserve score ordering
        record_map = {}
        for record in records:
            center_node = record["center"]
            nid = center_node["id"]
            record_map[nid] = record

        # Reconstruct in score order
        for node_id in found_ids:
            if len(retrieved_nodes) >= top_k:
                break
            record = record_map.get(node_id)
            if not record:
                continue
            center_node = record["center"]
            score = id_to_score.get(node_id, 0.0)
            node = TextNode(text=center_node.get("text", ''), id_=center_node["id"])
            node.metadata = dict(center_node)
            node.metadata["topology"] = {
                "upstream": record["upstreams"],
                "downstream": record["downstreams"],
                "dependencies": record["dependencies"],
                "source_file": record.get("source_file"),
            }
            comm_summary = record.get("comm_summary")
            if comm_summary:
                node.metadata["community_summary"] = comm_summary
            retrieved_nodes.append(NodeWithScore(node=node, score=score))

    except Exception as e:
        print(f"⚠️ Batch fetch failed: {e}")

    return retrieved_nodes

def get_all_community_summaries(vector_store, graph_store):
    """
    Retrieve all available high-level community summaries to inform the Planner.
    """
    print("🌍 Fetching Community Capabilities...")
    # Since we can't easily 'scan' vector store, we use a simple Cypher query 
    # assuming we wrote summaries back to Module nodes.
    cypher = """
    MATCH (n:LlamaNode) 
    WHERE n.community_L0 IS NOT NULL
    RETURN DISTINCT n.community_L0 as id, n.summary_L0 as summary
    LIMIT 50
    """
    summaries = []
    
    def _fetch_summaries(tx):
        result = tx.run(cypher)
        return [f"[Community ID: {record['id']}]: {record['summary']}" for record in result]

    try:
        summaries = run_with_retry(
            lambda: execute_read_with_session(graph_store._driver, _fetch_summaries)
        )
    except Exception as e:
        print(f"⚠️ Failed to fetch community summaries: {e}")
        return ""

    return "\n".join(summaries)

def retrieve_and_get_content(query, vector_store, graph_store, top_k=5, node_type=None, role_filter=None, community_id=None, include_topology=True):
    if vector_store is None or graph_store is None:
        vector_store, graph_store = get_global_stores()
    retrieved_nodes = custom_retrieve(query, vector_store, graph_store, top_k, node_type, role_filter, community_id)

    processed_nodes = []
    for node_with_score in retrieved_nodes:
        node = node_with_score.node
        score = node_with_score.score
        n_type = node.metadata.get("type") # Use n_type to avoid confusion with argument node_type

        enriched_text = ""
        topology_text = ""
        
        if include_topology:
            topology = node.metadata.get("topology", {})
            upstreams = topology.get("upstream", [])
            downstreams = topology.get("downstream", [])
            # Only include if not empty
            if upstreams or downstreams:
                topology_text = (
                    f"--- Context ---\n"
                    f"Proceded By: {[u['name'] for u in upstreams]}\n"
                    f"Followed By: {[d['name'] for d in downstreams]}\n"
                )

        if n_type == "WORKFLOW":
            topology = node.metadata.get("topology", {})
            dependencies = topology.get("dependencies", [])
            enriched_text = (
                f"Workflow: {node.metadata.get('name')}\n"
                f"Description: {node.metadata.get('text')}\n"
                f"Source: {node.metadata.get('source')}\n"
                f"--- Dependencies ---\n"
                f"Depends on {[d['name'] for d in dependencies if 'name' in d]}"
            )
        elif n_type == "MODULE":
            # Incorporate Community Context if available
            comm_summary = node.metadata.get("community_summary", "")
            context_str = f"Community Context: {comm_summary}\n" if comm_summary else ""
            
            enriched_text = (
                f"Module: {node.metadata.get('name')}\n"
                f"Source: {node.metadata.get('source')}\n"
                f"{context_str}"
                f"Text content: {node.metadata.get('text')}\n"
                f"{topology_text}"
            )
        else:
            enriched_text = (
                f"Article: {node.metadata.get('name')}\n"
                f"Text Conent: {node.metadata.get('text')}\n"
            )
        # Reconstruct node
        new_node = TextNode(text=enriched_text, id_=node.node_id)
        new_node.metadata = node.metadata
        
        processed_nodes.append(NodeWithScore(node=new_node, score=score))

    return processed_nodes

def find_structural_path(source_text, target_text, vector_store, graph_store, rel_type="DATA_FLOW", top_k=5):
    """
    Find the structural path between two nodes in the graph.
    """
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=2) as _ex:
        _f_src = _ex.submit(custom_retrieve, source_text, vector_store, graph_store, top_k, "MODULE")
        _f_tgt = _ex.submit(custom_retrieve, target_text, vector_store, graph_store, top_k, "MODULE")
        src_nodes = _f_src.result()
        tgt_nodes = _f_tgt.result()
    src_ids = [node.node.id_ for node in src_nodes]
    tgt_ids = [node.node.id_ for node in tgt_nodes]
    if not src_ids or not tgt_ids:
        return []
    found_paths = []

    if rel_type == "CONTAINS":        
        pass

    else:
        cypher = """
        MATCH (s:LlamaNode), (t:LlamaNode)
        WHERE s.id IN $src_ids AND t.id IN $tgt_ids
        AND s.id <> t.id
        MATCH p = shortestPath((s)-[*1..5]-(t))
        WHERE any(r in relationships(p) WHERE type(r) = 'FLOWS_TO')
        RETURN
            [n in nodes(p) | {id: n.id, name: n.name, text: n.text, function: n.function}] as path_nodes,
            [r in relationships(p) | type(r)] as relations,
            [r in relationships(p) | properties(r)] as rel_props,
            length(p) as len
        ORDER BY len ASC
        LIMIT 10
        """
        def _find_paths(tx, src_ids, tgt_ids):
            result = tx.run(cypher, src_ids=src_ids, tgt_ids=tgt_ids)
            return list(result)

        try:
            records = run_with_retry(
                lambda: execute_read_with_session(graph_store._driver, _find_paths, src_ids, tgt_ids)
            )
        except Exception as e:
            print(f"⚠️ Failed to find paths: {e}")
            return []

        unique_signatures = set()
        
        for record in records:
            path_nodes = record["path_nodes"]
            relations = record["relations"]
                
            # Generate a signature based on the actual data flow structure
            # This dedups paths that only differ by their entry point (via HAS_ROLE)
            flow_signature_parts = []
            for i, rel in enumerate(relations):
                if rel == "FLOWS_TO":
                        flow_signature_parts.append(f"{path_nodes[i]['name']}->{path_nodes[i+1]['name']}")
            
            signature = "|".join(flow_signature_parts)
            # If no flows_to (shouldn't happen due to WHERE), skip this path
            if not signature:
                continue
            if signature in unique_signatures:
                continue
            unique_signatures.add(signature)
            
            if len(found_paths) >= 3:
                break

            path_data = {
                "source_query": source_text,
                "target_query": target_text,
                "path_nodes": record["path_nodes"],
                "relations": record["relations"],
                "rel_props": record["rel_props"]
            }
            found_paths.append(path_data)

            node_names = "-> ".join([node["name"] for node in record["path_nodes"]])
            print(f"   ✅ Path found: {node_names}")

    
    return found_paths


if __name__ == "__main__":
    vector_store = get_vector_store()
    graph_store = get_graph_store()
    query = "The query is routed to the appropriate category manager, which then provides an answer using its knowledge base. If the query doesn't fit a category, a fine-tuned model generates the answer."

    paths = find_structural_path("detector", "facial recognition", vector_store, graph_store, rel_type="CONTAINS", top_k=5)
    print(paths)