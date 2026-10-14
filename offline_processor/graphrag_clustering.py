"""
GraphRAG Clustering Pipeline for Neo4j Module Nodes

This script implements:
1. KNN-based similarity edge creation with weights (within same role)
2. Per-role Louvain clustering with resolution scaled by role size
3. LLM-based community summarization (with caching and few-shot prompting)
4. Writing community_id back to the database (with cleanup)

Clustering strategy:
    - Nodes are grouped by canonical role before clustering
    - Resolution scales with role size so all roles produce similarly-sized communities
    - Target: ~15 nodes per community
    - Roles with only 1 node become forced singletons
"""

import os
import sys
import httpx
import json
import hashlib
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import defaultdict


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from server.retriever_neo4j import get_vector_store, get_graph_store


# configuration
http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(http_client=http_client)

# Clustering Parameters
KNN_K = 10
MIN_SIMILARITY = 0.5
MAX_HIERARCHY_LEVELS = 3
MIN_REDUCTION_RATIO = 0.10  # If communities reduce by less than 10%, stop.

TARGET_COMMUNITY_SIZE = 20   # Desired average nodes per community

def resolution_for_role(n_nodes: int) -> float:
    """
    Return a Louvain resolution scaled so the expected community size
    is ~TARGET_COMMUNITY_SIZE regardless of how many nodes a role has.

    Empirically, resolution ≈ n_nodes / (TARGET * baseline_factor).
    We use baseline_factor=10 (calibrated so that a 150-node role at
    resolution=1.0 yields ~15 communities of ~10 nodes each).

    Clamped to [0.5, 8.0] to avoid degenerate extremes.
    """
    raw = n_nodes / (TARGET_COMMUNITY_SIZE * 10.0)
    return round(max(0.5, min(8.0, raw)), 2)

# Path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "server", "kg_cluster")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "community_summary_cache.json")

class GraphClustering:
    def __init__(self, emb_matrix, emb_ids, graph_store):
        # emb_matrix: np.ndarray (N, D); emb_ids: list[str]
        self._embedding_dict = {nid: emb_matrix[i] for i, nid in enumerate(emb_ids)}
        self.graph_store = graph_store
        self.module_nodes = []
        self.embeddings = []
        self.node_id_to_idx = {}

        self.hierarchy_map = defaultdict(list)
        self.hierarchy_summaries = defaultdict(dict)
        self.level_communities = defaultdict(set)
        
        # Load Cache
        self.summary_cache = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.summary_cache = json.load(f)
                print(f"✅ Loaded {len(self.summary_cache)} cached summaries.")
            except:
                print("⚠️ Failed to load cache, starting fresh.")

    def _save_cache(self):
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.summary_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def _check_role_compatibility(self, role_a, role_b):
        if not role_a or not role_b:
            return True
            
        def extract_role(r):
            parts = r.split(':')
            return parts[1].strip().lower() if len(parts) > 1 else r.strip().lower()
            
        r1 = extract_role(role_a)
        r2 = extract_role(role_b)
        return r1 == r2
        
    def create_constrained_edges(self, write_edges_to_neo4j=True):
        """
        Step 1: Load module nodes/embeddings, optionally write KNN edges to Neo4j.
        Pass write_edges_to_neo4j=False to skip the slow Neo4j write (e.g. on re-runs).
        """
        print("\n" + "="*60)
        print("STEP 1: Creating KNN Similarity Edges")
        print(write_edges_to_neo4j)
        
        # 1.1 Fetch all MODULE nodes from Neo4j
        cypher = """
        MATCH (n:LlamaNode {type: 'MODULE'})
        RETURN n.id as id, n.name as name, n.text as text, n.role as role
        """
        
        def fetch_modules(tx):
            return [dict(record) for record in tx.run(cypher)]
            
        with self.graph_store._driver.session() as session:
            self.module_nodes = session.execute_read(fetch_modules)
        
        # 1.2 Get embeddings from vector store
        embedding_dict = self._embedding_dict
        valid_nodes = []
        for node in self.module_nodes:
            if node['id'] in embedding_dict:
                self.embeddings.append(embedding_dict[node['id']])
                self.node_id_to_idx[node['id']] = len(valid_nodes)
                valid_nodes.append(node)

        self.module_nodes = valid_nodes
        self.embeddings = np.array(self.embeddings)
        print(f"Processing {len(self.module_nodes)} nodes with embeddings")

        if len(self.module_nodes) == 0:
            print("No nodes found. Exiting.")
            return 0

        if not write_edges_to_neo4j:
            print("⏭️  Skipping KNN computation and Neo4j edge write.")
            return 0

        # 1.3 Compute KNN
        print(f"\n🔍 Computing KNN (k={KNN_K})...")
        nbrs = NearestNeighbors(n_neighbors=min(KNN_K+1, len(self.embeddings)), metric='cosine').fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        
        # 1.4 Create edges in Neo4j (using batch operations)
        print("\n💾 Creating SIMILAR_TO edges in Neo4j...")
        
        # Prepare batch data
        edges_to_create = []
        for i, node in enumerate(self.module_nodes):
            for j in range(1, indices.shape[1]): # Iterate actual neighbors found
                neighbor_idx = indices[i][j]
                neighbor_node = self.module_nodes[neighbor_idx]
                if self._check_role_compatibility(node['role'], neighbor_node['role']):
                    similarity = 1 - distances[i][j]
                    if similarity >= MIN_SIMILARITY:
                        edges_to_create.append({
                            'source': node['id'],
                            'target': neighbor_node['id'],
                            'weight': float(similarity)
                        })
        
        print(f"   Prepared {len(edges_to_create)} edges for batch creation...")
        
        def clear_edges(tx):
            tx.run("MATCH ()-[r:SIMILAR_TO]-() DELETE r")
            
        def build_edges_batch(tx, batch_data):
            cypher_query = """
            UNWIND $edges as edge
            MATCH (a:LlamaNode {id: edge.source})
            MATCH (b:LlamaNode {id: edge.target})
            MERGE (a)-[r:SIMILAR_TO]-(b)
            SET r.weight = edge.weight
            """
            tx.run(cypher_query, edges=batch_data)

        with self.graph_store._driver.session() as session:
            # First, remove existing SIMILAR_TO edges to avoid duplicates
            session.execute_write(clear_edges)
            print("   Cleared existing SIMILAR_TO edges")
            
            # Batch create edges using UNWIND
            batch_size = 2000
            for i in range(0, len(edges_to_create), batch_size):
                batch = edges_to_create[i:i+batch_size]
                session.execute_write(build_edges_batch, batch)
        return len(edges_to_create)
    
    def leiden_clustering(self):
        """
        Step 2: Per-role Louvain clustering.

        Each canonical role is clustered independently with a resolution
        proportional to its size (resolution_for_role).  This avoids the
        problem where a giant role (e.g. decision_logic with 1000+ nodes)
        dominates and produces a single massive community.

        Community IDs are global and unique: L0_{role_slug}_{local_id}.
        Only L0 is produced (single-level); hierarchy is not needed because
        the role grouping already provides the semantic hierarchy.
        """
        print("\n" + "="*60)
        print("STEP 2: Per-Role Louvain Clustering")
        print("="*60)

        import community as community_louvain

        # Build embedding dict for fast lookup
        emb_dict = {node['id']: self.embeddings[i]
                    for i, node in enumerate(self.module_nodes)}

        # Group nodes by canonical role
        role_groups: dict[str, list[dict]] = defaultdict(list)
        for node in self.module_nodes:
            role = node.get('role') or 'unknown'
            if ':' in role:
                parts = role.split(':', 1)
                role = f"{parts[0].strip()}:{parts[1].strip()}"
            role_groups[role].append(node)

        print(f"  Roles to cluster: {len(role_groups)}")

        global_partition: dict[str, str] = {}   # node_id -> community_id
        community_counter = 0

        for role, nodes in sorted(role_groups.items()):
            n = len(nodes)
            role_slug = role.replace(':', '_').replace(' ', '_').replace('/', '_')

            if n == 1:
                # Forced singleton
                comm_id = f"L0_{community_counter}"
                global_partition[nodes[0]['id']] = comm_id
                self.level_communities[0].add(comm_id)
                community_counter += 1
                continue

            # Build embeddings array for this role
            role_embs = np.array([emb_dict[node['id']] for node in nodes
                                   if node['id'] in emb_dict])
            role_ids  = [node['id'] for node in nodes if node['id'] in emb_dict]

            if len(role_ids) < 2:
                comm_id = f"L0_{community_counter}"
                global_partition[role_ids[0]] = comm_id
                self.level_communities[0].add(comm_id)
                community_counter += 1
                continue

            # KNN within role
            real_k = min(KNN_K + 1, len(role_ids))
            nbrs = NearestNeighbors(n_neighbors=real_k, metric='cosine').fit(role_embs)
            distances, indices = nbrs.kneighbors(role_embs)

            G = nx.Graph()
            for nid in role_ids:
                G.add_node(nid)
            for i, nid in enumerate(role_ids):
                for j in range(1, indices.shape[1]):
                    nb_idx = indices[i][j]
                    nb_id  = role_ids[nb_idx]
                    sim = 1.0 - distances[i][j]
                    if sim >= MIN_SIMILARITY:
                        G.add_edge(nid, nb_id, weight=float(sim))

            # Scale resolution by role size
            res = resolution_for_role(n)
            print(f"  {role:50s}  n={n:5d}  res={res}")

            partition = community_louvain.best_partition(G, weight='weight', resolution=res)

            # Map local partition IDs to global IDs
            local_to_global: dict[int, str] = {}
            for node_id, local_id in partition.items():
                if local_id not in local_to_global:
                    local_to_global[local_id] = f"L0_{community_counter}"
                    self.level_communities[0].add(f"L0_{community_counter}")
                    community_counter += 1
                global_partition[node_id] = local_to_global[local_id]

        print(f"\n  Total L0 communities: {len(self.level_communities[0])}")
        self.hierarchy_map[0] = global_partition
            
    def generate_summaries(self):
        """
        Step 3: Generate LLM summaries for each community (Cached)
        """
        print("\n" + "="*60)
        print("STEP 3: Generating Community Summaries")
        print("="*60)
        
        def get_content_from_comm(level, comm_id):
            partition = self.hierarchy_map[level]
            child_ids = [k for k, v in partition.items() if v == comm_id]
            
            # Sort for stable hashing
            child_ids.sort()
            
            texts = []
            
            # Create a hash signature for this community's composition
            # logic: Level + Sorted(ChildIDs)
            sig_str = f"{level}_{','.join(child_ids)}"
            comm_hash = hashlib.md5(sig_str.encode()).hexdigest()

            if level == 0:
                if not hasattr(self, '_node_text_lookup'):
                    self._node_text_lookup = {n['id']: n['text'] for n in self.module_nodes}
                    self._node_role_lookup = {n['id']: n['role'] for n in self.module_nodes}
                
                roles = []
                for cid in child_ids:
                    txt = self._node_text_lookup.get(cid, "")
                    role = self._node_role_lookup.get(cid, "")
                    texts.append(f"[Role: {role}] {txt[:450]}")
                    roles.append(role)
                return "\n".join(texts[:25]), list(set(roles)), comm_hash
            else:
                prev_summaries = self.hierarchy_summaries[level-1]
                for cid in child_ids:
                    if cid in prev_summaries:
                        texts.append(f"[Sub-Group Summary]: {prev_summaries[cid]}")
                return "\n".join(texts), [], comm_hash

        sorted_levels = sorted(self.hierarchy_map.keys())
        
        # Example to guide the LLM (Few-Shot)
        few_shot_prompt = """
        Example 1:
        Input Modules:
        - Module: EEG_Headset
            Role: sensing_input:bio_sensor
            Function: Captures electrical brain activity.
            Technical Details: 8-channel dry electrode system, 256Hz sampling.
            Input: Neural electrical signals
            Output: Raw EEG stream

        - Module: EMG_Sensor
            Role: sensing_input:bio_sensor
            Function: Detects muscle activation.
            Technical Details: Surface electromyography.
            Input: Muscle movement
            Output: Analog voltage signal
        
        Output: ```
        The community collectively functions as physiological signal acquisition hardware, focusing on non-invasive monitoring of human bio-signals.
        Typical Inputs: Physical physiological states; Outputs: Raw analog or digitized time-series bio-data (EEG, EMG, GSR).
        Innovative Highlights: Features advanced dry electrode systems enabling unconstrained setup compared to traditional wet electrodes.
        The assigned role: 'sensing_input:bio_sensor'
        ```

        Example 2:
        Input Modules:
        - Module: Keyframe_Extraction
            Role: processing_core:feature_extraction
            Function: Identifies salient frames in video.
            Technical Details: Uses VGG16 feature vectors and IMU data for similarity analysis.
            Input: Video Stream, IMU Data
            Output: Keyframe Images

        - Module: Intent_Classifier
            Role: processing_core:feature_extraction
            Function: Understands user requests.
            Technical Details: Fine-tuned BERT model for intent classification.
            Input: Text Query
            Output: Intent Class Labels
        
        Output: ```
        The community collectively functions as a Multimodal Feature Extraction Engine, utilizing Deep Learning models (VGG16, BERT) to derive structured representations from unstructured raw data.
        Typical Inputs: Video streams, Text queries; Outputs: Discrete visual keyframes and symbolic intent labels.
        Innovative Highlights: Employs cross-modal similarity analysis (VGG16 + IMU) to identify precise interaction moments instead of standard uniform sampling.
        The assigned role: 'processing_core: feature_extraction'
        ```
        """

        for level in sorted_levels:
            comm_ids = self.level_communities[level]
            print(f"--- Level {level} ({len(comm_ids)} communities) ---")
            
            new_gen_count = 0
            cached_count = 0
            
            for comm_id in comm_ids:
                content_text, roles, comm_hash = get_content_from_comm(level, comm_id)
                
                # CHECK CACHE
                if comm_hash in self.summary_cache:
                    self.hierarchy_summaries[level][comm_id] = self.summary_cache[comm_hash]
                    cached_count += 1
                    continue

                # Generate Prompt
                if level == 0:
                    prompt = f"""Target: Analyze the provided {len(roles)} modules to create a high-level Capability Summary.
                        ## Input Context:
                        Roles involved: {', '.join(roles)}
                        Modules: 
                        {content_text}

                        ## Task
                        Synthesize a **technical capability summary** that strictly follows the format and tone of the reference examples.
                        The summary must be 3-4 sentences max.
                        Crucially, you MUST include an "Innovative Highlights:" section. Look at the technical details of the modules and explicitly state any advanced, unconventional, or breakthrough methods that distinguish this community from basic/standard implementations. If nothing is highly innovative, describe the most complex technical feature.

                        ## Reference Examples
                        {few_shot_prompt}
                        """
                else:
                    prompt = f"""Target: Analyze this System Capability Group (Level {level}) composed of sub-groups.
                        Current Input:
                        {content_text}

                        Task: Summarize the High-Level System Architecture capability this group represents. 
                        Focus on the problem it solves within the larger architecture. Keep it concise (2-3 sentences).
                        """
                try:
                    response = client.chat.completions.create(
                        model = "gpt-4o-mini",
                        messages=[{"role":"user", "content":prompt}],
                        temperature=0
                    )
                    summary = response.choices[0].message.content
                    self.hierarchy_summaries[level][comm_id] = summary
                    
                    # Update Cache
                    self.summary_cache[comm_hash] = summary
                    new_gen_count += 1
                    
                except Exception as e:
                    print(f"Error summarizing community {comm_id}: {str(e)}")
            
            print(f"   ✅ Finished: {new_gen_count} generated, {cached_count} cached.")
            
            # Save cache after every level to be safe
            self._save_cache()
    
    def write_hierarchy_to_db(self):
        """
        Step 4: Write L0 community data to Neo4j and clean up stale L1-L5 attributes.
        """
        print("\n" + "="*60)
        print("STEP 4: Writing Community Data to Neo4j")
        print("="*60)

        partition = self.hierarchy_map[0]
        summaries = self.hierarchy_summaries[0]

        update_data = []
        for node in self.module_nodes:
            nid = node['id']
            comm = partition.get(nid)
            if not comm:
                continue
            update_data.append({
                'id':          nid,
                'community_L0': comm,
                'summary_L0':   summaries.get(comm, ""),
            })

        print(f"Preparing to update {len(update_data)} nodes...")

        def set_batch(tx, cypher_query, batch_data):
            tx.run(cypher_query, batch=batch_data)

        with self.graph_store._driver.session() as session:
            # 1. Remove stale L1-L5 attributes from previous multi-level runs
            print("   Cleaning up stale L1-L5 attributes...")
            session.run(
                "MATCH (n:LlamaNode) "
                "REMOVE n.community_L1, n.summary_L1, "
                "n.community_L2, n.summary_L2, "
                "n.community_L3, n.summary_L3, "
                "n.community_L4, n.summary_L4, "
                "n.community_L5, n.summary_L5"
            )

            # 2. Write L0
            batch_size = 500
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i:i + batch_size]
                session.execute_write(
                    set_batch,
                    "UNWIND $batch as node "
                    "MATCH (n:LlamaNode {id: node.id}) "
                    "SET n.community_L0 = node.community_L0, "
                    "    n.summary_L0   = node.summary_L0",
                    batch
                )
                print(f"   Batch {i//batch_size + 1} written.")

        # 3. INDEX SUMMARIES INTO VECTOR STORE (numpy store)
        print("\n" + "="*60)
        print("STEP 5: Indexing Summaries to Vector Store")
        print("="*60)

        from llama_index.embeddings.openai import OpenAIEmbedding
        import json as _json

        STORAGE_DIR = os.path.join(BASE_DIR, "server", "graph_storage")
        npy_path = os.path.join(STORAGE_DIR, "embeddings.npy")
        ids_path = os.path.join(STORAGE_DIR, "embedding_ids.json")

        # Load existing store
        existing_matrix = np.load(npy_path)
        with open(ids_path) as f:
            existing_ids: list = _json.load(f)

        # Remove old COMM_ entries
        keep_mask = [not nid.startswith("COMM_") for nid in existing_ids]
        kept_ids    = [nid for nid, k in zip(existing_ids, keep_mask) if k]
        kept_matrix = existing_matrix[keep_mask]
        print(f"   Removed {len(existing_ids) - len(kept_ids)} old COMM_ entries from vector store.")

        # Embed new summaries
        embed_model = OpenAIEmbedding(
            model_name="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_BASE_URL")
        )

        new_ids: list[str] = []
        new_vecs: list = []
        for comm_id, summary_text in self.hierarchy_summaries[0].items():
            if not summary_text:
                continue
            node_id = f"COMM_{comm_id}"
            vec = embed_model.get_text_embedding(summary_text)
            new_ids.append(node_id)
            new_vecs.append(vec)

        if new_vecs:
            new_matrix = np.array(new_vecs, dtype=np.float32)
            final_matrix = np.vstack([kept_matrix, new_matrix])
            final_ids    = kept_ids + new_ids
        else:
            final_matrix = kept_matrix
            final_ids    = kept_ids

        np.save(npy_path, final_matrix)
        with open(ids_path, 'w') as f:
            _json.dump(final_ids, f)

        print(f"   Added {len(new_ids)} new COMM_ entries. Vector store saved.")
        print(f"   Total vector store size: {len(final_ids)}")
    
    @staticmethod
    def perform_clustering_in_memory(embeddings, ids, k=10, resolution=1.0):
        """
        Performs KNN + Louvain clustering in-memory.
        Returns: dict { cluster_id: [node_id, ...] }
        """
        import community as community_louvain
        
        X = np.array(embeddings)
        n_samples = len(X)
        if n_samples == 0: return {}
        if n_samples == 1: return {0: [ids[0]]}
        
        # KNN
        real_k = min(k, n_samples - 1)
        if real_k < 1: real_k = 1
        
        nbrs = NearestNeighbors(n_neighbors=real_k + 1, metric='cosine').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        G = nx.Graph()
        for i in range(n_samples):
            G.add_node(i)
            
        for i in range(n_samples):
            for j_idx in range(1, indices.shape[1]):
                neighbor_idx = indices[i][j_idx]
                dist = distances[i][j_idx]
                similarity = 1.0 - dist
                if similarity > 0:
                    G.add_edge(i, neighbor_idx, weight=similarity)
                    
        # Louvain
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
        
        clusters = defaultdict(list)
        for idx, label in partition.items():
            clusters[label].append(ids[idx])
            
        return dict(clusters)

    def run_full_pipeline(self):
        """
        Execute the complete GraphRAG clustering pipeline
        """
        print("\n🚀 Starting GraphRAG Clustering Pipeline")
        print("="*60)

        self.create_constrained_edges(write_edges_to_neo4j=False)
        self.leiden_clustering()
        self.generate_summaries()
        self.write_hierarchy_to_db()


def main():
    print("🔗 Connecting to Neo4j and Vector Store...")
    matrix, ids = get_vector_store()
    graph_store  = get_graph_store()

    clustering = GraphClustering(matrix, ids, graph_store)
    clustering.run_full_pipeline()


if __name__ == "__main__":
    main()
