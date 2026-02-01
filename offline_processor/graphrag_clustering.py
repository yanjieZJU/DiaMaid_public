"""
GraphRAG Clustering Pipeline for Neo4j Module Nodes

This script implements:
1. KNN-based similarity edge creation with weights
2. Leiden algorithm for hierarchical clustering (with smart stopping)
3. LLM-based community summarization (with caching and few-shot prompting)
4. Writing community_id back to the database (with cleanup)

Adjusting Cluster granularity:
    - increase the resolution, e.g. res=1.5
    - increase the MIN_SIMILARITY, e.g. MIN_SIMILARITY = 0.6
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
from server.retriever_neo4j import load_stores


# configuration
http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(http_client=http_client)

# Clustering Parameters
KNN_K = 10
MIN_SIMILARITY = 0.5  
MAX_HIERARCHY_LEVELS = 3
MIN_REDUCTION_RATIO = 0.10  # If communities reduce by less than 10%, stop.

# Path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "server", "kg_cluster")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "community_summary_cache.json")

class GraphClustering:
    def __init__(self, vector_store, graph_store):
        self.vector_store = vector_store
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
        r1 = role_a.split(':')[1].strip().lower()
        r2 = role_b.split(':')[1].strip().lower()
        return r1 == r2
        
    def create_constrained_edges(self):
        """
        Step 1: Create KNN-based similarity edges with weights
        """
        print("\n" + "="*60)
        print("STEP 1: Creating KNN Similarity Edges")
        
        # 1.1 Fetch all MODULE nodes from Neo4j
        cypher = """
        MATCH (n:LlamaNode {type: 'MODULE'})
        RETURN n.id as id, n.name as name, n.text as text, n.role as role
        """
        
        with self.graph_store._driver.session() as session:
            result = session.run(cypher)
            for record in result:
                self.module_nodes.append(dict(record))
        
        # 1.2 Get embeddings from vector store
        embedding_dict = self.vector_store._data.embedding_dict
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
        
        with self.graph_store._driver.session() as session:
            # First, remove existing SIMILAR_TO edges to avoid duplicates
            session.run("MATCH ()-[r:SIMILAR_TO]-() DELETE r")
            print("   Cleared existing SIMILAR_TO edges")
            
            # Batch create edges using UNWIND
            batch_size = 500
            for i in range(0, len(edges_to_create), batch_size):
                batch = edges_to_create[i:i+batch_size]
                cypher = """
                UNWIND $edges as edge
                MATCH (a:LlamaNode {id: edge.source})
                MATCH (b:LlamaNode {id: edge.target})
                MERGE (a)-[r:SIMILAR_TO]-(b)
                SET r.weight = edge.weight
                """
                session.run(cypher, edges=batch)
        return len(edges_to_create)
    
    def leiden_clustering(self):
        """
        Step 2: Perform Recursive Leiden clustering with Info Gain Check
        """
        print("\n" + "="*60)
        print("STEP 2: Recursive Leiden Clustering")
        print("="*60)
        
        # 2.1 Build NetworkX graph from Neo4j
        G = nx.Graph()        
        # Add nodes
        for node in self.module_nodes:
            G.add_node(node['id'])
        
        # Add edges with weights
        cypher = """
        MATCH (a:LlamaNode)-[r:SIMILAR_TO]-(b:LlamaNode)
        WHERE a.type = 'MODULE' AND b.type = 'MODULE'
        RETURN a.id as source, b.id as target, r.weight as weight
        """
        
        with self.graph_store._driver.session() as session:
            result = session.run(cypher)
            for record in result:
                G.add_edge(record['source'], record['target'], 
                          weight=record['weight'])
        
        current_graph = G
        import community as community_louvain
        
        prev_community_count = current_graph.number_of_nodes() # Start with N communities
        
        for level in range(MAX_HIERARCHY_LEVELS):
            print(f"Processing Level {level}")
            if current_graph.number_of_nodes() < 2:
                break
                
            res = 1.5 if level == 0 else 0.8 # Slightly looser for higher levels
            partition = community_louvain.best_partition(current_graph, weight='weight', resolution=res)
            unique_comms = set(partition.values())
            curr_community_count = len(unique_comms)
            
            print(f"   Nodes: {current_graph.number_of_nodes()} -> Communities: {curr_community_count}")

            # Check Information Gain / Reduction Ratio
            reduction = (prev_community_count - curr_community_count) / prev_community_count
            print(f"   Reduction Ratio: {reduction:.2%}")
            
            if level > 0 and reduction < MIN_REDUCTION_RATIO:
                 print(f"🛑 Stopping: Reduction {reduction:.2%} < Threshold {MIN_REDUCTION_RATIO:.2%}. Info gain too low.")
                 break

            prev_community_count = curr_community_count

            formatted_partition = {}
            for child_id, raw_comm_id in partition.items():
                new_comm_id = f"L{level}_{raw_comm_id}"
                formatted_partition[child_id] = new_comm_id
                self.level_communities[level].add(new_comm_id)
        
            self.hierarchy_map[level] = formatted_partition
            
            # --- Prepare Next Level ---
            # Check absolute convergence (single root or no change)
            if curr_community_count == 1:
                print("   Stopped: Converged to single root.")
                break
            
            print("   Building SuperGraph for next level ...")
            next_graph = nx.Graph()

            for comm_id in self.level_communities[level]:
                next_graph.add_node(comm_id)

            # Aggregate Edges
            for u, v, data in current_graph.edges(data=True):
                comm_u = formatted_partition[u]
                comm_v = formatted_partition[v]
                if comm_u != comm_v:
                    w = data.get('weight', 1.0)
                    if next_graph.has_edge(comm_u, comm_v):
                        next_graph[comm_u][comm_v]['weight'] += w
                    else:
                        next_graph.add_edge(comm_u, comm_v, weight=w)
            
            current_graph = next_graph
            
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
                        The summary sentence should be 2-3 sentences max. Prioritize mechanisms.
                        If the community contains mixed functionalities, focus on the DOMINANT capability but briefly mention the secondary auxiliary functions.

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
                        temperature=0.3
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
        Step 4: Write hierarchical data and CLEANUP old levels
        """
        print("\n" + "="*60)
        print("STEP 4: Writing Hierarchy to Neo4j")
        print("="*60)
        
        # Prepare a map: NodeID -> {L0_ID, L0_Sum, L1_ID, L1_Sum...}
        node_updates = defaultdict(dict)
        
        # Helper to trace a node up the tree
        def trace_node(node_id):
            # Get L0
            l0_comm = self.hierarchy_map[0].get(node_id)
            if not l0_comm: return
            
            node_updates[node_id]['community_L0'] = l0_comm
            node_updates[node_id]['summary_L0'] = self.hierarchy_summaries[0].get(l0_comm, "")
            
            # Trace up
            curr_comm = l0_comm
            for level in range(1, len(self.hierarchy_map)):
                if curr_comm in self.hierarchy_map[level]:
                    parent = self.hierarchy_map[level][curr_comm]
                    node_updates[node_id][f'community_L{level}'] = parent
                    node_updates[node_id][f'summary_L{level}'] = self.hierarchy_summaries[level].get(parent, "")
                    curr_comm = parent
                else:
                    break
        
        for node in self.module_nodes:
            trace_node(node['id'])
            
        print(f"Preparing to update {len(node_updates)} nodes...")
        
        # Batch Update
        update_data = []
        for nid, props in node_updates.items():
            props['id'] = nid
            update_data.append(props)
            
        with self.graph_store._driver.session() as session:
            # 1. CLEANUP OLD ATTRIBUTES
            print("   🧹 Cleaning up old/stale hierarchy attributes...")
            max_level = len(self.hierarchy_map) - 1
            
            # Remove any L{X} where X > max_level (in case previous runs went deeper)
            # We arbitrarily clean up to Level 5 to be safe
            remove_clauses = []
            for l in range(max_level + 1, 6): 
                remove_clauses.append(f"n.community_L{l}")
                remove_clauses.append(f"n.summary_L{l}")
            
            if remove_clauses:
                remove_str = ", ".join(remove_clauses)
                session.run(f"MATCH (n:LlamaNode) REMOVE {remove_str}")

            # 2. WRITE NEW ATTRIBUTES
            set_clauses = []
            for l in range(max_level + 1):
                set_clauses.append(f"n.community_L{l} = node.community_L{l}")
                set_clauses.append(f"n.summary_L{l} = node.summary_L{l}")
            
            set_clause_str = ",\n".join(set_clauses)
            
            batch_size = 500
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i:i+batch_size]
                cypher = f"""
                UNWIND $batch as node
                MATCH (n:LlamaNode {{id: node.id}})
                SET {set_clause_str}
                """
                session.run(cypher, batch=batch)
                print(f"   Batch {i//batch_size + 1} written.")

        # 3. INDEX SUMMARIES INTO VECTOR STORE
        print("\n" + "="*60)
        print("STEP 5: Indexing Summaries to Vector Store")
        print("="*60)
        
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        # Initialize Embedding Model
        embed_model = OpenAIEmbedding(
            model_name="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_BASE_URL")
        )
        
        nodes_to_index = []
        for level, summaries in self.hierarchy_summaries.items():
            for comm_id, summary_text in summaries.items():
                if not summary_text: continue
                
                # Create a persistent Node for the community
                # ID format: COMM_L{level}_{id}
                # e.g. COMM_L0_12
                node_id = f"COMM_{comm_id}"
                
                node = TextNode(
                    id_=node_id,
                    text=summary_text,
                    metadata={
                        "type": "COMMUNITY_SUMMARY",
                        "level": level,
                        "community_id": comm_id,
                        "name": f"Community {comm_id}"
                    }
                )
                # Compute Embedding
                node.embedding = embed_model.get_text_embedding(summary_text)
                nodes_to_index.append(node)
        
        if nodes_to_index:
            print(f"Adding {len(nodes_to_index)} community summaries to vector store...")
            self.vector_store.add(nodes_to_index)
            self.vector_store.persist(persist_path="./server/graph_storage/default__vector_store.json")
            print("✅ Vector Store Updated and Persisted.")
        else:
            print("No summaries to index.")
    
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
        
        self.create_constrained_edges()
        self.leiden_clustering()
        self.generate_summaries()
        self.write_hierarchy_to_db()


def main():
    # Load stores
    print("🔗 Connecting to Neo4j and Vector Store...")
    vector_store, graph_store = load_stores()
    
    # Initialize clustering
    clustering = GraphClustering(vector_store, graph_store)
    
    # Run full pipeline
    clustering.run_full_pipeline()


if __name__ == "__main__":
    main()
