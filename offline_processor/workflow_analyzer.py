import os
import sys
import json
import collections
import numpy as np
from sklearn.cluster import KMeans
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
import shutil

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Define output path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "server", "kg_cluster")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "System_Topology_Knowledge.json")

class TopologyAnalyzer:
    def __init__(self):
        print(f"Connecting to Neo4j at {NEO4J_URL}...")
        self.graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URL,
        )
        self.driver = self.graph_store._driver
        # Initialize embedding model for scenario clustering
        self.embed_model = OpenAIEmbedding(
            model_name="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_BASE_URL")
        )
        
        # Initialize LLM Client for claim generation
        import httpx
        from openai import OpenAI
        http_client = httpx.Client(proxy=None, trust_env=False)
        self.client = OpenAI(
            http_client=http_client, 
            api_key=os.environ.get("OPENAI_API_KEY"), 
            base_url=os.environ.get("OPENAI_BASE_URL")
        )

    def fetch_workflow_role_sequences(self):
        """
        Fetches ordered sequences of roles by traversing the FLOWS_TO graph for each article.
        Reconstructs paths from Source Modules -> Sink Modules.
        Returns: sequences_map where values are lists of (role, function_description).
        """
        print("Fetching graph topology from Neo4j...")
        import networkx as nx

        # Query to get all modules and their connections per article
        query = """
        MATCH (a:LlamaNode {type: 'ARTICLE'})
        OPTIONAL MATCH (a)-[:DEFINES_MODULE]->(m:LlamaNode {type: 'MODULE'})
        OPTIONAL MATCH (m)-[:FLOWS_TO]->(next_m:LlamaNode {type: 'MODULE'})
        RETURN 
            a.id as article_id,
            a.text as article_text,
            m.id as m_id,
            m.role as m_role,
            m.community_L0 as m_comm,
            m.function as m_func,
            next_m.id as next_m_id
        """
        
        data_by_article = collections.defaultdict(list)
        node_infos = {} 
        article_texts = {}

        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                aid = record["article_id"]
                if not aid: continue
                
                # Store article text once
                if aid not in article_texts:
                    article_texts[aid] = record["article_text"]
                
                mid = record["m_id"]
                if mid:
                    # Store node info
                    if mid not in node_infos:
                        node_infos[mid] = {
                            "role": record["m_role"],
                            "comm": record["m_comm"],
                            "func": record.get("m_func", "")
                        }
                        if len(node_infos) < 5:
                            print(f"DEBUG: Node {mid} - Role: {record['m_role']}, Comm: {record['m_comm']}")
                    
                    # Store edge (if next_m exists)
                    next_mid = record["next_m_id"]
                    data_by_article[aid].append((mid, next_mid))

        sequences_map = {}
        path_text_map = {}
        path_comm_map = collections.defaultdict(list)

        print(f"Processing topology for {len(data_by_article)} articles...")

        for aid, edges in data_by_article.items():
            G = nx.DiGraph()
            
            # Build graph for this article
            for src, tgt in edges:
                G.add_node(src)
                if tgt:
                    G.add_edge(src, tgt)
            
            if len(G.nodes) < 2: 
                continue
            
            # Identify Sources and Sinks
            sources = [n for n in G.nodes if G.in_degree(n) == 0]
            sinks = [n for n in G.nodes if G.out_degree(n) == 0]
            
            # If graph is fully cyclic or weird, fallback to all nodes?
            # Heuristic: If no sources, pick node withMin in-degree
            if not sources:
                 # Skip cyclic components for now to avoid infinite loops or complexity
                 continue 

            paths = []
            MAX_PATHS = 20 # Limit paths per article to avoid explosion
            
            for s in sources:
                for t in sinks:
                    if s == t: continue
                    try:
                        # Find all simple paths with length limit
                        # We use a cutoff to prevent extremely long looping paths
                        found_paths = list(nx.all_simple_paths(G, source=s, target=t, cutoff=12))
                        paths.extend(found_paths)
                        if len(paths) > MAX_PATHS: break
                    except:
                         pass
                if len(paths) > MAX_PATHS: break
            
            # Deduplicate and Store
            unique_paths = set(tuple(p) for p in paths)
            
            for idx, p_tuple in enumerate(unique_paths):
                # Convert Node IDs to Roles + Functions
                role_func_seq = []
                comm_seq = []
                
                valid = True
                for nid in p_tuple:
                    r = node_infos[nid].get("role")
                    c = node_infos[nid].get("comm")
                    f = node_infos[nid].get("func")
                    
                    if not r or r == "unknown": 
                        pass
                    
                    # Store (role, function) tuple
                    role_func_seq.append((r, f))
                    if c: comm_seq.append(c)
                
                if len(role_func_seq) >= 2:
                    path_id = f"{aid}_p{idx}"
                    sequences_map[path_id] = role_func_seq
                    path_text_map[path_id] = article_texts[aid]
                    path_comm_map[path_id] = comm_seq

        print(f"Found {len(sequences_map)} valid topological paths (length >= 2).")
        return sequences_map, path_text_map, path_comm_map

    def extract_scenario(self, text):
        if not text: return ""
        import re
        # Look for "Application Scenarios: ..."
        match = re.search(r"Application Scenarios:\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback to Function
        match_func = re.search(r"Function:\s*(.*)", text, re.IGNORECASE)
        if match_func:
            return match_func.group(1).strip()
            
        return ""

    def cluster_scenarios(self, wf_text_map):
        """
        Clusters workflows based on their article's 'application_scenarios' using Graph Clustering (KNN + Louvain).
        Returns a dict: { cluster_id: [wf_id1, wf_id2, ...], ... }
        And a dict of cluster descriptions (keywords).
        """
        print("Clustering workflows by Application Scenario (Graph-based)...")
        
        # Import GraphClustering
        from graphrag_clustering import GraphClustering
        from collections import Counter
        from openai import OpenAI
        import httpx
        
        # Initialize OpenAI client
        http_client = httpx.Client(proxy=None, trust_env=False)
        client = OpenAI(http_client=http_client, api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))

        wf_ids = list(wf_text_map.keys())
        scenarios = []
        for wf_id in wf_ids:
            raw_text = wf_text_map.get(wf_id, "")
            scenario = self.extract_scenario(raw_text)
            if not scenario: scenario = "General System"
            scenarios.append(scenario)
            
        # 1. Embed scenarios
        embeddings = self.embed_model.get_text_embedding_batch(scenarios)
        
        # 2. Use GraphClustering static method
        clusters = GraphClustering.perform_clustering_in_memory(embeddings, wf_ids, k=10, resolution=1.0)
        
        # 3. Generate descriptions with LLM
        cluster_descriptions = {}
        # Need to map back from cluster_id -> scenarios to generate description
        # clusters is { label: [wf_id, ...] }
        
        # Create a map wf_id -> scenario
        wf_scenario_map = dict(zip(wf_ids, scenarios))
        
        print("Generating cluster summaries with LLM...")
        for label, member_ids in clusters.items():
            member_scenarios = [wf_scenario_map[wid] for wid in member_ids]
            
            # Prepare context for LLM
            # Limit to top 20 unique scenarios to avoid token limit issues if cluster is huge
            unique_scenarios = list(set(member_scenarios))
            context_text = "\n".join(unique_scenarios[:20])
            
            prompt = f"""
            Analyze the following list of application scenarios/functionalities from various software artifacts:
            
            {context_text}
            
            Task: Synthesize the core characteristics of this cluster in 3-4 sentences.
            1. First, summarize the common functionality and main application domain of this group (e.g., "Healthcare management systems", "Immersive VR Education", etc.).
            2. Second, explicitly highlight the **Innovative Features & Technical Focus** of this cluster. What specific advanced use-cases, unconventional goals, or cutting-edge problem domains distinguish this group of applications from standard systems?
            
            Keep the output strictly to a structured summary paragraph.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating summary for cluster {label}: {e}")
                # Fallback to frequency-based description
                c = Counter(member_scenarios)
                top_scenario, _ = c.most_common(1)[0]
                summary = f"Domain: {top_scenario} (Fallback summary)"

            cluster_descriptions[int(label)] = summary
            print(f"  Cluster {label} ({len(member_ids)} workflows): {summary[:100]}...")
            
        return clusters, cluster_descriptions

    def generate_claims(self, source_role, target_role, context_pairs):
        """
        Generates a semantic 'claim' or rationale for why a transition exists, 
        using specific function descriptions from the modules.
        """
        # Summarize context to avoid token overflow. Take up to 5 examples.
        examples_text = ""
        unique_pairs = list(set(context_pairs))[:5]
        for idx, (s_func, t_func) in enumerate(unique_pairs):
            if s_func and t_func:
                examples_text += f"   Example {idx+1}:\n     Source ({source_role}): {s_func}\n     Target ({target_role}): {t_func}\n"
        
        prompt = f"""
        Analyze the functional transition between two roles in an AI system architecture.
        
        Transition: [{source_role}] -> [{target_role}]
        
        Context (Specific Module Functions observed in this transition):
        {examples_text}
        
        Task: Synthesize a general technical claim explaining what this transition achieves in this type of system.
        Explain the "why" and "how" based on the examples.
        Start with a verb like "Enables...", "Facilitates...", "Transforms...", "Validates...".
        Keep it concise (1 sentence).
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Failed to generate claim for {source_role}->{target_role}: {e}")
            return ""

    def analyze_transitions(self, sequences, enrich_with_llm=False, probability_threshold=0.35):
        """
        Analyzes 1-step transitions: Role A -> Role B
        Sequences are now lists of (role, function) tuples.
        Returns a nested dict: { RoleA: { next_likely_steps: [ ... ] } }
        If enrich_with_llm is True, generates a 'claim' for transitions with prob > threshold.
        """
        # transitions[source_role][target_role] = list of (source_func, target_func)
        transitions = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                try:
                    # unpack role and function
                    curr_role, curr_func = seq[i]
                    next_role, next_func = seq[i+1]
                    
                    if curr_role and next_role:
                        transitions[curr_role][next_role].append((curr_func, next_func))
                except:
                    continue
        
        # Convert to probability format
        topology_knowledge = {}
        
        # Cache claims not just by pair, but maybe we shouldn't cache globally if context differs?
        # Actually, for "Global Analysis", context is global. For "Cluster Analysis", context is cluster-specific.
        # So caching within this function call is fine.
        claim_cache = {} 
        
        print(f"Analyzing transitions (Total source roles: {len(transitions)})...")
        
        for current_role, follower_data in transitions.items():
            # Calculate total transitions from current_role
            total = sum(len(pairs) for pairs in follower_data.values())
            next_steps = []
            
            # Sort by count descending
            sorted_next = sorted(follower_data.items(), key=lambda x: len(x[1]), reverse=True)
            
            for next_role, context_pairs in sorted_next:
                count = len(context_pairs)
                prob = round(count / total, 3)
                
                step_info = {
                    "role": next_role,
                    "prob": prob,
                    "count": count
                }
                
                # Check for claim generation
                if enrich_with_llm and prob >= probability_threshold:
                    pair_key = f"{current_role}->{next_role}"
                    
                    # We generate claim fresh for this call (context specific to this cluster/global set)
                    # But if we have duplicates in the same run loop (unlikely since keys are unique), cache helps.
                    if pair_key not in claim_cache:
                        print(f"  generating claim for high-prob flow: {pair_key} (prob={prob})")
                        claim_cache[pair_key] = self.generate_claims(current_role, next_role, context_pairs)
                    
                    step_info["claim"] = claim_cache[pair_key]
                
                next_steps.append(step_info)
            
            topology_knowledge[current_role] = {
                "next_likely_steps": next_steps
            }
            
        return topology_knowledge

    def generate_pattern_claim(self, role_sequence, context_examples):
        """
        Generates a rationale involves a sequence of roles.
        """
        seq_str = " -> ".join(role_sequence)
        
        examples_text = ""
        # Limit examples
        for idx, func_seq in enumerate(context_examples[:5]):
             # func_seq is a tuple of function descriptions
             steps_text = "\n".join([f"       Step {i+1} ({role_sequence[i]}): {func}" for i, func in enumerate(func_seq)])
             examples_text += f"   Example {idx+1}:\n{steps_text}\n"

        prompt = f"""
        Analyze the common workflow pattern in an AI system.
        
        Pattern Sequence: [{seq_str}]
        
        Context (Real implementation examples):
        {examples_text}
        
        Task: Synthesize a general technical insight explaining the logical flow and purpose of this sequence.
        Explain "how" the data is transformed across these steps.
        Start with "Enables...", "Facilitates...", or similar.
        Keep it to 1 sentence.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return ""

    def analyze_subsequences(self, sequences, length=3, enrich_with_llm=False, probability_threshold=0.2):
        """
        Finds common subsequences of a given length.
        enrich_with_llm: If True, generates claims for top patterns.
        """
        subseq_counter = collections.Counter()
        subseq_instances = collections.defaultdict(list)
        
        for seq in sequences:
            if len(seq) < length:
                continue
            
            # Extract roles and functions
            role_seq = [item[0] for item in seq]
            func_seq = [item[1] for item in seq]
            
            for i in range(len(role_seq) - length + 1):
                sub = tuple(role_seq[i : i+length])
                f_sub = tuple(func_seq[i : i+length])
                
                subseq_counter[sub] += 1
                if enrich_with_llm:
                    subseq_instances[sub].append(f_sub)
                
        # Format
        results = []
        # Get total to calc frequency if needed, or just top N
        total_subseqs = sum(subseq_counter.values())
        
        for seq_tuple, count in subseq_counter.most_common(5): # Top 5 patterns
            pat_entry = {
                "sequence": list(seq_tuple),
                "count": count
            }
            
            if enrich_with_llm and total_subseqs > 0:
                # Generate claim if enough data
                # We can reuse the probability logic or just do it for top 5
                print(f"  Generating claim for pattern: {seq_tuple} (count={count})")
                claim = self.generate_pattern_claim(list(seq_tuple), subseq_instances[seq_tuple])
                pat_entry["claim"] = claim
                
            results.append(pat_entry)
            
        return results

    def calculate_community_distribution(self, wf_ids, wf_community_map):
        """
        Calculates the distribution of functional communities within a group of workflows.
        Returns: { community_id: score (0-1) }
        """
        counts = collections.Counter()
        for wid in wf_ids:
            comms = wf_community_map.get(wid, [])
            counts.update(comms)
        
        total = sum(counts.values())
        if total == 0: return {}
        
        dist = {}
        # Keep top 15 communities to avoid noise
        for comm, count in counts.most_common(15): 
            dist[comm] = round(count / total, 3)
        return dist

    def run(self):
        sequences_map, wf_metadata_map, wf_community_map = self.fetch_workflow_role_sequences()
        all_sequences = list(sequences_map.values())

        # Cluster workflows by application scenario
        wf_clusters, cluster_descs = self.cluster_scenarios(wf_metadata_map)

        cluster_insights = {}
        for label, wf_ids in wf_clusters.items():
            cluster_seqs = [sequences_map[wid] for wid in wf_ids if wid in sequences_map]

            if not cluster_seqs: continue

            c_patt3 = self.analyze_subsequences(cluster_seqs, 3, enrich_with_llm=True)
            c_patt4 = self.analyze_subsequences(cluster_seqs, 4, enrich_with_llm=True)
            c_comm_dist = self.calculate_community_distribution(wf_ids, wf_community_map)

            cluster_insights[str(label)] = {
                "name": cluster_descs[label],
                "workflow_count": len(cluster_seqs),
                "common_patterns_3": c_patt3,
                "common_patterns_4": c_patt4,
                "community_distribution": c_comm_dist
            }

        # Construct Final Object (no global_stats / transitions)
        knowledge_base = {
            "metadata": {
                "total_workflows_analyzed": len(all_sequences),
                "generated_at": "offline (with clustering)",
                "version": "1.2"
            },
            "scene_clusters": cluster_insights
        }

        # Save
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

        print(f"✅ System Topology Knowledge saved to {OUTPUT_FILE}")

        self.driver.close()

if __name__ == "__main__":
    analyzer = TopologyAnalyzer()
    analyzer.run()
