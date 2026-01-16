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

OUTPUT_FILE = "System_Topology_Knowledge.json"

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
    
    def fetch_workflow_role_sequences(self):
        """
        Fetches ordered sequences of roles for all workflows.
        It links Article -> Workflow -> Modules to also get Application Scenario if needed.
        """
        query = """
        MATCH (a:LlamaNode {type: 'ARTICLE'})-[:HAS_WORKFLOW]->(w:LlamaNode {type: 'WORKFLOW'})
        MATCH (w)-[r:EXECUTES_STEP]->(m:LlamaNode {type: 'MODULE'})
        RETURN 
            a.name as article_name,
            w.id as workflow_id,
            m.role as role,
            m.community_L0 as community,
            r.step_index as step_index,
            a.text as article_text
        ORDER BY workflow_id, step_index
        """
        
        print("Fetching workflow sequences from Neo4j...")
        sequences = collections.defaultdict(list)
        wf_community_map = collections.defaultdict(list)
        wf_text_map = {} # wf_id -> article_text

        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                wf_id = record["workflow_id"]
                val_role = record["role"]
                
                if not val_role or val_role == "unknown":
                    continue
                
                sequences[wf_id].append(val_role)
                
                comm = record.get("community")
                if comm:
                    wf_community_map[wf_id].append(comm)
                    
                if wf_id not in wf_text_map:
                    wf_text_map[wf_id] = record["article_text"]

        # Filter out short sequences
        valid_sequences_map = {k: v for k, v in sequences.items() if len(v) >= 2}
        print(f"Found {len(valid_sequences_map)} valid workflow sequences (length >= 2).")
        return valid_sequences_map, wf_text_map, wf_community_map

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
            
            Task: Summarize the common functionality and application domain of this group in 2-3 sentences. 
            Focus on what kind of systems these are (e.g., "Healthcare management systems", "E-commerce platforms", etc.) and their key capabilities.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
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

    def analyze_transitions(self, sequences):
        """
        Analyzes 1-step transitions: Role A -> Role B
        Returns a nested dict: { RoleA: { next_likely_steps: [ ... ] } }
        """
        transitions = collections.defaultdict(collections.Counter)
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                try:
                    curr = seq[i]
                    next_node = seq[i+1]
                    transitions[curr][next_node] += 1
                except:
                    continue
        
        # Convert to probability format
        topology_knowledge = {}
        
        for current_role, follower_counts in transitions.items():
            total = sum(follower_counts.values())
            next_steps = []
            
            for next_role, count in follower_counts.most_common():
                next_steps.append({
                    "role": next_role,
                    "prob": round(count / total, 3),
                    "count": count
                })
            
            topology_knowledge[current_role] = {
                "next_likely_steps": next_steps,
                # Description could be enriched by LLM later, plain for now
                "description": f"Observed {total} transitions from this role." 
            }
            
        return topology_knowledge

    def analyze_subsequences(self, sequences, length=3):
        """
        Finds common subsequences of a given length.
        """
        subseq_counter = collections.Counter()
        
        for seq in sequences:
            if len(seq) < length:
                continue
            for i in range(len(seq) - length + 1):
                sub = tuple(seq[i : i+length])
                subseq_counter[sub] += 1
                
        # Format
        results = []
        for seq_tuple, count in subseq_counter.most_common(20):
            results.append({
                "sequence": list(seq_tuple),
                "count": count
            })
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
        
        # 1. Global Analysis
        print("Running Global Analysis...")
        global_transitions = self.analyze_transitions(all_sequences)
        global_patterns_3 = self.analyze_subsequences(all_sequences, 3)
        global_patterns_4 = self.analyze_subsequences(all_sequences, 4)
        
        # 2. Cluster Analysis
        # Graph clustering determines clusters automatically
        wf_clusters, cluster_descs = self.cluster_scenarios(wf_metadata_map)
        
        cluster_insights = {}
        for label, wf_ids in wf_clusters.items():
            # Get valid sequences for this cluster
            cluster_seqs = [sequences_map[wid] for wid in wf_ids if wid in sequences_map]
            
            if not cluster_seqs: continue
            
            c_trans = self.analyze_transitions(cluster_seqs)
            c_patt3 = self.analyze_subsequences(cluster_seqs, 3)
            c_comm_dist = self.calculate_community_distribution(wf_ids, wf_community_map)
            
            cluster_insights[str(label)] = {
                "name": cluster_descs[label],
                "workflow_count": len(cluster_seqs),
                "transitions": c_trans,
                "common_patterns_3": c_patt3,
                "community_distribution": c_comm_dist
            }

        # 3. Construct Final Object
        knowledge_base = {
            "metadata": {
                "total_workflows_analyzed": len(all_sequences),
                "generated_at": "offline (with clustering)",
                "version": "1.1"
            },
            "global_stats": {
                "transitions": global_transitions,
                "common_patterns_3": global_patterns_3,
                "common_patterns_4": global_patterns_4
            },
            "scene_clusters": cluster_insights
        }
        
        # 4. Save
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
        print(f"✅ System Topology Knowledge (Clustered) saved to {local_path}")

        self.driver.close()

if __name__ == "__main__":
    analyzer = TopologyAnalyzer()
    analyzer.run()
