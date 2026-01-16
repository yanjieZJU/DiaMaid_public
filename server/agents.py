import os
import re
import httpx
import json
import sys
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from retriever_neo4j import (
    load_stores, 
    retrieve_and_get_content, 
    find_structural_path
)

http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(
    http_client=http_client,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
    )

TAXONOMY = """
1. **user_entity**
    - target_user: the user who is the final user of the system
    - stakeholder: observer or assistant, e.g., instructor, researcher
2. **sensing_input**
    - bio_sensor: internal human physiological signals, e.g., EEG, EMG, ECG
    - behavior_sensor: exernal human behaviors/movements. e.g., IMU, eye tracker, motion capture
    - env_sensor: environment perception, e.g., RGB Camera, temperature sensor
    - direct_input: direct human-computer interaction input based on interfaces, e.g., Keyboard, Mouse, Touchscreen
3. **processing_core**
    - signal_optimization: making the signal cleaner without changing the data type, e.g., filter, normalizer
    - feature_extraction: convert raw high-dimensional data (images, audio streams) into structured features or symbolic representations, e.g., gaze tracker, pose estimator, speech-to-text
    - state_assessment: qualitative or quantitative evaluation of the extracted features (judge how good/bad/intense something is), e.g., emotion recognition, fatigue detection
    - decision_logic: determine the next step based on the state assessment, e.g., recommender logic, dialog manager (even if it uses an LLM, if the output is a control flag or strategy, it belongs here)
    - content_generation: e.g., text generation (llm, chatbot), image generation (diffusion model)
4. **interface_output**
    - immersive_display: immersive visual output, e.g., VR headset, AR glasses, Unity Scene
    - gui_display: traditional visual display, e.g., Web UI, Mobile screen
    - haptic_feedback: e.g., Vibration motors, exoskeleton feedback
    - audio_output: e.g., TTS, Sound effects
5. **infrastructure**
    - data_storage: storage unit, e.g., vector DB, Database, Knowledge graph
    - communication: e.g., HTTP Server, WebSocket, Bluetooth
"""

class PlannerAgent:
    def _precompute_cluster_embeddings(self):
        if not self.topology_knowledge: return
        
        clusters = self.topology_knowledge.get("scene_clusters", {})
        print(f"Pre-computing embeddings for {len(clusters)} clusters...")
        for label, data in clusters.items():
            text = data.get("name", "")
            if text:
                try:
                    resp = self.client.embeddings.create(input=text, model="text-embedding-3-small")
                    emb = resp.data[0].embedding
                    self.cluster_embeddings[label] = emb
                except Exception as e:
                    print(f"Error embedding cluster {label}: {e}")

    def match_scenario(self, query):
        if not self.cluster_embeddings: return None, None
        
        try:
            resp = self.client.embeddings.create(input=query, model="text-embedding-3-small")
            query_emb = np.array(resp.data[0].embedding).reshape(1, -1)
        except:
            return None, None
            
        best_label = None
        best_score = -1
        
        for label, emb in self.cluster_embeddings.items():
            emb_arr = np.array(emb).reshape(1, -1)
            score = cosine_similarity(query_emb, emb_arr)[0][0]
            if score > best_score:
                best_score = score
                best_label = label
                
        return best_label, best_score

    def __init__(self, client, graph_store):
        self.client = client
        self.graph_store = graph_store
        
        # Load Topology Knowledge
        self.topology_knowledge = {}
        try:
            # Path relative to this file: ../offline_processor/System_Topology_Knowledge.json
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            topo_path = os.path.join(base_dir, "offline_processor", "System_Topology_Knowledge.json")
            with open(topo_path, "r", encoding="utf-8") as f:
                self.topology_knowledge = json.load(f)
            print("✅ Loaded System Topology Knowledge.")
        except Exception as e:
            print(f"⚠️ System Topology Knowledge not found: {e}")
            
        # Cache Cluster Embeddings
        self.cluster_embeddings = {}
        self._precompute_cluster_embeddings()
        self.client = client
        self.graph_store = graph_store  

    def extract_arch(self, user_query):
        """
        Extract explicit architecture from user query.

        Args:
            user_query (str): The user query.

        Returns:
            dict: The extracted architecture.
        """
        
        
        prompt = f"""
            You are a system architect and requirement analyst.
            
            ## Task
            Given a user query with a user-constructed workflow, output its normalized structure in JSON code.
            1. Identify mandatory components (anchors) and the conenction topology (including parallel connections, data flows, subgraphs, etc. if any) from user query. It might involve the functions that need to be implemented, the technical modules, fixed inputs/outputs, etc.
            2. Complete the design goal based on these anchors and relations which were not originally included in the original query.
            3. For each anchor: based on the user's requirements and the current context, infer the user's expectations for each anchor (their role within the entire workflow), and formulate a concise description.
            4. Extract ALL user defined relations (explicit) from the user query. For example, the arrow '-->' means the data flow from source to target, the use of subgraph means the inclusion relationship. DO NOT REPEAT.
            5. For each anchor, you must also infer its functional role based on the following taxonomy:
            {TAXONOMY}\n\n
            (Output the Level 1 category, or 'Level 1: Level 2' if you are sure)
            Output Structure (in JSON format)
            {{
                "design_goal": "...",
                "anchors": [
                    {{
                        "user_term": "the user term for this anchor, e.g., AI tutor",
                        "role": "the role of this anchor, choose from [user_entity, sensing_input, processing_core, interface_output, infrastructure]",
                        "community_id": "the ID of the matched functional cluster (from Available System Capabilities), or null if no clear match",
                        "description": "the description of this anchor, e.g., answer questions about the user's topic."
                    }}
                ],
                "relations":[
                    {{
                        "source": "the source anchor of this relation, e.g., AI tutor",
                        "target": "the target anchor of this relation, e.g., user",
                        "type": "the type of this relation, e.g., DATAFLOW, CONTAINS"
                    }}
                ]
            }}

            User Query: {user_query}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            print(f"LLM Response: {str(response)}")
        except Exception as e:
            print(f"Error: {e}")
            return None
        try:
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return None
        
    def divergent_planning(self, user_query, base_anchors, num_outputs=2):
        """
        Generates divergent extension plans based on the user query and already extracted base anchors.
        
        Args:
            user_query (str): The original user query.
            base_anchors (list): List of anchors already extracted by extract_arch.
            num_outputs (int): The number of distinct divergent plans to generate.

        Returns:
            list: A list of dicts, each containing 'perspective' and 'additional_anchors'.
        """
        
        # Format base anchors for the prompt
        base_anchors_str = json.dumps([a['user_term'] for a in base_anchors], ensure_ascii=False)
        
        # 1. Match Scenario & Get Context
        cluster_id, score = self.match_scenario(user_query)
        relevant_communities = []
        
        if cluster_id:
            cluster_data = self.topology_knowledge["scene_clusters"][cluster_id]
            print(f"Matched Scenario Cluster: {cluster_data['name'][:200]}... (Score: {score:.2f})")
            
            # A. Get Relevant Communities
            comm_dist = cluster_data.get("community_distribution", {})
            # Get top 8 communities
            top_comms = sorted(comm_dist.items(), key=lambda x: x[1], reverse=True)[:8]
            relevant_communities = [c[0] for c in top_comms]
            
            # pattern_cypher = """
            # MATCH (n:LlamaNode)-[r:FLOWS_TO]->(m:LlamaNode)
            # WHERE (n.community_L0 IN $comm_ids OR m.community_L0 IN $comm_ids)
            # AND n.name IS NOT NULL AND m.name IS NOT NULL
            # AND id(n) <> id(m)
            # RETURN distinct n.role as src_role, n.name as src_name, m.role as tgt_role, m.name as tgt_name
            # LIMIT 15
            # """
            # concrete_patterns = []
            # with self.graph_store._driver.session() as session:
            #     params = {"comm_ids": relevant_communities if relevant_communities else []}
            #     result = session.run(pattern_cypher, **params)
            #     for record in result:
            #         src_str = f"[{record['src_role']}] {record['src_name']}"
            #         tgt_str = f"[{record['tgt_role']}] {record['tgt_name']}"
            #         concrete_patterns.append(f"{src_str} -> {tgt_str}")

            # if not concrete_patterns and "transitions" in cluster_data:
            #     transitions = cluster_data.get("transitions", {})
            #     for role, data in transitions.items():
            #         top_next = sorted(data.get("next_likely_steps", []), key=lambda x: x["prob"], reverse=True)[:2]
            #         if top_next:
            #             next_roles = ", ".join([f"{n['role']} ({n['prob']})" for n in top_next])
            #             concrete_patterns.append(f"- {role} -> [{next_roles}] (Abstract Pattern)")
            
            # feasible_patterns_text = "\n".join(concrete_patterns[:10]) # Limit to 10
        
        # 2. Fetch Capabilities (Targeted or Global)
        print(f"Fetching capabilities for {len(relevant_communities) if relevant_communities else 'ALL'} communities...")
        
        cypher = """
        MATCH (n:LlamaNode) 
        WHERE n.community_L0 IS NOT NULL
        AND ($comm_ids IS NULL OR n.community_L0 IN $comm_ids)
        RETURN DISTINCT n.community_L0 as id, n.summary_L0 as summary
        LIMIT 50
        """
        summaries = []
        with self.graph_store._driver.session() as session:
            # If relevant_communities is empty, pass None to get global (limited)
            params = {"comm_ids": relevant_communities if relevant_communities else None}
            result = session.run(cypher, **params)
            for record in result:
                summaries.append(f"[Community ID: {record['id']}]: {record['summary']}")
        
        available_capabilities = "\n".join(summaries)

        prompt = f"""
        You are a creative System Architect.
        
        ## Context
        The user wants to design a system described as: ```{user_query}```.
        We have already identified the following BASE components (Anchors): ```{base_anchors_str}```.
        
        ## Task
        Generate **{num_outputs}** DISTINCT and DIVERGENT extension plans to enhance this system. 
        
        ## Requirements
        1. **Distinct Perspectives**: 
           - Each plan should explore a DIFFERENT technical perspective or feature set that complements the base components.
           - Based on the new perspectives, you can either supplement the existing anchors with additional details or introduce new modules.
           - Your expansion should be based on user needs, and should supplement and improve the current process form different perspectives, so as to make it have more specific processing methods and richer ditails and functions.
        
        2. **Incremental Only**: 
           - Do NOT repeat the Base Anchors unless there's a special need to modify or improve them. 
           - Output ONLY the ADDITIONAL components (modules) needed for the extension.
           
        3. **Specific Anchors**:
           - Each additional component belongs to a ```function module``` and can be connected to the existing components. Before you output the final plan, please recheck if the data flow is correct and reasonable, and meet user needs.
           - For each additional component, provide a specific 'user_term', a functional 'description', a precise 'role', and a 'community_id' if applicable.

        ### Role Taxonomy
        {TAXONOMY}\n\n

        ## Available System Capabilities (from Knowledge Base)
            The following are the existing functional clusters in our component library. 
            Try to map the user's requirements to these capabilities where possible to ensure availability.
            {available_capabilities}


        ## Output Structure (JSON)
        {{
            "plans": [
                {{
                    "perspective": "Short title of the perspective (e.g., AI-Enhanced)",
                    "rationale": "Why this extension adds value",
                    "additional_anchors": [
                        {{
                            "user_term": "Specific Module Name",
                            "description": "Functional description of what it does",
                            "role": "category: subcategory",
                            "community_id": "ID from Available Capabilities or null"
                        }}
                    ]
                }},
                ... (total {num_outputs} plans)
            ]
        }}
        """
        print(prompt)
        print(f"\n🧠 Generating {num_outputs} divergent plans...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9, # Higher temperature for creativity
                response_format={"type": "json_object"},
            ).choices[0].message.content
            
            # print(f"LLM Response: {str(response)}")
            result = json.loads(response)
            plans = result.get("plans", [])
            
            # Validate output count
            if len(plans) < num_outputs:
                print(f"⚠️ Warning: LLM generated {len(plans)} plans, requested {num_outputs}.")
            
            return plans[:num_outputs]

        except Exception as e:
            print(f"❌ Error in divergent planning: {e}")
            return []

        
        
        
class RetrieverAgent:
    def __init__(self, vector_store, graph_store, client):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.client = client

    def retrieve_for_module(self, anchors_list, top_k=3, include_topology=True):
        anchor_mapping = {}
        for anchor in anchors_list:
            user_term = anchor["user_term"]
            description = anchor["description"]
            role = anchor.get("role", None) # Get role
            community_id = anchor.get("community_id", None) # Get structured community ID

            query_text = f"{user_term}: {description}"
        
            retrieved_nodes = retrieve_and_get_content(
                query_text, 
                self.vector_store, 
                self.graph_store, 
                top_k,
                node_type="MODULE",
                role_filter=role, # Pass role filter
                community_id=community_id, # Pass community filter
                include_topology=include_topology
            )

            if retrieved_nodes:
                anchor_mapping[user_term] = retrieved_nodes
                for n in retrieved_nodes:
                    print(f"Found: {n.node.metadata.get('name', '')} (Score: {n.score:.4f})")

        return anchor_mapping
    
    def retrieve_for_skeleton(self, design_goal, threshold=0.5):
        query_text = f"{design_goal}"
        retrieved_nodes = retrieve_and_get_content(
            query_text, 
            self.vector_store, 
            self.graph_store, 
            top_k=10,
            node_type="WORKFLOW",
        )
        valid_nodes = []
        # for node_with_score in retrieved_nodes:
        #     if node_with_score.score >= threshold:
        #         valid_nodes.append(node_with_score.node)
        return retrieved_nodes
    
    def retrieve_structure_paths(self, user_relations):
        structral_hints = []
        for rel in user_relations:
            src = rel["source"]
            tgt = rel["target"]
            paths = find_structural_path(src, tgt, rel_type=rel["type"], vector_store=self.vector_store, graph_store=self.graph_store)
            structral_hints.extend(paths)

        return structral_hints
    
    def rerank_with_llm(self, query, retrieved_skeletons):
        if not retrieved_skeletons: return []
        candidates_text = ""
        for i, node in enumerate(retrieved_skeletons):
            content_preview = node.node.text[:500].replace("\n", " ")
            candidates_text += f"--- Candidate {i} ---\n {content_preview}\n\n"
        prompt = f"""
        ##TASK:
        Identify the SINGLE candidate from the list below that BEST matches the user's specific requirements.

        ##CRITERIA:
        1. The candidate must be the most relevant to the user's requirements.
        2. The candidate must be the most complete to include the specific nodes mentioned in the user's query.

        ##Candidates:
        {candidates_text}

        ##User Query:
        {query}

        ##Output Format:
        Return the top 3 indices of the best candidates in order of preference (e.g., "1, 2, 0").

        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            ).choices[0].message.content
            indices = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
            valid_skeletons = []
            for i in indices:
                if 0 <= i < len(retrieved_skeletons):
                    valid_skeletons.append(retrieved_skeletons[i])

            if len(valid_skeletons) < 3:
                remaining_indices = [idx for idx in range(len(retrieved_skeletons)) if idx not in indices]
                for idx in remaining_indices:
                    if len(valid_skeletons) < 3:
                        valid_skeletons.append(retrieved_skeletons[idx])

            return valid_skeletons
            
        except Exception as e:
            print(f"Error: {e}")
            return None


class IntegratorAgent:
    def __init__(self, client):
        self.client = client

    def format_hints(self, structural_hints):
        if not structural_hints:
            return ""
        
        hints_text = "### Validated Connection Paths from Literature\n"
        for i, path in enumerate(structural_hints):
            chain_desc = []
            nodes = path['path_nodes']
            rels = path.get('relations', [])
            rel_props = path.get('rel_props', [])
            
            # Build chain description
            chain_desc.append(f"[{nodes[0]['name']}]")
            for j, rel in enumerate(rels):
                target_node = nodes[j+1]['name']
                
                # Extract edge label if available
                data_content = ""
                if j < len(rel_props) and rel_props[j] and 'edge_label' in rel_props[j]:
                    lbl = rel_props[j]['edge_label']
                    if lbl and lbl != "null":
                        data_content = f": {lbl}"

                if rel == "FLOWS_TO":
                    chain_desc.append(f" --(flows to{data_content})--> [{target_node}]")
                elif rel == "HAS_ROLE":
                        chain_desc.append(f" ==(equivalent role)==> [{target_node}]")
                else:
                        chain_desc.append(f" --({rel}{data_content})--> [{target_node}]")

            hints_text += f"   * Connection '{path['source_query']} -> {path['target_query']}':\n"
            hints_text += f"     Reference Path: {''.join(chain_desc)}\n"
            
            # Smartly extract intermediate suggestions
            # Only suggest nodes that are part of a data flow (outgoing relation is FLOWS_TO)
            # This filters out nodes that are just role-equivalence jumps
            real_intermediates = []
            for k in range(1, len(nodes)-1):
                if k < len(rels) and rels[k] == "FLOWS_TO":
                        real_intermediates.append(nodes[k]['name'])

            if real_intermediates:
                hints_text += f"Suggestion: You might need intermediate modules like: {', '.join(real_intermediates)}\n"
        hints_text += "\n"
        return hints_text
    
    def integrate_arch(self, query, skeletons, retrieved_info, structural_hints=[], added_nodes=[]):
        """
        Integrates all information to generate the final system architecture.

        Args:
            query (str): The user query.
            skeletons (list): Reference system skeletons.
            retrieved_info (dict): Retrieved specific module information.
            structural_hints (list): Structural paths found between components.
            added_nodes (list): New nodes introduced by the divergent planner.

        Returns:
            str: The final integrated architecture description.
        """
        skeletons_text = ""
        if skeletons:
            for i, s in enumerate(skeletons):
                skeletons_text += f"--- Skeleton {i} ---\n {s.node.text}\n\n"

        hints_text = self.format_hints(structural_hints)

        added_nodes_text = ""
        if added_nodes:
            added_nodes_text = "\n\n".join([f"{node['user_term']} ({node['description']})" for node in added_nodes])

        # Slim down retrieved_info to reduce noise and token usage
        # Identify added anchor keys for special handling
        added_keys = set()
        if added_nodes:
            added_keys = {n['user_term'] for n in added_nodes}

        slim_retrieved_info = {}
        for anchor, nodes in retrieved_info.items():
            slim_nodes = []
            for n in nodes:
                node_data = {
                    "module": n.get('metadata', {}).get('name', 'Unknown'),
                    "content": n.get('text', '')
                }
                
                # If this is an added node, we explicitly preserve its topology 
                # to help the LLM understand how to connect this new component
                if anchor in added_keys:
                    # Try to get raw topology from metadata as a cleaner signal
                    raw_topo = n.get('structural_path', None) # We stored it here in run_pipeline
                    if not raw_topo: 
                        raw_topo = n.get('metadata', {}).get('topology', None)
                        
                    if raw_topo:
                        # Simplify topology data to avoid huge dumps
                        simplified_topo = {
                            "upstream": [u['name'] for u in raw_topo.get('upstream', [])],
                            "downstream": [d['name'] for d in raw_topo.get('downstream', [])]
                        }
                        node_data["structural_context"] = simplified_topo
                
                slim_nodes.append(node_data)
            slim_retrieved_info[anchor] = slim_nodes

        prompt = f"""
        You are a professional designer and researcher in the field of HCI. Design a complete pipeline based on the user's requirements and the retrieved information.
        ## User Requirements
        The user requirements are the principle you should follow the most. The nodes in the query must be included in the pipeline.
        User Query: ```{query}```
        \n\n
        ## Retrieved Information
        You have three sources of information. Use them according to the following rules.
        1. **Retrieved Modules**: 
            - These are verified modules found in the literature for each conceptual step in the user's requirements, incorporating the technical methods, data processing approaches, and the specific algorithms.
            - Use them to fill in the technical implementation of these specific nodes.
            - Each module may have retrieved multiple nodes with similar functions. You need to conduct appropriate screening to be able to assemble a complete process.
        2. **Structural Paths**: 
            - These are retrieved connection paths for explicit data flows between nodes in the user's requirements.
            - Use them to determine HOW the user's nodes should be connected.
            - If the paths suggest an intermediate node is needed (e.g., Sensor -> [Buffer] -> LLM), you should add it.
        3. **Reference Skeletons**: 
            - These are retrieved workflows found in the literature that are similar to the user's requirements.
            - Use them ONLY to enrich the pipeline, that is, to borrow auxiliary modules (like databases, UI, pre-processors) from the Skeleton to make it a complete, robust system.
        For the entire system architecture, here is the most similar system found in the literature.

        ## Data Sources
        Added Nodes by Divergent Planner: ```{added_nodes_text}```\n\n
        Retrieved Modules: ```{json.dumps(slim_retrieved_info, ensure_ascii=False, indent=2)}```\n\n
        Structural Paths: ```{hints_text}```\n\n
        Reference Skeletons: ```{skeletons_text}```\n\n
        
        ## TASK
        Select the appropriate nodes from the retrieved nodes, conceive the complete data link and functional architecture, and ultimately synthesize a cohesive flowchart/architecture diagram. 
        ### Critical Rules
        1. **Connect the Flow**: Use the topology info to decide logical connections. If Module A usually follows Module B in Paper X, try to preserve that flow if it fits.
        2. **Cite Sources**: Reference the original papers where the nodes are from.
        3. **Redundancy Check**: REMOVE redundant nodes and connections. 
        4. **Incorporate Divergent Ideas**: Ensure the added nodes from the divergent planner are integrated meaningfully.
        
        Final Output Structure (in JSON format)
        {{
            "overall_description": "...",
            "module_details": [
                "<module name>": {{
                    "name": "the module name",
                    "function": "the functions of this module",
                    "technical_methods": "the technical methods used in this module",
                    "references": "the code of the referenced module (if any). Use 'None' if not available.",
                    "sub_nodes":[
                        "<sub_node name (if any)>": {{
                            "name": "the sub node name",
                            "function": "the functions of this sub node",
                            "technical_methods": "the technical methods used in this sub node",
                        }}
                    ],
                    "ports": {{
                        "incoming (if any)": [
                            {{
                                "source_mod": "the identifier of the incoming module",
                                "source_node (if any)": "the identifier of the incoming node",
                                "edge_label": "the label of the incoming edge, i.e., the meaning of the data flow",
                            }}
                        ],
                        "outgoing (if any)": [
                            {{
                                "target_mod": "the identifier of the outgoing module",
                                "target_node (if any)": "the identifier of the outgoing node",
                                "edge_label": "the label of the outgoing edge, i.e., the meaning of the data flow",
                            }}
                        ]
                    }}
                }}
            ]
        }}
    """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                response_format={"type": "json_object"},
            ).choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None
        try:
            print(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return None
        return response
    
class DrawerAgent:
    def __init__(self, client):
        self.client = client
    
    def draw_flowchart(self, arch_plan):
        prompt = f"""
        ## Task
        Translate the given system architecture/pipeline into a Mermaid flowchart.
        ## The input structure:
        {{
            "overall_description": "...",
            "module_details": [
                "<module name>": {{
                    "name": "the module name",
                    "function": "the functions of this module",
                    "technical_methods": "the technical methods used in this module",
                    "references": "the definition code of this module in the reference paper.",
                    "sub_nodes":[
                        "<sub_node name (if any)>": {{
                            "name": "the sub node name",
                            "function": "the functions of this sub node",
                            "technical_methods": "the technical methods used in this sub node",
                        }}
                    ],
                    "ports": {{
                        "incoming (if any)": [
                            {{
                                "source_mod": "the identifier of the incoming module",
                                "source_node (if any)": "the identifier of the incoming node",
                                "edge_label": "the label of the incoming edge, i.e., the meaning of the data flow",
                            }}
                        ],
                        "outgoing (if any)": [
                            {{
                                "target_mod": "the identifier of the outgoing module",
                                "target_node (if any)": "the identifier of the outgoing node",
                                "edge_label": "the label of the outgoing edge, i.e., the meaning of the data flow",
                            }}
                        ]
                    }}
                }}
            ]
        }}
        ## INTERPRETATION RULES:
            1. **Keep align with the plan**: The flowchart should strictly follow the plan. 
                a. Nodes are only represented by their ids and names, without any other information appearing in the diagram (including functions, technical methods, and references).
                b. Use subgraphs to represent modules if there are **more than one subnodes and data flow between them** in a single module. DO NOT use subgraphs if there is only one node and the name of the node is the same as the module name.
                c. The definition codes within the module are included in the "references" section, which should be contained in the code of the module (subgraph) if necessary.
                d. The connection and data flow between nodes/modules should display through the label on the connection line.
                e. Do not add any extra nodes or edges.
            2. **Adhere strictly to the grammar of Mermaid**: ensure the final code does not generate any errors when compiled. 
                a. Always enclose the labels with double quatation marks.
                b. Do not include style attributes.
            3. **Use the code in the reference with caution.** Modify the names and descriptions of each node and relationship according to the user's requirements and context, so that the user can fully understand this flowchart.

        ## REFLECTION:
        Before outputting the final code, please reflect and recheck the code.
        1. The **subgraph** has been correctly defined and an end marker "end" has been added.
        2. The name of each node (enclosed in parentheses or squares brackets) do not contain any other parentheses or brackets, or are correctly enclosed within double quotes.
        3. NO style attributes, NO "click" events, NO "class" identification.
        
        Final Output Structure (in JSON format)
        {{
            "mermaid_code": "...",
        }}

        ## The input plan:
        {arch_plan}
    """
        try: 
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            raw_code = json.loads(response)["mermaid_code"]
            clean_code = self.clean_mermaid_code(raw_code)
            return clean_code
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def clean_mermaid_code(self,code):
        """
        清理Mermaid代码中的转义字符
        """
        # 替换转义字符
        code = code.replace('\\n', '\n')
        code = code.replace('\\"', '"')
        if code.startswith('"') and code.endswith('"'):
            code = code[1:-1]
        # 移除多余的空白字符
        code = code.strip()

        if '```mermaid' in code:
            start = code.find('```mermaid') + len('```mermaid')
            end = code.find('```', start)
            code = code[start:end]
        elif '```' in code:
            parts = code.split('```')
            if len(parts) > 2:
                code = parts[1]
        
        return code
            
def run_pipeline(user_query, num_outputs=3):
    vector_store, graph_store = load_stores()
    planner = PlannerAgent(client, graph_store)
    retriever = RetrieverAgent(vector_store, graph_store, client)
    integrator = IntegratorAgent(client)
    drawer = DrawerAgent(client)

    arch_plan = planner.extract_arch(user_query)
    # print(arch_plan)
    skeletons = retriever.retrieve_for_skeleton(arch_plan["design_goal"])
    filtered_skeletons = retriever.rerank_with_llm(user_query, skeletons)
    user_relations = arch_plan["relations"]

    target_skeletons = filtered_skeletons[:num_outputs]
    
    results = []
    structural_hints = retriever.retrieve_structure_paths(user_relations)
    
    # Print structural hints once
    if structural_hints:
        print(integrator.format_hints(structural_hints))

    # === Pre-calculate Divergent Plans for ALL Options ===
    # Notice: We generate divergent plans for ALL requested outputs
    divergent_plans = planner.divergent_planning(
        user_query=user_query,
        base_anchors=arch_plan["anchors"],
        num_outputs=num_outputs
    )

    # === Step 1: Global Retrieval for Base Anchors (Higher Recall) ===
    print(f"\n🎯 Retrieving Base Anchors (Top-6 for richer context)...")
    # Base anchors are standard components, we don't need their confusing local topology
    base_nodes_map = retriever.retrieve_for_module(arch_plan["anchors"], top_k=6, include_topology=False)
    
    # Process base nodes into standard format
    base_ref_info = {}
    for anchor_name, retrieved_list in base_nodes_map.items():
        anchor_nodes = []
        for node_with_score in retrieved_list:
            # Handle potential None result if retrieval failed partially
            if not node_with_score: continue
            
            node = node_with_score.node
            anchor_nodes.append({
                'id': node.id_,
                'text': node.text,
                'score': node_with_score.score,
                'structural_path': node.metadata.get('topology', None),
                'metadata': dict(node.metadata)
            })
        base_ref_info[anchor_name] = anchor_nodes

    # === Generate Options ===
    # TODO: skeleton在这里的作用先暂时忽略，可能后续作为divergent_planner的参考
    for i in range(num_outputs):
        # Use skeletons cyclically if we have fewer skeletons than requested outputs
        # skeleton_idx = i % len(target_skeletons)
        # s = target_skeletons[skeleton_idx]
        
        print(f"\n{'='*60}")
        # print(f"=== Generating Option {i+1} using Skeleton {skeleton_idx} ===")
        print(f"{divergent_plans[i]}\n")
        
        current_ref_info = base_ref_info.copy()

        # Apply Divergent Plan (if available)
        
        d_plan = divergent_plans[i]
        additional_anchors = d_plan.get("additional_anchors", [])
        perspective = d_plan.get("perspective", "Enhanced")
        rationale = d_plan.get("rationale", "")
        
        print(f"Applying Divergent Perspective: {perspective}")
        print(f"Rationale: {rationale}")
        
        if additional_anchors:
            # Divergent anchors are new ideas, their topology is valuable context
            additional_map = retriever.retrieve_for_module(additional_anchors, top_k=3, include_topology=True)
            for anchor_name, retrieved_list in additional_map.items():
                # Note: We append, so if multiple plans use same anchor name, it overwrites. 
                # Ideally divergent anchors have unique user_terms.
                current_ref_info[anchor_name] = []
                for node_with_score in retrieved_list:
                    node = node_with_score.node
                    current_ref_info[anchor_name].append({
                        'id': node.id_,
                        'text': node.text,
                        'score': node_with_score.score,
                        'structural_path': node.metadata.get('topology', None),
                        'metadata': dict(node.metadata)
                    })
        else:
            print(f"No additional anchors suggested for this plan")

        additional_desc = f"Perspective: {perspective}\nRationale: {rationale}\n"
        
        print(f"\n📦 Retrieved modules summary for Option {i+1}:")
        print(f"Base Anchors: {len(base_ref_info)} keys")
        print(f"Total Anchors: {len(current_ref_info)} keys")
        
        # Generate architecture using the MERGED retrieved modules
        new_arch = integrator.integrate_arch(
            query=user_query+"; Perspective: "+d_plan.get("perspective", "")+"; Rationale: "+d_plan.get("rationale", ""), 
            # skeletons=[s],
            skeletons=None, 
            retrieved_info=current_ref_info, 
            structural_hints=structural_hints,
            added_nodes=additional_anchors
        )
        
        if new_arch:
            mermaid_code = drawer.draw_flowchart(new_arch)
            
            # Add perspective info to description
            raw_desc = json.loads(new_arch).get("overall_description", "")
            final_desc = additional_desc + raw_desc
                
            results.append({
                "option_id": i + 1,
                "perspective": perspective,
                # "based_on_skeleton_score": s.score,
                "mermaid_code": mermaid_code,
                "description": final_desc
            })
            print(f"\n--- Mermaid Code (Option {i+1}) ---\n{mermaid_code}\n")
        else:
            print(f"❌ Failed to generate architecture for option {i+1}")

    return results


if __name__ == "__main__":
    query = "design a VR system supporting gesture interaction"
    vector_store, graph_store = load_stores()
    planner = PlannerAgent(client, graph_store)
    arch_plan = planner.extract_arch(query)
    divergent_plans = planner.divergent_planning(query, arch_plan["anchors"], num_outputs=3)
    print(divergent_plans)
    