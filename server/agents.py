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
    - primary_user: the user who is the final user of the system
    - secondary_user: stakeholder, observer or assistant, e.g., instructor, researcher
    - agents: non-human agents that can running independently (outside the system), e.g., robots, collaborative AI
2. **sensing_input**
    - bio_sensor: internal human physiological signals, e.g., EEG, EMG, ECG
    - behavior_sensor: external human behaviors/movements. e.g., IMU, eye tracker, motion capture
    - env_sensor: environment perception, e.g., RGB Camera, temperature sensor
    - direct_input: direct human-computer interaction input based on interfaces, e.g., Keyboard, Mouse, Touchscreen
    - system_context: equipment, system status and application log, etc.
    - data_import: offline documents, knowledge base, third-party API stream that serve as inputs
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
    - model_lifecycle: model repository/version, training and post-training pipeline, online learning, etc. 
    - personalization_store: user persona, prefenrence, threshold / calibration parameters, etc.
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
            # Path relative to this file: ./kg_cluster/System_Topology_Knowledge.json
            base_dir = os.path.dirname(os.path.abspath(__file__)) # server dir
            topo_path = os.path.join(base_dir, "kg_cluster", "System_Topology_Knowledge.json")
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
            1. Identify mandatory components (anchors) and the connection topology (including parallel connections, data flows, subgraphs, etc. if any) from user query. 
            2. Extract **ONLY** the entities explicitly mentioned in the user query or those strictly implied as standard infrastructure (e.g., a database if "save data" is mentioned). **DO NOT** invent new functional modules that were not requested.
            3. For each anchor: based on the user's requirements and the current context, infer the user's expectations for each anchor (their role within the entire workflow), and formulate a concise description.
            4. Extract ALL user defined relations (explicit) from the user query. For example, the arrow '-->' means the data flow from source to target, the use of subgraph means the inclusion relationship. DO NOT REPEAT.
            5. For each anchor, you must also infer its functional role based on the following taxonomy:
            {TAXONOMY}\n\n
            (Output the Level 1 category, or 'Level 1: Level 2' if you are sure)
            Output Structure (in JSON format)
            {{
                "design_goal": "A concise summary of the system's purpose based strictly on the query.",
                "anchors": [
                    {{
                        "user_term": "the user term for this anchor, e.g., AI tutor",
                        "role": "the role of this anchor, choose from [user_entity, sensing_input, processing_core, interface_output, infrastructure]",
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
        Uses Knowledge-Distance Perspective Injection to find both near-field and far-field inspirations.
        """
        
        # Format base anchors for the prompt
        base_anchors_str = json.dumps([a['user_term'] for a in base_anchors], ensure_ascii=False)
        
        # 1. Calculate similarities with ALL clusters
        if not self.cluster_embeddings: 
            print("No cluster embeddings available for divergent planning.")
            return []

        try:
            resp = self.client.embeddings.create(input=user_query, model="text-embedding-3-small")
            query_emb = np.array(resp.data[0].embedding).reshape(1, -1)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

        scored_clusters = []
        for label, emb in self.cluster_embeddings.items():
            emb_arr = np.array(emb).reshape(1, -1)
            score = cosine_similarity(query_emb, emb_arr)[0][0]
            scored_clusters.append((label, score))
            
        # Sort by score descending
        scored_clusters.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Select Reference Clusters (Near vs Far)
        # Strategy:
        # - Plan 0 (if num_outputs > 0) -> Near Field (Top 1) -> "Deep Dive"
        # - Plan 1..N -> Far Field (Random from score 0.3 - 0.6) -> "Cross-domain Transfer"
        
        # Enforce max outputs to 3 as requested
        num_outputs = min(max(1, num_outputs), 3)

        selected_references = []
        
        # A. Near Field (Top 1)
        if scored_clusters:
            near_cluster_id = scored_clusters[0][0]
            near_score = scored_clusters[0][1]
            selected_references.append({
                "type": "NEAR_FIELD",
                "cluster_id": near_cluster_id,
                "score": near_score,
                "name": self.topology_knowledge["scene_clusters"][near_cluster_id]["name"]
            })
            print(f"🎯 Near-Field Context: {self.topology_knowledge['scene_clusters'][near_cluster_id]['name'][:50]}... (Score: {near_score:.2f})")

        # B. Far Field (Random Sampling)
        near_id_set = {scored_clusters[0][0]} if scored_clusters else set()
        far_candidates = [c for c in scored_clusters if 0.3 <= c[1] <= 0.6 and c[0] not in near_id_set]
        
        needed_far = max(0, num_outputs - 1)
        if far_candidates and needed_far > 0:
            import random
            # Sample without replacement if possible
            k = min(len(far_candidates), needed_far)
            chosen_far = random.sample(far_candidates, k)
            
            for c_id, c_score in chosen_far:
                selected_references.append({
                    "type": "FAR_FIELD",
                    "cluster_id": c_id,
                    "score": c_score,
                    "name": self.topology_knowledge["scene_clusters"][c_id]["name"]
                })
                print(f"🚀 Far-Field Context: {self.topology_knowledge['scene_clusters'][c_id]['name'][:50]}... (Score: {c_score:.2f})")
        
        # Fill remaining with random near/mid if we ran out of far candidates
        while len(selected_references) < num_outputs:
             # Just pick next best near ones or randoms
             # for simplicity, just pick from top list excluding already picked
             existing_ids = set(r["cluster_id"] for r in selected_references)
             residue = [c for c in scored_clusters if c[0] not in existing_ids]
             if not residue: break
             c_id, c_score = residue[0]
             selected_references.append({
                "type": "SUPPLEMENTAL",
                "cluster_id": c_id,
                "score": c_score,
                "name": self.topology_knowledge["scene_clusters"][c_id]["name"]
             })
        
        # 3. Retrieve Capabilities for specific references
        # We want to give the LLM specific "Reference Packs"
        
        reference_context_str = ""
        
        for idx, ref in enumerate(selected_references):
            c_id = ref["cluster_id"]
            cluster_data = self.topology_knowledge["scene_clusters"][c_id]
            
            # Get top N communities for this cluster
            comm_dist = cluster_data.get("community_distribution", {})
            top_comms_ids = sorted(comm_dist.items(), key=lambda x: x[1], reverse=True)[:5] # Top 5 communities
            relevant_comm_ids = [c[0] for c in top_comms_ids]
            
            # Fetch summaries for these communities
            comm_summaries_text = []
            if relevant_comm_ids:
                with self.graph_store._driver.session() as session:
                    cypher = """
                    MATCH (n:LlamaNode) 
                    WHERE n.community_L0 IN $comm_ids
                    RETURN DISTINCT n.community_L0 as id, n.summary_L0 as summary
                    """
                    result = session.run(cypher, comm_ids=relevant_comm_ids)
                    for record in result:
                        comm_summaries_text.append(f"- [Community {record['id']}]: {record['summary']}")
            
            ref_block = f"""
            ### Source {idx + 1} ({ref['type']} - Dist: {1-ref['score']:.2f}): {cluster_data['name']}
            **Key Functional Communities in this domain:**
            {chr(10).join(comm_summaries_text)}
            """
            reference_context_str += ref_block

        prompt = f"""
        You are a creative System Architect.
        
        ## Context
        The user wants to design a system described as: ```{user_query}```.
        We have already identified the following BASE components (Anchors): ```{base_anchors_str}```.
        
        ## Task
        Generate **{num_outputs}** DISTINCT and DIVERGENT extension plans to enhance this system. 
        
        ## Reference Material (Knowledge Injection)
        We have retrieved knowledge from different domain clusters to inspire you.
        Please strictly follow the mapping below to generate your plans:
        
        {reference_context_str}
        
        ## Requirements by Plan
        
        **Plan 1 (Deep Optimization)**:
        - Use **Source 1** (Near-Field) as your inspiration.
        - Look for professional details or standard modules in this domain that are missing from the Base Anchors.
        - Refine the system to be more professional and complete.
        
        **Plan 2 (Cross-Domain Inspiration)** (If requested):
        - Use **Source 2** (Far-Field) as your inspiration.
        - Perform **Concept Blending**: specific methods from Source 2 (e.g., "Medical Monitoring") should be metaphorically transferred to the User's domain (e.g., "Education Monitoring").
        - Ask yourself: "How would an expert in Source 2 solve the problems in the User's query?"
        
        **Plan 3+** (If requested):
        - Use **Source 3+** similarly for distinct cross-domain features.

        ## General Constraint
        1. **Incremental Only**: 
           - Do NOT repeat the Base Anchors. Output ONLY the ADDITIONAL components (modules).
           - **CRITICAL**: Do NOT propose modules that are merely synonyms or simple rephrasings of Base Anchors (e.g., if "Tutor" exists, do not add "Instructor Agent"). New anchors must add **distinct new functionality**.
        2. **Specific Anchors**:
           - Each additional component must have a 'user_term', a 'description', a 'role', and a 'community_id' (from the reference source).
           - Recheck data flow logic.

        ### Role Taxonomy
        {TAXONOMY}\n\n

        ## Output Structure (JSON)
        {{
            "plans": [
                {{
                    "perspective": "use short sentence or phrases to describe the perspective (e.g., 'Biometric-Adaptive Interaction')",
                    "rationale": "Explain explicitly: 'Inspired by [Source Name], we introduced [Feature] to achieve [Goal]...'",
                    "additional_anchors": [
                        {{
                            "user_term": "Specific Module Name",
                            "description": "Functional description",
                            "role": "category: subcategory",
                            "community_id": "ID from Source text"
                        }}
                    ]
                }},
                ...
            ]
        }}
        """
        print(prompt)
        print(f"\n🧠 Generating {num_outputs} divergent plans...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-11-20",
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
            # Append functional description to provide context
            start_node_name = nodes[0]['name']
            chain_desc.append(f"[{start_node_name}]")
            
            for j, rel in enumerate(rels):
                target_node = nodes[j+1]['name']
                if nodes[j+1].get('function'):
                    target_node += f" ({nodes[j+1]['function']}...)"
                elif nodes[j+1].get('text'):
                    target_node += f" ({nodes[j+1]['text'][:300]}...)"
                
                # Extract edge label if available
                data_content = ""
                if j < len(rel_props) and rel_props[j] and 'edge_label' in rel_props[j]:
                    lbl = rel_props[j]['edge_label']
                    if lbl and lbl != "null":
                        data_content = f": {lbl}"

                if rel == "FLOWS_TO":
                    chain_desc.append(f" --(label: {data_content})--> [{target_node}]")
                elif rel == "HAS_ROLE":
                        chain_desc.append(f" ==(equivalent role)==> [{target_node}]")
                else:
                        chain_desc.append(f" --({rel}{data_content})--> [{target_node}]")

            hints_text += f"   * Connection '{path['source_query']} -> {path['target_query']}':\n"
            hints_text += f"     Reference Path: {''.join(chain_desc)}\n"

            real_intermediates = []
            for k in range(1, len(nodes)-1):
                if k < len(rels) and rels[k] == "FLOWS_TO":
                        real_intermediates.append(nodes[k]['name'])

        hints_text += "\n"
        return hints_text
    
    def integrate_arch(self, query, retrieved_info, structural_hints=[], added_nodes=[]):
        """
        Integrates all information to generate the final system architecture.

        Args:
            query (str): The user query.
            retrieved_info (dict): Retrieved specific module information.
            structural_hints (list): Structural paths found between components.
            added_nodes (list): New nodes introduced by the divergent planner.

        Returns:
            dict: The final integrated architecture plan.
        """

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
                
                # If this is an added node, explicitly preserve its topology 
                if anchor in added_keys:
                    # Try to get raw topology from metadata as a cleaner signal
                    raw_topo = n.get('structural_path', None)
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
        You are a professional designer and architect in the field of HCI. Design a complete pipeline based on the user's requirements and the retrieved information.
        ## User Requirements
        The user requirements are the principle you should follow the most. The nodes in the query must be included in the pipeline.
        User Query: ```{query}```
        \n\n
        ## Retrieved Information
        You have the following sources of information. Use them according to the rules.
        1. **Retrieved Modules**: 
            - These are modules extracted from the domain-specific literature that are similar to each conceptual step. Their descriptions include their functions, data processing methods and specific technical methods.
            - Refer to them to fill in the technical implementation of these specific nodes.
            - Each conceptual step may have retrieved multiple nodes with similar functions. You need to conduct appropriate screening or combination to be able to assemble a complete process.
        2. **Structural Paths**: 
            - These are retrieved data flow connections between conceptual steps derived from the user's requirements.
            - Use them to determine HOW the user's nodes should be connected.
            - If the paths suggest the need for an intermediate node (e.g., Sensor -> [Buffer] -> LLM), then such a node should be added.

        ## Data Sources
        Added Nodes by Divergent Planner: ```{added_nodes_text}```\n\n
        Retrieved Modules: ```{json.dumps(slim_retrieved_info, ensure_ascii=False, indent=2)}```\n\n
        Structural Paths: ```{hints_text}```\n\n
        
        ## TASK
        Conceive the complete data link and functional architecture, and ultimately synthesize a cohesive flowchart/architecture diagram. 
        ### Steps and Rules 
        1. **Define Modules**: Sort out user requirements and retrieved information and define the necessary modules in the entire architecture.
            - **Naming Rule**: Use **Generic Functional Names** (e.g., "Gaze Tracker", "Dialogue Manager") for the modules. **DO NOT** use specific paper titles, citation indices, or long implementation names (e.g. NOT "Gaze360: Physically Unconstrained...").
            - Ensure the added nodes from the divergent plans are integrated meaningfully.
            - If there are subnodes within the modules (based on the retrieved information and relevant to the current context), they should also be defined.
        2. **Connect the Flow**: Use the topology info to decide logical connections. 
            - All the connections should be implemented from the modules defined above. Do not include any other nodes, especially when defining the source_mod and the target_mod in ports.
            - **Graph Cleaning**: Merge parallel edges between the same two nodes. Do not create 'shadow' nodes that duplicate existing functions. Ensure the graph is a single connected component if possible.
            - If Module A usually follows Module B in Paper X, try to preserve that flow if it fits.

        3. **Check the architecture**:
            - REMOVE redundant nodes and connections (perform the same function even with different names).
            - REMOVE or RECHECK any nodes that are isolated or unused.
        
        ### Final Output Structure (in JSON format)
        {{
            "overall_description": "...",
            "module_details": [
                "<module name>": {{
                    "name": "the module name",
                    "function": "the functions of this module",
                    "technical_methods": "the technical methods used in this module",
                    "role": "choose from ["input", "output", "processing", "data"]",
                    "references": "the code of the referenced module (if any). Use 'None' if not available.",
                    "sub_nodes":[
                        {{
                            "name": "the sub node name (if any)",
                            "function": "the functions of this sub node (if any)",
                            "technical_methods": "the technical methods used in this sub node (if any)",
                        }}
                    ],
                    "ports": {{
                        "incoming (if any)": [
                            {{
                                "source_mod": "the identifier of the incoming module",
                                "source_node": "the identifier of the incoming node. If the source module has subnodes, use the subnode identifier that the incoming edge connects to. Else, use `None` means the data flow from the module.",
                                "connects_to_internal_node": "the identifier of the internal subnode that the incoming edge connects to. If the incoming edge connects to the module, use `None`.",
                                "edge_label": "the label of the incoming edge, i.e., the meaning of the data flow",
                            }}
                        ],
                        "outgoing (if any)": [
                            {{
                                "target_mod": "the identifier of the outgoing module",
                                "target_node": "the identifier of the outgoing node. If the target module has subnodes, use the subnode identifier that the outgoing edge connects to. Else, use `None` means the data flow to the module.",
                                "originates_from_internal_node": "the identifier of the internal subnode that the outgoing edge originates from. If the outgoing edge originates from the module, use `None`.",
                                "edge_label": "the label of the outgoing edge, i.e., the meaning of the data flow",
                            }}
                        ],
                        "internal_flow (if any that happens within the module, from one subnode to another)": [
                            {{
                                "from": "the identifier of the subnode",
                                "to": "the identifier of the subnode",
                                "edge_label": "the label of the internal edge, i.e., the meaning of the data flow",
                            }}
                        ]
                    }}
                }}
            ]
        }}
    """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-11-20",
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

    def refine_arch(self, original_query, current_arch, error_report):
        """
        Asks the LLM to fix the architecture based on the error report from ConflictSolver.
        """
        print(f"🔧 Integrator: Refining architecture based on errors...")
        prompt = f"""
        You are a System Architect. You previously generated a system design JSON, but it failed the integrity check.
        
        ## User Requirements
        {original_query}

        ## Current (Flawed) Design
        {current_arch}

        ## Error Report (Why it is invalid)
        {error_report}

        ## Task
        Fix the errors listed in the report. 
        1. If a node is missing (dangling edge), define it.
        2. If a node is isolated, connect it or remove it if truly unnecessary.
        3. Ensuring all 'source_mod' and 'target_mod' in ports refer to valid keys in 'module_details'.

        Return the FULLY CORRECTED JSON object.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Lower temperature for precision fixes
                response_format={"type": "json_object"},
            ).choices[0].message.content
            return response
        except Exception as e:
            print(f"Error checking integrity: {e}")
            return current_arch # Return original if fix fails

class DrawerAgent:
    def __init__(self, client):
        self.client = client
    
    def draw_flowchart(self, arch_plan):

        prompt = f"""
        You are a Mermaid code generator.
        ## Task
        Convert the given system architecture (JSON) into a standard Mermaid flowchart code.
        
        ## Input Architecture
        {arch_plan}

        ## Mapping Rules
        1. arch_plan["module_details"] -> nodes
            - Nodes are only represented by their ids and names, without any other information appearing in the diagram (including functions, technical methods, and references).
        2. arch_plan["module_details"]["sub_nodes"] -> subnodes (if any) 
            - Use subgraphs to represent modules if there are more than one subnodes and data flow between them in a single module. 
            - DO NOT use subgraphs if there is only one node and the name of the node is the same as the module name.
            - Use connection lines to represent the data flow sequence of subnodes within a module.
        3. arch_plan["module_details"]["ports"] -> edges
            - "target_mod" and "source_mod" represent the module id.
            - "target_node" and "source_node" represent the node id (within the target module and source module respectively).
            - "connects_to_internal_node" and "originates_from_internal_node" represent the subnode id (within the target module and source module respectively).
            - "edge_label" represents the label of the edge. 
        4. arch_plan["module_details"]["ports"]["internal_flow"] -> internal edges (if any)
        DO NOT add any extra nodes or edges!
        
        ## Drawing Rules
        1. Use 'graph TD' (top-down) layout.
        2. For EDGES:
            - Put descriptive text ON THE EDGE, not inside the node name.
            - Format: `NodeA -->|description| NodeB`
            - Do NOT change the node ID or Label to include the description.
        3. Avoid Errors:
            - Always enclose the labels with double quatation marks.
            - DO NOT include style attributes.
            - NO 'click' events or 'class' identification.
            - The **subgraph** has been correctly defined and an end marker "end" has been added.
        4. Output Format:
           - Return a JSON object with a single key "mermaid_code".
           - The value should be the pure Mermaid code string.
           - Example: {{ "mermaid_code": "graph TD\\n..." }}
    """
        try: 
            response = self.client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            raw_code = json.loads(response)["mermaid_code"]
            clean_code = self.clean_mermaid_code(raw_code)
            return clean_code
        except Exception as e:
            print(f"Error: {e}")
            return "graph TD\nError[Generation Failed]"
        
    def clean_mermaid_code(self,code):
        """
        清理Mermaid代码中的转义字符
        """
        # 1. 尝试提取代码块
        if '```mermaid' in code:
            pattern = r"```mermaid\s*(.*?)\s*```"
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1)
        elif '```' in code:
            pattern = r"```\s*(.*?)\s*```"
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1)

        # 2. 替换常见转义问题
        code = code.replace('\\n', '\n')
        code = code.replace('\\"', '"')
        
        # 3. 去除首尾空白
        code = code.strip()
        
        # 4. 去除可能残留的引号（如果整个代码被包裹在引号中）
        if code.startswith('"') and code.endswith('"'):
            code = code[1:-1]
            code = code.strip()

        return code
            
class ConflictSolver:
    def __init__(self):
        pass
    
    def detect_conflicts(self, arch_json_str):
        """
        Inspects the JSON for logical errors and returns a list of issues.
        Returns: (is_valid: bool, report: str)
        """
        try:
            if isinstance(arch_json_str, str):
                data = json.loads(arch_json_str)
            else:
                data = arch_json_str
        except json.JSONDecodeError:
            return False, "CRITICAL: Invalid JSON format."

        module_details = data.get("module_details", {})
        if not module_details:
            return False, "CRITICAL: 'module_details' field is missing or empty."

        issues = []
        valid_mod_ids = set(module_details.keys())
        connected_modules = set()

        # Check Edges validity
        for m_id, m_info in module_details.items():
            ports = m_info.get("ports", {})
            
            # Incoming
            for edge in ports.get("incoming", []):
                src = edge.get("source_mod")
                if src not in valid_mod_ids:
                    issues.append(f"- Invalid Connection: Module '{m_id}' expects input from '{src}', but '{src}' is not defined.")
                else:
                    connected_modules.add(m_id)
                    connected_modules.add(src)

            # Outgoing
            for edge in ports.get("outgoing", []):
                tgt = edge.get("target_mod")
                if tgt not in valid_mod_ids:
                    issues.append(f"- Invalid Connection: Module '{m_id}' sends output to '{tgt}', but '{tgt}' is not defined.")
                else:
                    connected_modules.add(m_id)
                    connected_modules.add(tgt)

        # Check for Isolated Nodes (Islands)
        # We consider a node isolated if it is not in the 'connected_modules' set
        # But we must be careful: some nodes might be purely internal or implicit start/end? 
        # Typically every node should have at least one connection unless there's only 1 node total.
        if len(valid_mod_ids) > 1:
            for m_id in valid_mod_ids:
                if m_id not in connected_modules:
                    issues.append(f"- Isolated Node: Module '{m_id}' has no incoming or outgoing connections to other modules.")

        if not issues:
            return True, "No conflicts found."
        
        return False, "\n".join(issues)

    def resolve(self, arch_json_str):
        """
        Parses the architecture JSON, validates topology, fixes broken edges,
        and removes isolated nodes to ensure a valid graph.
        """
        try:
            if isinstance(arch_json_str, str):
                data = json.loads(arch_json_str)
            else:
                data = arch_json_str
        except json.JSONDecodeError:
            print("❌ ConflictSolver: Invalid JSON string.")
            return arch_json_str

        module_details = data.get("module_details", {})
        if not module_details:
            return json.dumps(data)

        print("🕵️ ConflictSolver: Checking topology integrity...")
        
        # 1. Collect Valid IDs
        valid_mod_ids = set(module_details.keys())
        valid_sub_ids = {} # module_id -> set of subnode names
        
        for m_id, m_info in module_details.items():
            subs = set()
            if "sub_nodes" in m_info and isinstance(m_info["sub_nodes"], list):
                for sub in m_info["sub_nodes"]:
                    if "name" in sub:
                        subs.add(sub["name"])
            valid_sub_ids[m_id] = subs

        # 2. Fix Edges (Ports)
        connected_modules = set()
        
        for m_id, m_info in module_details.items():
            ports = m_info.get("ports", {})
            new_ports = {
                "incoming": [],
                "outgoing": [],
                "internal_flow": ports.get("internal_flow", [])
            }
            
            # Check Incoming
            if "incoming" in ports and isinstance(ports["incoming"], list):
                for edge in ports["incoming"]:
                    src_mod = edge.get("source_mod")
                    # Check if source module exists
                    if src_mod not in valid_mod_ids:
                        print(f"  Start Refactoring: Removed dead incoming link from '{src_mod}' to '{m_id}' (Source not found)")
                        continue
                    
                    # Connection is valid
                    new_ports["incoming"].append(edge)
                    connected_modules.add(m_id)
                    connected_modules.add(src_mod)

            # Check Outgoing
            if "outgoing" in ports and isinstance(ports["outgoing"], list):
                for edge in ports["outgoing"]:
                    tgt_mod = edge.get("target_mod")
                    # Check if target module exists
                    if tgt_mod not in valid_mod_ids:
                        print(f"  Start Refactoring: Removed dead outgoing link from '{m_id}' to '{tgt_mod}' (Target not found)")
                        continue
                        
                    # Connection is valid
                    new_ports["outgoing"].append(edge)
                    connected_modules.add(m_id)
                    connected_modules.add(tgt_mod)
            
            # Update ports
            m_info["ports"] = new_ports

        # 3. Identify and Remove Isolated Nodes
        # Re-calculate exact usage
        referenced_modules = set()
        for m_id, m_info in module_details.items():
            ports = m_info.get("ports", {})
            for e in ports.get("incoming", []):
                referenced_modules.add(e.get("source_mod"))
                referenced_modules.add(m_id)
            for e in ports.get("outgoing", []):
                referenced_modules.add(e.get("target_mod"))
                referenced_modules.add(m_id)
        
        final_modules = {}
        cleaned_count = 0
        
        for m_id, m_info in module_details.items():
            if m_id in referenced_modules:
                final_modules[m_id] = m_info
            else:
                print(f"  Start Refactoring: Isolated node '{m_id}' removed.")
                cleaned_count += 1
                
        data["module_details"] = final_modules
        
        if cleaned_count > 0:
            print(f"✅ ConflictSolver: Removed {cleaned_count} isolated modules.")
        else:
            print(f"✅ ConflictSolver: No isolated modules found.")

        return json.dumps(data, ensure_ascii=False)
    