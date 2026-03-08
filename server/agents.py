import os
import re
import httpx
import json
import sys
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from retriever_neo4j import (
    load_stores, 
    retrieve_and_get_content, 
    find_structural_path,
    run_with_retry,
    execute_read_with_session
)

http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(
    http_client=http_client,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
    )

TAXONOMY = """
1. **sensing_input**
    - bio_sensor: internal human physiological signals, e.g., EEG, EMG, ECG
    - behavior_sensor: external human behaviors/movements. e.g., IMU, eye tracker, motion capture
    - env_sensor: environment perception, e.g., RGB Camera, temperature sensor
    - direct_input: direct human-computer interaction input based on interfaces, e.g., Keyboard, Mouse, Touchscreen
    - system_context: equipment, system status and application log, etc.
    - data_import: offline documents, knowledge base, third-party API stream that serve as inputs
2. **processing_core**
    - signal_optimization: making the signal cleaner without changing the data type, e.g., filter, normalizer
    - feature_extraction: convert raw high-dimensional data (images, audio streams) into structured features or symbolic representations, e.g., gaze tracker, pose estimator, speech-to-text
    - state_assessment: qualitative or quantitative evaluation of the extracted features (judge how good/bad/intense something is), e.g., emotion recognition, fatigue detection
    - decision_logic: determine the next step based on the state assessment, e.g., recommender logic, dialog manager (even if it uses an LLM, if the output is a control flag or strategy, it belongs here)
    - content_generation: e.g., text generation (llm, chatbot), image generation (diffusion model)
3. **interface_output**
    - immersive_display: immersive visual output, e.g., VR headset, AR glasses, Unity Scene
    - gui_display: traditional visual display, e.g., Web UI, Mobile screen
    - haptic_feedback: e.g., Vibration motors, exoskeleton feedback
    - audio_output: e.g., TTS, Sound effects
4. **infrastructure**
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
            
            if near_score < 0.5:
                selected_references.append({
                    "type": "NEAR_FIELD_MISSING",
                    "cluster_id": None,
                    "score": near_score,
                    "name": "No Relevant Papers Found"
                })
                print(f"⚠️ Near-Field Context Score low ({near_score:.2f} < 0.5). Marked as missing.")
            else:
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
             existing_ids = set(r["cluster_id"] for r in selected_references if r["cluster_id"] is not None)
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
            if ref["type"] == "NEAR_FIELD_MISSING":
                ref_block = f"""
            ### Source {idx + 1} (Near-Field):
            Condition: Similarity score {ref['score']:.2f} < 0.5.
            Message: No papers highly relevant to this system were retrieved.
            """
                reference_context_str += ref_block
                continue
                
            c_id = ref["cluster_id"]
            if c_id is None: continue # Should not happen unless logic error

            cluster_data = self.topology_knowledge["scene_clusters"][c_id]
            
            # Get communities for this cluster
            comm_dist = cluster_data.get("community_distribution", {})
            sorted_comms = sorted(comm_dist.items(), key=lambda x: x[1], reverse=True)
            top_comms_ids = sorted_comms[:6]
                
            relevant_comm_ids = [c[0] for c in top_comms_ids]
            
            # Fetch summaries for these communities
            comm_summaries_text = []
            if relevant_comm_ids:
                def _fetch_comm_summaries(tx, comm_ids):
                    cypher = """
                    MATCH (n:LlamaNode) 
                    WHERE n.community_L0 IN $comm_ids
                    RETURN DISTINCT n.community_L0 as id, n.summary_L0 as summary
                    """
                    result = tx.run(cypher, comm_ids=comm_ids)
                    return list(result)

                # Use run_with_retry for Neo4j interaction
                try:
                    records = run_with_retry(
                        lambda: execute_read_with_session(self.graph_store._driver, _fetch_comm_summaries, relevant_comm_ids)
                    )
                    for record in records:
                        comm_summaries_text.append(f"- [Community {record['id']}]: {record['summary']}")
                except Exception as e:
                    print(f"⚠️ Failed to fetch community summaries for planning: {e}")
            
            # Extract Common Patterns with Claims
            patterns_text = [] 
            for p_key in ["common_patterns_3", "common_patterns_4"]:
                patterns = cluster_data.get(p_key, [])
                # Take top 3 sorted by count
                patterns.sort(key=lambda x: x.get("count", 0), reverse=True)
                for pat in patterns[:3]:
                    seq = pat.get("sequence", [])
                    claim = pat.get("claim", "")
                    if seq and claim:
                        seq_str = " -> ".join(seq)
                        patterns_text.append(f"- Pattern: [{seq_str}]\n  * Insight: {claim}")
            
            patterns_section = ""
            if patterns_text:
                patterns_section = "\n**Common Workflow Patterns & Insights:**\n" + "\n".join(patterns_text)

            ref_block = f"""
            ### Source {idx + 1} ({ref['type']} - Dist: {1-ref['score']:.2f}): {cluster_data['name']}
            **Key Functional Communities in this domain:**
            {chr(10).join(comm_summaries_text)}
            {patterns_section}
            """
            reference_context_str += ref_block

        prompt = f"""
        You are a creative System Architect.
        
        ## Context
        The user wants to design a system described as: ```{user_query}```.
        We have already identified the following BASE components (Anchors): ```{base_anchors_str}```.
        
        ## Task
        Generate **{num_outputs}** DISTINCT and DIVERGENT extension plans from different function implementation paths to enhance this system. 
        
        ## Reference Material (Knowledge Injection)
        We have retrieved knowledge from different domain clusters to inspire you.
        Please strictly follow the mapping below to generate your plans:
        
        {reference_context_str}
        
        ## Requirements by Plan
        
        **Plan 1 (Deep Optimization)**:
        - Use **Source 1** (Near-Field) as your inspiration.
        - If Source 1 states "No papers highly relevant", explicitly mention "No papers highly relevant to this system were retrieved" in your rationale, and rely on general HCI knowledge to professionalize the system.
        - Focus on discovering **innovative and feasible functional embeddings** and deep professional details missing from the Base Anchors. Focus on the most advanced or unique technical methods in the community summaries, rather than the most conventional ones.
        
        **Plan 2 (Cross-Domain Inspiration)** (If requested):
        - Use **Source 2** (Far-Field) as your inspiration.
        - Focus on **innovative cross-domain transfer**: Instead of just introducing conventional ideas, extract typical processing patterns, novel technologies or main implementation methods from Source 2 that can significantly improve the User's domain.
        - Perform **Concept Blending**: specific methods from Source 2 (e.g., "Medical Monitoring") should be metaphorically and innovatively transferred to the User's domain (e.g., "Education Monitoring").
        - Ask yourself: "How would an expert with new tech in Source 2 solve the problems in the User's query?"
        
        **Plan 3+** (If requested):
        - Use **Source 3+** similarly to propose a DISTINCT innovative functional embedding, differing in implementation methods or referenced technologies from previous plans.

        ## General Constraint
        1. When you add new functions, always consider its *feasibility* in the given context.
        2. **Base Anchors Preservation**:
           - **CRITICAL**: The existing Base Anchors represent the core of the system. Your divergent plans MUST acknowledge them, build upon them, and seamlessly integrate new functionalities with them across ALL plans.
        3. **Incremental Only**: 
           - Do NOT output the existing Base Anchors in your array. Output ONLY the KEY and ADDITIONAL innovative components (modules). IF Base Anchors do NOT exist, extract the key components from the user's query and add them.
           - **CRITICAL**: Do NOT propose modules that are merely synonyms or simple rephrasings of Base Anchors (e.g., if "Tutor" exists, do not add "Instructor Agent"). New anchors must add **distinct and innovative new functionality**.
        4. **Specific Anchors**:
           - Each additional component must have a 'user_term', a 'description', a 'role', and a 'community_id' (from the reference source).
           - Recheck data flow logic.

        ### Role Taxonomy
        {TAXONOMY}\n\n

        ## Output Structure (JSON)
        {{
            "plans": [
                {{
                    "perspective": "use phrases or one short sentence to describe the perspective, highlighting the key functions or biggest advantages and innovations of this design (e.g., 'Immersive interactive feedback'). Do not just copy the perspective title or vague terms like 'cross-domain'",
                    "rationale": "Explain explicitly: 'Inspired by [Source Name], we introduced [Feature] to achieve [Goal]...'",
                    "additional_anchors": [
                        {{
                            "user_term": "Specific Module Name",
                            "description": "Functional description",
                            "role": "category: subcategory",
                            "community_id": "ID from Source text, use NONE if no match"
                        }}
                    ]
                }},
                ...
            ]
        }}
        """
        # print(prompt)
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
        
        def fetch_single_anchor(anchor):
            user_term = anchor["user_term"]
            description = anchor["description"]
            role = anchor.get("role", None)
            community_id = anchor.get("community_id", None)
            
            query_text = f"{user_term}: {description}"
            
            retrieved_nodes = retrieve_and_get_content(
                query_text, 
                self.vector_store, 
                self.graph_store, 
                top_k,
                node_type="MODULE",
                role_filter=role,
                community_id=community_id,
                include_topology=include_topology
            )
            return user_term, retrieved_nodes

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(anchors_list) or 1)) as executor:
            future_to_anchor = {executor.submit(fetch_single_anchor, anchor): anchor for anchor in anchors_list}
            for future in concurrent.futures.as_completed(future_to_anchor):
                try:
                    user_term, retrieved_nodes = future.result()
                    if retrieved_nodes:
                        anchor_mapping[user_term] = retrieved_nodes
                        for n in retrieved_nodes:
                            print(f"Found: {n.node.metadata.get('name', '')} (Score: {n.score:.4f})")
                except Exception as exc:
                    print(f"Retrieval generated an exception: {exc}")

        return anchor_mapping
    
    def retrieve_structure_paths(self, user_relations):
        structural_hints = []
        
        def fetch_single_path(rel):
            src = rel["source"]
            tgt = rel["target"]
            return find_structural_path(src, tgt, rel_type=rel["type"], vector_store=self.vector_store, graph_store=self.graph_store)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(user_relations) or 1)) as executor:
            future_to_rel = {executor.submit(fetch_single_path, rel): rel for rel in user_relations}
            for future in concurrent.futures.as_completed(future_to_rel):
                try:
                    paths = future.result()
                    structural_hints.extend(paths)
                except Exception as exc:
                    print(f"Structural path retrieval generated an exception: {exc}")

        return structural_hints
    
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
    
    def integrate_arch(self, query, retrieved_info, structural_hints=[], divergent_plan=None, base_anchors=[]):
        """
        Integrates all information to generate the final system architecture.

        Args:
            query (str): The user query.
            retrieved_info (dict): Retrieved specific module information.
            structural_hints (list): Structural paths found between components.
            divergent_plan (dict): The structured plan from divergent_planning (optional).
            base_anchors (list): The list of user-defined base anchors.

        Returns:
            dict: The final integrated architecture plan.
        """

        hints_text = self.format_hints(structural_hints)

        # Process Divergent Plan Info
        extension_context = ""
        added_keys = set()
        
        if divergent_plan:
            perspective = divergent_plan.get("perspective", "Standard")
            rationale = divergent_plan.get("rationale", "")
            added_nodes = divergent_plan.get("additional_anchors", [])
            
            if added_nodes:
                added_keys = {n['user_term'] for n in added_nodes}
                
                # Format extension modules significantly for the prompt
                modules_list_str = "\n".join([f"- {n['user_term']}: {n['description']}" for n in added_nodes])
                
                extension_context = f"""
                ## Divergent Extensions (New Sub-systems to Integrate)
                We have identified a valid extension plan to enhance the system. You must integrate these new modules into the Main Pipeline responsibly.
                
                **Plan Perspective**: "{perspective}"
                **Rationale**: {rationale}
                
                **Extension Modules**:
                {modules_list_str}
                
                **Integration Strategy**:
                1. **Sub-System Mounting**: TDepending on their nature, weave these extension modules deeply into the architecture.
                - For the modules that align with the main idea of the plan, design around this new model and restructure the related basic nodes to make them the core functions.
                - Ensure a rich data flow and interconnections. If they inherently need to interact with multiple components (such as memory tracking, sensory input, and user interface), do not limit them to a single "hook point".
                
                2. **Conflict Handling**:
                - If an Extension Module conflicts with a User Module (e.g., "Standard Camera" vs. "Specialized Eye-Tracker"), **Prioritize the User's explicit choice**, but consider using the Extension Module as an *alternative input* or *enhancement*.
                """
        else:
             # Fallback if old call signature is used or no plan
             extension_context = "No specific divergent extensions provided."

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
        You are a professional designer and architect in the field of HCI. Design a *complete and detailed* pipeline based on the user's requirements and the retrieved information.
        
        ## User Requirements
        User Query: ```{query}```
        
        ## Base Anchors (MANDATORY)
        The following Base Anchors were explicitly requested by the user. **You MUST include ALL of these modules in your final pipeline architecture.** You can modify its name, details, and implementation method, but please retain the main functions as requested by the users. Do not drop them under any circumstances.
        Base Anchors: ```{json.dumps(base_anchors, ensure_ascii=False) if base_anchors else "None provided"}```
        \n\n
        ## Retrieved Information
        You have the following sources of information. Use them according to the rules.
        1. **Retrieved Modules**: 
            - These are modules extracted from the domain-specific literature that are similar to each conceptual step. Their descriptions include their functions, data processing methods and specific technical methods.
            - Refer to them to fill in the technical implementation of these specific nodes.
            - Each conceptual step may have retrieved multiple nodes with similar functions. You need to conduct appropriate screening or combination to be able to assemble a complete process.
        2. **Structural Paths**: 
            - These are retrieved data flow connections between conceptual steps derived from the user's requirements.
            - **CRITICAL**: Use them as the ground truth to determine HOW the user's nodes should be connected. Do not invent arbitrary connections if a structural path already defines them.
            - If the paths suggest the need for an intermediate node (e.g., Sensor -> [Buffer] -> LLM), then such a node MUST be added to connect the Base Anchors logically.

        ## Data Sources
        {extension_context}

        Retrieved Modules: ```{json.dumps(slim_retrieved_info, ensure_ascii=False, indent=2)}```\n\n
        Structural Paths: ```{hints_text}```\n\n
        
        ## TASK
        Conceive the complete data link and functional architecture, and ultimately synthesize a cohesive flowchart/architecture diagram. 
        ### Steps and Rules 
        1. **Define Modules**: Sort out user requirements and retrieved information and define the necessary modules in the entire architecture.
            - **Naming Rule**: Use **Generic Functional Names** (e.g., "Gaze Tracker", "Dialogue Manager") for the modules. **DO NOT** use specific paper titles, citation indices, or long implementation names (e.g. NOT "Gaze360: Physically Unconstrained...").
            - Ensure the added nodes from the divergent plans are integrated meaningfully.
            - If there are subnodes within the modules (based on the retrieved information and relevant to the current context), they should also be defined. Explicitly map out the intermediate processes and data flow within the module using subnodes so its internal logic is transparent.
            - **Detail Requirement**: Provide highly detailed descriptions for both `function` and `technical_methods` from the retrieved information BUT DO NOT FABRICATE. You must explicitly articulate the module's specific role, core responsibilities, and crucial importance within the overall system. Explain exactly HOW the technical methods work, process data, and why they were chosen, avoiding brief or vague summaries.
        2. **Connect the Flow**: Use the topology info to decide logical connections. 
            - Ensure new conceptual modules have comprehensive connections with the base logic to fully realize their functional value. DO NOT let them become dead ends unless explicitly defined as output endpoints.
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
                    "brief_description": "A one-sentence concise explanation of the module's core functional role, meant directly for rendering on the visual node. Emphasize how it serves the current scenario without paper-specific jargon.",
                    "function": "Detailed explanation of the module's function, its core responsibilities, and its crucial role and importance within the overall system architecture.",
                    "technical_methods": "Detailed description of the specific algorithms, models, or technical methods used, including HOW they process data and why they are selected. You MUST abstract away the original paper context. Translate the core technical concepts into generic terms and tightly fuse them with the user's specific scenario. The generated explanations should feel like a native, tailored solution to the user's problem.",
                    "role": "choose from ["input", "output", "processing", "data"]",
                    "references": "Exact names of the sourced modules from Retrieved Information, comma-separated. Use 'None' if not available.",
                    "sub_nodes": [
                        {{
                            "name": "name of the subnode",
                            "function": "Detailed description of the subnode's specific function, purpose, and contribution to its parent module.",
                            "technical_methods": "Detailed description of the specific technical implementation and methods used by this subnode."
                        }}
                    ],
                    "ports": {{
                        "incoming": [
                            {{
                                "source_mod": "id of the source module",
                                "source_node": "id of the source subnode (if applicable, else null)",
                                "connects_to_internal_node": "id of the internal subnode (if applicable, else null)",
                                "edge_label": "description of data/control flow"
                            }}
                        ],
                        "outgoing": [
                             {{
                                "target_mod": "id of the target module",
                                "target_node": "id of the target subnode (if applicable, else null)",
                                "originates_from_internal_node": "id of the internal subnode (if applicable, else null)",
                                "edge_label": "description of data/control flow"
                            }}
                        ],
                        "internal_flow": [
                           {{
                                "source_node": "id of the internal source subnode",
                                "target_node": "id of the internal target subnode",
                                "edge_label": "description of data/control flow"
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
            return current_arch

    def collect_references(self, arch_plan, retrieved_info):
        """
        Organize used references and unused 'explorable' references from retrieved info.
        Args:
            arch_plan (dict): The final architecture plan (JSON).
            retrieved_info (dict): The raw retrieval dictionary {anchor: [nodes]}.
        Returns: 
            dict: { "used": { "Module": [evidence] }, "explorable": [evidence] }
        """
        if not arch_plan or "module_details" not in arch_plan:
             return {"used": {}, "explorable": []}

        # 1. Extract Used Reference Names from Plan
        # Map: RefName -> ModuleUserTerm
        used_ref_map = {} 
        
        module_details = arch_plan["module_details"]
        vals = module_details.values() if isinstance(module_details, dict) else module_details
        
        for mod in vals:
            m_name = mod.get("name", "Unknown")
            ref_str = mod.get("references", "")
            if ref_str and ref_str.lower() != "none":
                # Split by comma
                refs = [r.strip() for r in ref_str.split(",")]
                for r in refs:
                    if r: used_ref_map[r.lower()] = m_name

        # 2. Iterate Retrieved Info
        used_output = {}
        explorable_output = []
        
        seen_ids = set() 

        for anchor, nodes in retrieved_info.items():
            for node_data in nodes:
                n_id = node_data.get('id', 'unknown')
                if n_id in seen_ids: continue
                seen_ids.add(n_id)
                
                n_name = node_data.get('metadata', {}).get('name', '')
                if not n_name and 'node' in node_data:
                     n_name = node_data['node'].metadata.get('name', '')
                     
                
                n_text = node_data.get('text', '')
                if not n_text and 'node' in node_data:
                    n_text = node_data['node'].text
                
                n_source = node_data.get('metadata', {}).get('source', '')
                if not n_source:
                    n_source = node_data.get('metadata', {}).get('topology', {}).get('source_file', '')
                
                if not n_name: n_name = "Unknown Source"

                # Check if this node is in used_ref_map
                matched_module = None
                for ref_key, mod_term in used_ref_map.items():
                    if ref_key in n_name.lower() or n_name.lower() in ref_key:
                        matched_module = mod_term
                        break
                
                entry = {
                    "name": n_name,
                    "text": n_text,
                    "source": n_source,
                    "source_anchor": anchor
                }
                
                if matched_module:
                    if matched_module not in used_output:
                        used_output[matched_module] = []
                    used_output[matched_module].append(entry)
                else:
                    explorable_output.append(entry)
                    
        return {"used": used_output, "explorable": explorable_output}

class EditorAgent:
    def __init__(self, client):
        self.client = client

    def analyze_instruction(self, user_instruction, current_plan_dict):
        """
        Analyzes the user's instruction to determine if external knowledge retrieval is needed.
        """
        print(f"🧠 Editor: Analyzing if instruction needs external knowledge...")
        prompt = f"""
        Analyze the user's instruction to modify the following system architecture.
        
        ## Current Plan
        {json.dumps(current_plan_dict, indent=2)}
        
        ## User Instruction
        {user_instruction}
        
        ## Task
        Determine if the user's instruction requires external domain knowledge (e.g., adding new interactive features, clarifying complex processes, integrating new technologies).
        If the instruction is just a simple renaming, deletion, or basic rewiring, retrieval is NOT needed.
        If the instruction requires specific technical implementation details or structural inspiration, extract 1-3 highly descriptive keyword phrases to search for in a research database.
        
        ## Output Format (JSON only)
        {{
            "needs_retrieval": true/false,
            "search_queries": ["query1", "query2"] // Empty if needs_retrieval is false
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            ).choices[0].message.content
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing instruction: {e}")
            return {"needs_retrieval": False, "search_queries": []}

    def revise_arch(self, current_plan, user_instruction, initial_query, additional_context=""):
        """
        Revises the current architecture based on user instructions.
        
        Args:
            current_plan (str or dict): The current architecture plan (JSON).
            user_instruction (str): User's modification or optimization instruction.
            initial_query (str): The original user requirement.
            
        Returns:
            dict: The revised architecture plan.
        """
        # Normalize current_plan to dict if it's a string
        if isinstance(current_plan, str):
            try:
                current_plan_dict = json.loads(current_plan)
            except json.JSONDecodeError:
                print("Error: current_plan is not a valid JSON string.")
                return current_plan
        else:
            current_plan_dict = current_plan

        prompt = f"""
        You are an expert system architect specializing in HCI systems. Your task is to modify the current system architecture based on the user's specific instruction. 
        
        ## Context
        - **Original User Query**: {initial_query}
        - **Current Architecture**: 
          {json.dumps(current_plan_dict, indent=2)}
        - **User Instruction**: {user_instruction}
        
        ## Additional Domain Knowledge (If provided)
        {additional_context}
        
        ## Task
        Apply the user's instruction to make profound and structurally sound changes to the architecture. 
        If "Additional Domain Knowledge" is provided, you MUST use it to deeply restructure the system flow and provide specific technical details (HOW it works), rather than just attaching a superficial empty node.
        
        ## Guidelines
        1. **Locality**: Make only the changes that are directly related to the user's instruction. Do NOT alter parts of the architecture that are unrelated to the request.
        2. **Preservation**: Maintain all existing modules and connections that are not specifically targeted by the user's instruction.
        3. **Consistency**: Ensure the modified architecture remains consistent and coherent. Preserve the overall system flow and logic.
        4. **Accuracy**: When adding or modifying modules, maintain correct naming, functionality, and interconnections.
        5. **Integrity**: Ensure all 'source_mod' and 'target_mod' in ports refer to valid keys in 'module_details'.
        
        ## Example Instructions
        - "Add a database to store user profiles" → Add a new module named "Database" with appropriate function, and connect it where relevant (e.g., to a "User Manager").
        - "Make the system more secure" → Add security-related modules like "Authentication Module" or enhance existing modules with security functions.
        - "Remove the notification system" → Delete modules related to notifications and update connections accordingly.
        
        ## Output Format
        Return the full updated architecture in the same JSON structure as the input, including all unchanged parts:
        {{
            "overall_description": "...",
            "module_details": [
                {{
                    "name": "...",
                    "brief_description": "A one-short-sentence concise explanation of the module's core functional role.",
                    "function": "Detailed explanation of the module's function, its core responsibilities, and its crucial role and importance within the overall system architecture.",
                    "technical_methods": "Detailed description of the specific algorithms, models, or technical methods used, including HOW they process data and why they are selected.",
                    "role": "choose from [\"input\", \"output\", \"processing\", \"data\"]",
                    "references": "...",
                    "sub_nodes": [...],
                    "ports": {{
                        "incoming": [...],
                        "outgoing": [...],
                        "internal_flow": [...]
                    }}
                }},
                ...
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            
            # print(response)
            return response
        except Exception as e:
            print(f"Error in revising architecture: {e}")
            return current_plan

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

        # === Normalization: Handle List vs Dict ===
        if isinstance(module_details, list):
            # Convert list to dict using 'name' as key
            normalized_details = {}
            for item in module_details:
                name = item.get("name", "Unknown")
                normalized_details[name] = item
            module_details = normalized_details

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
        if len(valid_mod_ids) > 1:
            for m_id in valid_mod_ids:
                if m_id not in connected_modules:
                    issues.append(f"- Isolated Node: Module '{m_id}' has no incoming or outgoing connections to other modules.")
        
        # Check for Isolated Sub-nodes
        for m_id, m_info in module_details.items():
            sub_nodes = m_info.get("sub_nodes", [])
            if not sub_nodes: continue
            
            ports = m_info.get("ports", {})
            referenced_subnodes = set()
            
            # Check incoming (connects_to_internal_node)
            for edge in ports.get("incoming", []):
                internal = edge.get("connects_to_internal_node")
                if internal: referenced_subnodes.add(internal)
                
            # Check outgoing (originates_from_internal_node)
            for edge in ports.get("outgoing", []):
                internal = edge.get("originates_from_internal_node")
                if internal: referenced_subnodes.add(internal)
                
            # Check internal_flow
            for flow in ports.get("internal_flow", []):
                referenced_subnodes.add(flow.get("source_node"))
                referenced_subnodes.add(flow.get("target_node"))
                
            for sub in sub_nodes:
                sub_name = sub.get("name")
                if sub_name and sub_name not in referenced_subnodes:
                     issues.append(f"- Isolated Sub-node: Sub-node '{sub_name}' in Module '{m_id}' is not connected to any internal or external port.")

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

        # Normalization: Handle List vs Dict
        if isinstance(module_details, list):
            print("🕵️ ConflictSolver: Normalized 'module_details' from List to Dict.")
            normalized_details = {}
            for item in module_details:
                name = item.get("name", "Unknown")
                normalized_details[name] = item
            module_details = normalized_details
            data["module_details"] = module_details

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
    

if __name__ == "__main__":
    vector_store, graph_store = load_stores()
    planner = PlannerAgent(client, graph_store)
    user_query = "Design an LLM-based code education system for children"
    base_anchors = []
    divergent_plans = planner.divergent_planning(
            user_query=user_query,
            base_anchors=base_anchors,
            num_outputs=1
    )
    print(divergent_plans)