import os
import re
import httpx
import json
import sys
import os
import itertools
import threading
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import wrappers, traceable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import hashlib

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load .env defensively — main.py already loads it, but agents.py is also
# imported by tests / standalone scripts. load_dotenv is idempotent.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# from config import *
from retriever_neo4j import (
    get_vector_store,
    get_graph_store,
    retrieve_and_get_content,
    find_structural_path,
    run_with_retry,
    execute_read_with_session,
    embed_via_client_with_fallback,
)

# === Round-robin client pool ===
# One OpenAI client per key; concurrent requests rotate through the pool
# so no single key absorbs all traffic.
# To add keys: set OPENAI_API_KEYS in .env as a comma-separated list, OR
# add a single OPENAI_API_KEY. Multi-key takes precedence.
_http_client = httpx.Client(proxy=None, trust_env=False)
_base_url = os.environ.get("OPENAI_BASE_URL")

_multi = os.environ.get("OPENAI_API_KEYS", "").strip()
if _multi:
    _keys = [k.strip() for k in _multi.split(",") if k.strip()]
else:
    _single = (os.environ.get("OPENAI_API_KEY") or "").strip()
    _keys = [_single] if _single else []

if not _keys:
    raise RuntimeError(
        "No OpenAI API key configured. Set OPENAI_API_KEY (or comma-separated "
        "OPENAI_API_KEYS) in .env at the project root."
    )

def _force_non_thinking(client):
    """Default DeepSeek-V4 calls to non-thinking mode (one place, all call sites).

    Thinking mode is ON by default on the official API and (a) roughly triples
    latency and (b) rejects temperature/top_p/penalty params — which this codebase
    uses everywhere. setdefault means a caller can still opt a specific call back
    into thinking by passing extra_body={"thinking": {"type": "enabled"}}.
    Only touches deepseek-v4* models, so legacy aliases are unaffected.
    """
    _orig = client.chat.completions.create

    def _patched(*args, **kwargs):
        if "deepseek-v4" in (kwargs.get("model") or ""):
            eb = dict(kwargs.get("extra_body") or {})
            eb.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = eb
        return _orig(*args, **kwargs)

    client.chat.completions.create = _patched
    return client


_client_pool = [
    _force_non_thinking(
        wrappers.wrap_openai(OpenAI(http_client=_http_client, api_key=k, base_url=_base_url))
    )
    for k in _keys
]
_pool_iter = itertools.cycle(range(len(_client_pool)))
_pool_lock = threading.Lock()
CHAT_MODEL = os.environ.get("CHAT_MODEL")

def get_client() -> OpenAI:
    """Return the next client in round-robin order (thread-safe)."""
    with _pool_lock:
        idx = next(_pool_iter)
    return _client_pool[idx]


print(f"✅ OpenAI client pool: {len(_client_pool)} key(s) at {_base_url}")


# Global cache for cluster embeddings to avoid recomputation
CLUSTER_EMBEDDINGS_CACHE = {}

def get_cluster_embeddings_cache_key(cluster_data):
    """Generate a cache key based on cluster data content."""
    cluster_text = json.dumps(cluster_data, sort_keys=True)
    return hashlib.md5(cluster_text.encode()).hexdigest()

def load_cached_embeddings():
    """Load cached embeddings from file if available."""
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_cluster", "embeddings_cache.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_embeddings_cache(cache_data):
    """Save embeddings cache to file."""
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_cluster", "embeddings_cache.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save embeddings cache: {e}")

# Load cache on module import
CLUSTER_EMBEDDINGS_CACHE = load_cached_embeddings()

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
        
        # Generate cache key for current topology knowledge
        topology_hash = get_cluster_embeddings_cache_key(self.topology_knowledge)
        
        # Check if we have cached embeddings for this topology
        if topology_hash in CLUSTER_EMBEDDINGS_CACHE:
            print(f"✅ Using cached embeddings for topology {topology_hash[:8]}...")
            self.cluster_embeddings = CLUSTER_EMBEDDINGS_CACHE[topology_hash]
            return
            
        print(f"🔄 Computing new embeddings for topology {topology_hash[:8]}...")
        
        # Compute new embeddings
        new_embeddings = {}
        for label, data in clusters.items():
            text = data.get("name", "")
            if text:
                emb = embed_via_client_with_fallback(self.client, text)
                if emb is None:
                    print(f"Error embedding cluster {label}: both primary and fallback failed", flush=True)
                    continue
                new_embeddings[label] = emb
        
        # Update cache
        self.cluster_embeddings = new_embeddings
        CLUSTER_EMBEDDINGS_CACHE[topology_hash] = new_embeddings
        save_embeddings_cache(CLUSTER_EMBEDDINGS_CACHE)
        print(f"💾 Saved embeddings to cache for topology {topology_hash[:8]}")

    def match_scenario(self, query):
        if not self.cluster_embeddings: return None, None

        query_emb_raw = embed_via_client_with_fallback(self.client, query)
        if query_emb_raw is None:
            return None, None
        query_emb = np.array(query_emb_raw).reshape(1, -1)
            
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
    @traceable(name="planner.planning", run_type="chain")
    def divergent_planning(self, user_query, base_anchors, num_outputs=2):
        """
        Generates divergent extension plans based on the user query and already extracted base anchors.
        Uses Knowledge-Distance Perspective Injection to find both near-field and far-field inspirations.
        """
        
        # Format base anchors for the prompt
        base_anchors_str = json.dumps([a['user_term'] for a in base_anchors], ensure_ascii=False)
        
        # 1. Calculate similarities with ALL clusters
        if not self.cluster_embeddings:
            print("No cluster embeddings available for divergent planning.", flush=True)
            return []

        query_emb_raw = embed_via_client_with_fallback(self.client, user_query)
        if query_emb_raw is None:
            print("❌ Divergent planning aborted: query embedding unavailable (primary + fallback both failed).", flush=True)
            return []
        query_emb = np.array(query_emb_raw).reshape(1, -1)

        scored_clusters = []
        all_labels = list(self.cluster_embeddings.keys())
        all_embs = np.array([self.cluster_embeddings[l] for l in all_labels])
        scores = cosine_similarity(query_emb, all_embs)[0]
        scored_clusters = sorted(zip(all_labels, scores.tolist()), key=lambda x: x[1], reverse=True)
        
        # 2. Select Reference Clusters (Near vs Far)
        # Strategy:
        # - Plan 0 (if num_outputs > 0) -> Near Field (Top 1) -> "Deep Dive"
        # - Plan 1..N -> Far Field (Random from score 0.3 - 0.6) -> "Cross-domain Transfer"
        
        # Enforce max outputs to 3 as requested
        num_outputs = min(max(1, num_outputs), 3)

        selected_references = []

        def _find_neighbor_cluster_ids(primary_id, candidate_pool, exclude_ids, max_neighbors=2, sim_threshold=0.6):
            """
            Among candidate_pool (list of (cluster_id, query_score)), return up to max_neighbors
            cluster IDs whose embedding similarity to primary_id is >= sim_threshold,
            excluding any IDs in exclude_ids. Candidates are already query-filtered (0.3-0.6),
            so neighbors are both domain-adjacent AND query-relevant.
            """
            if primary_id not in self.cluster_embeddings:
                return []
            primary_emb = np.array(self.cluster_embeddings[primary_id]).reshape(1, -1)
            neighbors = []
            for other_id, _ in candidate_pool:
                if other_id == primary_id or other_id in exclude_ids:
                    continue
                if other_id not in self.cluster_embeddings:
                    continue
                sim = cosine_similarity(primary_emb, np.array(self.cluster_embeddings[other_id]).reshape(1, -1))[0][0]
                if sim >= sim_threshold:
                    neighbors.append((other_id, sim))
            neighbors.sort(key=lambda x: x[1], reverse=True)
            return [n[0] for n in neighbors[:max_neighbors]]

        # A. Near Field (Top 1) — logic unchanged
        if scored_clusters:
            near_cluster_id = scored_clusters[0][0]
            near_score = scored_clusters[0][1]

            if near_score < 0.5:
                selected_references.append({
                    "type": "NEAR_FIELD_MISSING",
                    "cluster_id": None,
                    "cluster_ids": [],
                    "score": near_score,
                    "name": "No Relevant Papers Found"
                })
                print(f"⚠️ Near-Field Context Score low ({near_score:.2f} < 0.5). Marked as missing.")
            else:
                selected_references.append({
                    "type": "NEAR_FIELD",
                    "cluster_id": near_cluster_id,
                    "cluster_ids": [near_cluster_id],
                    "score": near_score,
                    "name": self.topology_knowledge["scene_clusters"][near_cluster_id]["name"]
                })
                print(f"🎯 Near-Field Context: {self.topology_knowledge['scene_clusters'][near_cluster_id]['name'][:50]}... (Score: {near_score:.2f})")

        # B. Far Field (Random Sampling)
        # Each far-field perspective samples an anchor cluster, then expands with 1-2 same-domain
        # neighbors so the LLM receives a coherent cluster of related techniques.
        near_id_set = {scored_clusters[0][0]} if scored_clusters else set()
        far_candidates = [c for c in scored_clusters if 0.2 <= c[1] <= 0.4 and c[0] not in near_id_set]

        needed_far = max(0, num_outputs - 1)
        if far_candidates and needed_far > 0:
            import random
            k = min(len(far_candidates), needed_far)
            chosen_far = random.sample(far_candidates, k)
            used_far_ids = set()

            for c_id, c_score in chosen_far:
                used_far_ids.add(c_id)
                neighbor_ids = _find_neighbor_cluster_ids(
                    c_id,
                    candidate_pool=far_candidates,
                    exclude_ids=near_id_set | used_far_ids,
                    max_neighbors=2,
                    sim_threshold=0.6
                )
                used_far_ids.update(neighbor_ids)
                all_cluster_ids = [c_id] + neighbor_ids
                selected_references.append({
                    "type": "FAR_FIELD",
                    "cluster_id": c_id,
                    "cluster_ids": all_cluster_ids,
                    "score": c_score,
                    "name": self.topology_knowledge["scene_clusters"][c_id]["name"]
                })
                print(f"🚀 Far-Field Context: {self.topology_knowledge['scene_clusters'][c_id]['name'][:50]}... "
                      f"(Score: {c_score:.2f}, +{len(neighbor_ids)} neighbors: {neighbor_ids})")

        # Fill remaining with random near/mid if we ran out of far candidates
        while len(selected_references) < num_outputs:
            existing_ids = {
                c_id for r in selected_references
                for c_id in r.get("cluster_ids", [])
                if c_id is not None
            }
            residue = [c for c in scored_clusters if c[0] not in existing_ids]
            if not residue:
                break
            c_id, c_score = residue[0]
            selected_references.append({
                "type": "SUPPLEMENTAL",
                "cluster_id": c_id,
                "cluster_ids": [c_id],
                "score": c_score,
                "name": self.topology_knowledge["scene_clusters"][c_id]["name"]
            })
        
        # 3. Retrieve Capabilities for specific references
        # We want to give the LLM specific "Reference Packs"
        
        reference_context_str = ""

        # Pre-fetch community summaries for all selected references in parallel
        def _fetch_summaries_for_ref(ref):
            if ref["type"] == "NEAR_FIELD_MISSING":
                return ref.get("cluster_id"), [], []
            cluster_ids = ref.get("cluster_ids", [ref.get("cluster_id")])
            cluster_ids = [c for c in cluster_ids if c is not None]
            if not cluster_ids:
                return None, [], []

            primary_id = cluster_ids[0]

            # Merge community_distribution across all clusters in this reference pack.
            # Weight each cluster's distribution equally so dominant communities from any
            # cluster surface; use a dict to accumulate scores.
            merged_comm_scores: dict = {}
            for c_id in cluster_ids:
                cdata = self.topology_knowledge["scene_clusters"].get(c_id, {})
                for comm, score in cdata.get("community_distribution", {}).items():
                    merged_comm_scores[comm] = merged_comm_scores.get(comm, 0) + score

            if not merged_comm_scores:
                return primary_id, [], []

            sorted_comms = sorted(merged_comm_scores.items(), key=lambda x: x[1], reverse=True)

            # Top-3 dominant communities (across all merged clusters)
            top_comm_ids = [c[0] for c in sorted_comms[:3]]

            # 2 random tail communities (rank 6–20 of merged ranking)
            import random
            tail_candidates = [c[0] for c in sorted_comms[5:20]]
            tail_comm_ids = random.sample(tail_candidates, min(2, len(tail_candidates)))

            relevant_comm_ids = top_comm_ids + tail_comm_ids
            tail_set = set(tail_comm_ids)

            def _fetch_comm_summaries(tx, comm_ids):
                cypher = """
                MATCH (n:LlamaNode)
                WHERE n.community_L0 IN $comm_ids
                RETURN DISTINCT n.community_L0 as id, n.summary_L0 as summary
                """
                result = tx.run(cypher, comm_ids=comm_ids)
                return list(result)
            try:
                records = run_with_retry(
                    lambda: execute_read_with_session(self.graph_store._driver, _fetch_comm_summaries, relevant_comm_ids)
                )
                lines = []
                for r in records:
                    tag = " *(long-tail — distinctive niche)*" if r['id'] in tail_set else ""
                    lines.append(f"- [Community {r['id']}]{tag}: {r['summary']}")
                return primary_id, lines, relevant_comm_ids
            except Exception as e:
                print(f"⚠️ Failed to fetch community summaries for planning: {e}")
                return primary_id, [], []

        refs_needing_fetch = [r for r in selected_references if r["type"] != "NEAR_FIELD_MISSING" and r.get("cluster_ids")]
        comm_summaries_cache = {}   # c_id -> lines
        comm_ids_cache = {}         # c_id -> list of community ids shown to LLM
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(refs_needing_fetch) or 1) as _ex:
            for c_id, lines, comm_ids in _ex.map(_fetch_summaries_for_ref, refs_needing_fetch):
                if c_id is not None:
                    comm_summaries_cache[c_id] = lines
                    comm_ids_cache[c_id] = comm_ids

        # Build cross-plan shared community set for dedup hint
        all_comm_id_lists = list(comm_ids_cache.values())
        if len(all_comm_id_lists) >= 2:
            from collections import Counter
            flat = [cid for lst in all_comm_id_lists for cid in lst]
            shared_communities = {cid for cid, cnt in Counter(flat).items() if cnt > 1}
        else:
            shared_communities = set()

        for idx, ref in enumerate(selected_references):
            if ref["type"] == "NEAR_FIELD_MISSING":
                ref_block = f"""
            ### Source {idx + 1} (Near-Field):
            Condition: Similarity score {ref['score']:.2f} < 0.5.
            Message: No papers highly relevant to this system were retrieved.
            """
                reference_context_str += ref_block
                continue

            c_id = ref.get("cluster_id") or (ref.get("cluster_ids") or [None])[0]
            if c_id is None: continue

            cluster_data = self.topology_knowledge["scene_clusters"][c_id]
            comm_summaries_text = comm_summaries_cache.get(c_id, [])

            # Extract Common Patterns with Claims — from all clusters in this reference pack
            patterns_text = []
            for ref_c_id in ref.get("cluster_ids", [c_id]):
                ref_cluster_data = self.topology_knowledge["scene_clusters"].get(ref_c_id, {})
                for p_key in ["common_patterns_3", "common_patterns_4"]:
                    patterns = ref_cluster_data.get(p_key, [])
                    patterns.sort(key=lambda x: x.get("count", 0), reverse=True)
                    for pat in patterns[:2]:
                        seq = pat.get("sequence", [])
                        claim = pat.get("claim", "")
                        if seq and claim:
                            seq_str = " -> ".join(seq)
                            patterns_text.append(f"- Pattern: [{seq_str}]\n  * Insight: {claim}")

            patterns_section = ""
            if patterns_text:
                patterns_section = "\n**Common Workflow Patterns & Insights:**\n" + "\n".join(patterns_text)

            extra_clusters = ref.get("cluster_ids", [c_id])
            cluster_note = f" (+{len(extra_clusters)-1} related clusters)" if len(extra_clusters) > 1 else ""
            ref_block = f"""
            ### Source {idx + 1} ({ref['type']} - Dist: {1-ref['score']:.2f}): {cluster_data['name']}{cluster_note}
            **Key Functional Communities in this domain:**
            {chr(10).join(comm_summaries_text)}
            {patterns_section}
            """
            print(ref_block)
            reference_context_str += ref_block

        # Append shared-community hint so LLM knows which ones to avoid overusing
        if shared_communities:
            reference_context_str += f"""
            ### Cross-Source Note
            Communities {sorted(shared_communities)} appear in **multiple** sources above.
            These are general-purpose communities shared across domains. When assigning `community_id` to additional_anchors, **prefer the long-tail communities** (marked with "long-tail") over the shared ones to maximize differentiation between plans.
            """

        prompt = f"""
        You are a creative System Architect. You are required to design a system based on the user's requirements. Remember that you are facing a designer, so do not use too much technical jargon (like xxx algorithm, engine, neural network, etc.).
        
        ## Context
        The user wants to design a system described as: ```{user_query}```.
        We have already identified the following BASE components (Anchors): ```{base_anchors_str}```.
        
        ## Task
        Generate **{num_outputs}** DISTINCT and DIVERGENT extension plans from different function implementation paths to enhance this system. 
        No matter what perspective you adopt, always keep *align with user requirements* as the primary goal. You can achieve this by using different functions or technologies.

        ## Language Policy
        - Understand the user's input as-is (Chinese or English) without changing its intent.
        - Keep internal reasoning and retrieval assumptions unchanged.
        - Final JSON text fields should be primarily in Chinese for user-facing readability.
        - Proper nouns, model names, protocol names, and indispensable technical terms may remain in English.
        
        ## Reference Material (Inspiration Only — Not a Template)
        We have retrieved knowledge from adjacent domains to spark divergent thinking.
        **Do NOT copy or directly map these references into your plan.** Instead, extract the *underlying mechanism or design principle* and re-express it in the user's specific scenario.
        Ask yourself: "What functional problem does this reference solve, and how can I solve the analogous problem in the user's context?"
        If a reference feels irrelevant or forces an unnatural fit, **ignore it** and rely on domain reasoning instead.
        
        {reference_context_str}
        
        ## Requirements by Plan

        **Plan 1 (Deep Optimization)**:
        - Use **Source 1** (Near-Field) as inspiration. Extract the *core design principle* from the community summaries — do not port technology names or module structures wholesale.
        - If Source 1 states "No papers highly relevant", focus strictly on the user's own scenario: deeply analyze the Base Anchors and the user's requirements to propose functional enhancements that are a **natural, direct extension** of the existing system.
        - Prioritize **feasibility and scenario fit** over novelty. All additional_anchors must directly serve the user's stated goal.

        **Plan 2 (Cross-Domain Inspiration)** (If requested):
        - Use **Source 2** (Far-Field) as your inspiration. Clearly define the core mechanism and advantages of this analogy, and ensure that it significantly differs from other plans.
        - Focus on **innovative cross-domain transfer**: Instead of just introducing conventional ideas, extract typical processing patterns, novel technologies or main implementation methods from Source 2 that can significantly improve the performance of the system.
        - Perform **Concept Blending**: specific methods from Source 2 (e.g., "Medical Monitoring") should be metaphorically and innovatively transferred to the User's domain (e.g., "Education Monitoring").
        - **CRITICAL**: Cross-domain transfer must stay grounded in the user's scenario. The resulting additional_anchors must be recognizable as serving the user's stated goal. Do NOT simply import modules wholesale from Source 2 — they must be adapted, renamed, and functionally redefined in the user's context.

        **Plan 3+** (If requested):
        - Use **Source 3+** similarly to propose a DISTINCT innovative functional embedding, differing in implementation methods or referenced technologies from previous plans.
        - Apply the same grounding constraint as Plan 2: all anchors must clearly serve the user's stated scenario.

        ## General Constraint
        1. When you add new functions, always consider its *feasibility* in the given context.
        2. **Base Anchors Preservation**:
           - **CRITICAL**: The existing Base Anchors represent the core of the system. Your divergent plans MUST acknowledge them, build upon them, and seamlessly integrate new functionalities with them across ALL plans.
        3. **Incremental Only**:
           - Do NOT output the existing Base Anchors in your array. Output ONLY the KEY and ADDITIONAL innovative components (modules). IF Base Anchors do NOT exist, extract the key components from the user's query and add them.
           - **CRITICAL**: Do NOT propose modules that are merely synonyms or simple rephrasings of Base Anchors (e.g., if "Tutor" exists, do not add "Instructor Agent"). New anchors must add **distinct and innovative new functionality**.
           - **LIMIT**: Output at most **5** additional anchors per plan. Prioritize the most impactful and distinctive ones.
        4. **Specific Anchors**:
           - Each additional component must have a 'user_term', a 'description', a 'role', and a 'community_id' (from the reference source).
           - Recheck data flow logic.
        5. **Cross-Plan Differentiation (CRITICAL)**:
           - Before finalizing your output, review ALL plans together. **No two plans may share more than 1 anchor with the same `community_id`.**
           - If two plans would share a community_id on more than 1 anchor, replace the overlapping anchor in the lower-priority plan with a module grounded in a **long-tail community** (marked "long-tail" in the source, which appears less frequently but is more innovative ) or a different community entirely.
           - The `perspective` phrases of all plans must be meaningfully different — they must highlight distinct functional innovations, not just rephrase the same concept.

        ### Role Taxonomy
        {TAXONOMY}\n\n

        ## Naming and Description Rules for Anchors
        - **user_term**: a **noun-form component name** scoped to the user's domain — must pass the sentence test "X is a ____ component that takes A and produces B" with a concrete domain noun. Short (2–8 characters in the user's language). Good: a fruit-maturity recognizer, a branch-avoidance navigator, a harvest-task allocator. Bad: a "weighted re-ranking of results" (verb-flavored, describes a method, not a component).
        - **Hard exclusion**: do NOT propose anchors that are algorithms or techniques (weighting, re-ranking, incremental update, filtering, normalization, Kalman, PID), modalities or data formats (RGB stream, IMU, voice, video stream), or deployment/team groupings (cloud-side, on-device, team-A-owned). These are not components — at the IntegratorAgent stage they would be folded into a real component's `technical_methods` or onto edge labels.
        - **description**: Must explicitly state (1) what data/input this component receives, (2) what single responsibility it performs on that data, and (3) what output it produces. Format: "Receives [input], performs [single responsibility], outputs [result]". Avoid abstract summaries like "performs intelligent decision-making".

        ## Output Structure (JSON)
        {{
            "plans": [
                {{
                    "perspective": "use one short phrase to describe the perspective, highlighting the key functions or biggest advantages and innovations of this design (e.g., 'Immersive interactive feedback'). Do not just copy the perspective title or vague terms like 'cross-domain'",
                    "rationale": "Explain explicitly: 'Inspired by [Source Name: **Feature**], we introduced [Function] to achieve [Goal]...' Ground it in the user's specific scenario.",
                    "additional_anchors": [
                        {{
                            "user_term": "Scenario-specific noun-form component name (e.g., 'fruit-maturity recognition' not 'target detection')",
                            "description": "Receives [concrete input], performs [concrete single responsibility], outputs [concrete result]",
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
        import time
        last_err = None
        for _attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    response_format={"type": "json_object"},
                ).choices[0].message.content
                result = json.loads(response)
                plans = result.get("plans", [])
                # Hard cap: at most 4 additional_anchors per plan to prevent retrieval explosion
                for plan in plans:
                    anchors = plan.get("additional_anchors", [])
                    if len(anchors) > 5:
                        print(f"⚠️ Trimming additional_anchors from {len(anchors)} to 4")
                        plan["additional_anchors"] = anchors[:4]
                if len(plans) < num_outputs:
                    print(f"⚠️ Warning: LLM generated {len(plans)} plans, requested {num_outputs}.")
                return plans[:num_outputs]
            except Exception as e:
                last_err = e
                print(f"❌ Error in divergent planning (attempt {_attempt+1}): {e}")
                if _attempt < 2:
                    time.sleep(3 * (_attempt + 1))
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
                            print(f"Found: {n.node.metadata.get('name', '')} (Score: {n.score:.4f})", flush=True)
                except Exception as exc:
                    print(f"Retrieval generated an exception: {exc}", flush=True)

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


def _expand_compact_ports(response_str):
    """
    Convert compact port strings back to the full object format expected by
    ConflictSolver and the frontend.

    Compact formats:
      incoming:      "SourceMod->label"  or  "SourceMod.SubNode->label"
      outgoing:      "TargetMod->label"  or  "TargetMod.SubNode->label"
      internal_flow: "SubA->SubB->label"
    """
    if not response_str:
        return response_str
    try:
        data = json.loads(response_str)
    except Exception:
        return response_str

    modules = data.get("module_details", [])
    if isinstance(modules, dict):
        modules = list(modules.values())

    for mod in modules:
        ports = mod.get("ports")
        if not isinstance(ports, dict):
            continue

        # --- incoming ---
        raw_in = ports.get("incoming", [])
        expanded_in = []
        for item in raw_in:
            if isinstance(item, dict):
                expanded_in.append(item)
                continue
            if not isinstance(item, str):
                continue
            # split on last "->" to get label
            arrow = item.rfind('->')
            if arrow == -1:
                continue
            src_part = item[:arrow].strip()
            label = item[arrow+2:].strip()
            if '.' in src_part:
                src_mod, src_node = src_part.split('.', 1)
            else:
                src_mod, src_node = src_part, None
            expanded_in.append({
                "source_mod": src_mod.strip(),
                "source_node": src_node.strip() if src_node else None,
                "connects_to_internal_node": None,
                "edge_label": label
            })
        ports["incoming"] = expanded_in

        # --- outgoing ---
        raw_out = ports.get("outgoing", [])
        expanded_out = []
        for item in raw_out:
            if isinstance(item, dict):
                expanded_out.append(item)
                continue
            if not isinstance(item, str):
                continue
            arrow = item.rfind('->')
            if arrow == -1:
                continue
            tgt_part = item[:arrow].strip()
            label = item[arrow+2:].strip()
            if '.' in tgt_part:
                tgt_mod, tgt_node = tgt_part.split('.', 1)
            else:
                tgt_mod, tgt_node = tgt_part, None
            expanded_out.append({
                "target_mod": tgt_mod.strip(),
                "target_node": tgt_node.strip() if tgt_node else None,
                "originates_from_internal_node": None,
                "edge_label": label
            })
        ports["outgoing"] = expanded_out

        # --- internal_flow ---
        raw_flow = ports.get("internal_flow", [])
        expanded_flow = []
        for item in raw_flow:
            if isinstance(item, dict):
                expanded_flow.append(item)
                continue
            if not isinstance(item, str):
                continue
            # format: "SubA->SubB->label"  (first two arrows are node separators)
            parts = item.split('->')
            if len(parts) >= 3:
                src_node = parts[0].strip()
                tgt_node = parts[1].strip()
                label = '->'.join(parts[2:]).strip()
            elif len(parts) == 2:
                src_node = parts[0].strip()
                tgt_node = parts[1].strip()
                label = ""
            else:
                continue
            expanded_flow.append({
                "source_node": src_node,
                "target_node": tgt_node,
                "edge_label": label
            })
        ports["internal_flow"] = expanded_flow

    return json.dumps(data, ensure_ascii=False)


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
    
    @traceable(name="integrator.integrate_arch", run_type="chain")
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

                3. **Connection Discipline**:
                - Connect extension modules only where there is a **direct data dependency** — do NOT add connections simply because two modules are "related" or "could benefit from each other".
                - Each extension module should have at most 2 incoming/outgoing connections to the main pipeline.
                """
        else:
             # Fallback if old call signature is used or no plan
             extension_context = "No specific divergent extensions provided."

        slim_retrieved_info = {}
        for anchor, nodes in retrieved_info.items():
            slim_nodes = []
            for n in nodes:
                raw_content = n.get('text', '')
                # Truncate long content — LLM only needs the key idea, not full text
                content = raw_content[:800] + '…' if len(raw_content) > 800 else raw_content
                node_data = {
                    "module": n.get('metadata', {}).get('name', 'Unknown'),
                    "content": content
                }

                # If this is an added node, explicitly preserve its topology
                if anchor in added_keys:
                    raw_topo = n.get('structural_path', None)
                    if not raw_topo:
                        raw_topo = n.get('metadata', {}).get('topology', None)
                    if raw_topo:
                        node_data["structural_context"] = {
                            "upstream": [u['name'] for u in raw_topo.get('upstream', [])],
                            "downstream": [d['name'] for d in raw_topo.get('downstream', [])]
                        }

                slim_nodes.append(node_data)
            slim_retrieved_info[anchor] = slim_nodes

        prompt = f"""
You are a professional HCI system architect. Design a complete pipeline for the user's requirements.

## User Requirements
{query}

## Base Anchors (MANDATORY — include ALL)
{json.dumps(base_anchors, ensure_ascii=False) if base_anchors else "None provided"}

## Language Policy
User-facing text: primarily Chinese; proper nouns/essential technical terms may stay in English.

## Steps
**1. Main Storyline** — define a clear end-to-end functional mainline; write it as `overall_description`.

**2. Define inputs & outputs** — each system-level input/output is ONE component with a single responsibility. The variety of data sources/sinks lives on the **edges** (incoming/outgoing edge_label), not as sub_nodes. For example, an "environmental perception input" component has multiple incoming edges labeled "RGB image stream", "depth map", "IMU data" — but those modalities are NOT sub_nodes of the input component. Use sub_nodes only when the input/output itself decomposes into functional sub-goals (rare; see Test 1 below).

**3. Define Functional Modules** — cover all mainline steps (including all Base Anchors) that connect to the mainline. Use as many top-level modules as the system genuinely needs; do NOT compress unrelated responsibilities into one module just to hit a count.
- **Naming (CRITICAL)**: A node is a **noun-form component** with a single responsibility, scenario-grounded. Sentence test: "X is a ____ component that takes A and produces B" — if you can't fill the blank with a domain noun, it's not a node. Good: "fruit-maturity recognition" (sentence test passes); Bad: "weighted re-ranking of maturity scores" (verb-flavored, describes a method). No structural suffixes (module/unit/engine/system/component/layer). No algorithm names in node names ever.
- **Data flow clarity**: Every module's `function` field follows IPO: "Receives [specific input], performs [specific duty], outputs [specific result]". The "[specific input]" and "[specific result]" must be data-type names (e.g., "RGB image stream", "list of fruit coordinates") — these match the edge_label of incoming/outgoing edges.
- **Information nodes** (key data artifacts central to user understanding, e.g. "risk warning", "task context"): name describes the information; use sparingly.
- **Descriptions** must be detailed and grounded in retrieved info — no fabrication, no vague summaries.
  - `function`: ≥1 complete sentence in IPO form. Never stop mid-sentence.
  - `technical_methods`: **≥2 complete sentences**. This field is where algorithms/techniques live (weighting, re-ranking, incremental update, filtering, Kalman, normalization, etc.) — name the concrete technique, explain how it implements this component's responsibility in the user's domain, and state why it was chosen over alternatives. NOT processing steps (an "A then B" sequence belongs in `function`).
  - Apply the same depth requirement to sub_nodes.

**3.1 What can become a sub_node — three sequential tests**

A sub_node is a **functional sub-goal** of the parent's single responsibility. Both top-level modules and sub_nodes pass the same three tests; sub_nodes additionally need test T3.

> **[T1] Substitutability test (function vs. algorithm)**
> Imagine swapping the candidate's internal implementation with a completely different technical approach. Does its **position in the diagram and connections to neighbors** stay the same?
>   - Stays the same → it's a **function** (a node candidate)
>   - Changes / disappears → it's an **algorithm/implementation** — fold into parent's `technical_methods`
>
> Example: "lighting estimation" — swap an HSV-based scheme with a neural exposure-correction model; the diagram is unchanged because lighting estimation is internal to the parent "robust recognition" module. It's an implementation, not a function. Fold.

> **[T2] Domain-language test**
> Can a non-technical domain stakeholder name this candidate using only their own goal-language, with no technical vocabulary?
>   - Yes → function
>   - No → it's leaking implementation; fold into `technical_methods`
>
> Example: an orchard owner says "the system must still recognize fruit at dawn, dusk, and on overcast days" (a goal), not "I want lighting estimation" (an implementation). "Lighting estimation" fails T2.

> **[T3] Sibling same-altitude test (sub_nodes only)**
> Are all sub_nodes within one parent at the **same level of abstraction** — all functional sub-goals at user-visible granularity?
>   - If one sibling is more technique-y or scope-y than the others, the level is mixed. Either fold the outlier into `technical_methods`, or lift the whole group to a wider abstraction.
>
> Example (FAIL): parent "AR presentation and multimodal feedback" with siblings "user correction and annotation reporting / field-of-view overlay / voice broadcast" — first is a user-action input; second/third are output channels. Different layers. Fix: move modalities to edge_labels of the output module; "user correction" becomes its own input module.

**Tie-breaker rule**: when all three tests are ambiguous, judge by **information density** — does this candidate carry distinct, scenario-meaningful behavior that would be lost if folded? If yes, keep it as a node. If it's just a thin wrapper around a technique, fold. **Do not pre-compress nodes for parsimony's sake** — the goal is communicative clarity, not minimum count.

**Hard exclusions (always fail above tests; never become nodes or sub_nodes)**:
- **Modalities / data formats** (video stream, IMU, voice, RGB, depth map) — these are edge_labels on incoming/outgoing edges, never nesting.
- **Algorithm / technique words** (weighting, re-ranking, filtering, normalization, PID, Kalman, softmax) — these are sentences inside `technical_methods`.
- **Deployment / team / device groupings** (cloud-side, on-device, team-A-owned) — this system does not output deployment dimensions.

**Sub_node naming and depth requirements**:
- Sub_node `function` follows the same IPO format as top-level (≥1 sentence).
- Sub_node `technical_methods` ≥2 sentences: name the concrete technique, explain how it applies specifically to THIS sub_node's domain context, state activation condition.
- Sub_node names are component nouns scoped to the parent's domain. Example: parent "teach plant recognition" → sub_nodes "identify the plant / explain the knowledge / help the user remember" — three sub-goals at the same altitude. NOT processing stages, NOT modalities, NOT algorithms.
- Integrate all divergent-plan nodes meaningfully.

**4. Connect the Flow**
- Build a clear end-to-end flow. Each module needs ONLY its MINIMUN necessary connections. Do NOT add links indicating "related", "benificial", or "contextal linked".
- **Hard limit**: each module may have at most **2 outgoing** and **2 incoming** ports total. If you exceed this, remove the weakest edge.
- **Feedback / cross-connections**: allow at most **1** feedback loop in the entire diagram, and only when it represents a concrete control signal (e.g., error correction, adaptive calibration). Omit all other secondary edges.
- Default to module-level connections; use subnode-level only when the subnode is the true locus of computation and hiding it would obscure a key mechanism.
- `source_mod` / `target_mod` must reference names defined in `module_details` — no phantom nodes.
- Merge parallel edges between the same pair of nodes. Preserve literature-validated flow order where applicable.

**5. Final Check**
- For every node and sub_node, re-run T1+T2 (and T3 for sub_nodes). If any candidate now fails, fold it into the parent's `technical_methods` and rewire if needed.
- Remove only true duplicates (same function AND same data flow endpoints).
- Reconnect isolated nodes if possible; remove only if unreachable and content-free.

## Reference Material
**Retrieved Modules** — ground `function`/`technical_methods` in these; abstract away paper names; merge similar nodes.
**Structural Paths** — treat as ground truth for data-flow order.

{extension_context}

Retrieved Modules: {json.dumps(slim_retrieved_info, ensure_ascii=False)}

Structural Paths: {hints_text}

## Output (JSON)
Use compact string format for `ports` to save space:
- `incoming`: "SourceMod->edge_label" or "SourceMod.srcSubNode->edge_label"
- `outgoing`: "TargetMod->edge_label" or "TargetMod.tgtSubNode->edge_label"
- `internal_flow`: "SubA->SubB->edge_label"

**Edge Label Guidelines**:
- For **forward flow edges** (normal progression): describe the data/information being passed (e.g., "processed result", "user input", "analysis report")
- For **feedback/loop edges** (returning to earlier stages): describe the trigger condition or control signal (e.g., "anomaly detected", "priority changed", "needs re-evaluation", "quality below threshold")
- Keep labels concise (2-6 characters ideal for Chinese, 3-8 words for English)
- Avoid generic labels like "data" or "info" — be specific about what flows or when it triggers

{{
    "overall_description": "A plan description for less than 100 words, the structure is 'With [the core innovation of perspective] as the core design concept, this system is based on [core processing mechanism], [how to deal with the core challenges of the user scenarios]. [key functional modules and primary path of the data flow].'",
    "module_details": [
        {{
            "name": "noun-form component name (passes the sentence test)",
            "brief_description": "one sentence, scenario-specific, no jargon",
            "function": "IPO: Receives [specific input data], performs [single responsibility], outputs [specific result]",
            "technical_methods": "≥2 sentences. Algorithms/techniques (weighting/re-ranking/filtering/Kalman/normalization etc.) live HERE, not as nodes. Name technique + how implemented in this domain + why chosen.",
            "role": "input|output|processing|data",
            "references": "exact names from Retrieved Modules or 'None'",
            "sub_nodes": [{{"name":"functional sub-goal in parent's domain (passes T1+T2+T3) — never an algorithm, modality, or deployment group","function":"≥1 sentence: Receives [specific input], performs [specific sub-duty], outputs [specific result]","technical_methods":"≥2 sentences: technique implementing this sub-goal + activation condition"}}],
            "ports": {{ // MINIMUN NECESSARY CONNECTIONS
                "incoming": ["ModA->analysis_result", "ModB.SubX->control_signal"],
                "outgoing": ["ModC->processed_data", "ModA->anomaly_detected"],
                "internal_flow": ["SubA->SubB->intermediate_result"]
            }}
        }}
    ]
}}
"""
        prompt_chars = len(prompt)
        print(f"[integrate_arch] prompt size: {prompt_chars} chars (~{prompt_chars//4} tokens est)")
        for anchor, nodes in slim_retrieved_info.items():
            total_content = sum(len(n.get('content','')) for n in nodes)
            print(f"  anchor '{anchor}': {len(nodes)} nodes, {total_content} content chars")
        try:
            response = self.client.chat.completions.create(
                # model="gpt-4o-2024-11-20",
                model = CHAT_MODEL,

                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format={"type": "json_object"},
            ).choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            import time
            for _retry in range(2):
                time.sleep(3 * (_retry + 1))
                try:
                    print(f"Retrying integrate_arch (attempt {_retry + 2})...")
                    response = self.client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        response_format={"type": "json_object"},
                    ).choices[0].message.content
                    break
                except Exception as e2:
                    print(f"Error (retry {_retry + 2}): {e2}")
                    response = None
            else:
                return None
        if not response:
            return None
        # Strip // inline comments that LLMs sometimes insert (causes json.loads to fail)
        import re as _re
        response = _re.sub(r'//[^\n"]*', '', response)
        try:
            json.loads(response)  # validate before returning
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return None
        print(response)
        return _expand_compact_ports(response)

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
        3. Ensuring all 'source_mod' and 'target_mod' in ports refer to valid keys (or 'name' fields if using list) in 'module_details'.

        **CRITICAL REQUIREMENT:**
        `module_details` MUST be a LIST containing the FULL, COMPLETE architecture with ALL nodes (both modified and unmodified) as objects. Do NOT only output the changed nodes. Failure to output unmodified nodes will break the system.

        **Field Preservation (CRITICAL):**
        Preserve ALL existing fields on every module exactly as they appear in the current design — including but not limited to `references`, `function`, `technical_methods`, `role`, `sub_nodes`, `brief_description`. Do NOT drop the `references` string field; the downstream pipeline relies on it to surface literature sources to the user. Only modify what is strictly necessary to fix the reported errors.

        **Naming Discipline (do NOT touch unless directly responsible for an error):**
        Existing node and sub_node names follow these rules — preserve them; do NOT rewrite them while fixing structural issues:
          - A node is a noun-form component with a single responsibility (e.g., "fruit-maturity recognition"); never an algorithm or technique name.
          - Sub_nodes are functional sub-goals of the parent; never modalities (RGB stream / IMU / voice), algorithms (weighting / re-ranking / filtering), or deployment groupings.
          - Algorithms and techniques live in `technical_methods`; modalities live as edge_labels on edges.
        If you must add a NEW node to fix a dangling edge, follow the same rules.

        ## Language Policy
        - Preserve architecture structure and node/link consistency exactly.
        - Keep user-facing textual fields primarily in Chinese.
        - Proper nouns and necessary technical terms may remain in English.

        Return the FULLY CORRECTED JSON object.
        """
        import time
        for _attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                ).choices[0].message.content
                return response
            except Exception as e:
                print(f"Error checking integrity (attempt {_attempt+1}): {e}")
                if _attempt < 1:
                    time.sleep(3 * (_attempt + 1))
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
        if not arch_plan or not isinstance(arch_plan, dict) or "module_details" not in arch_plan:
             return {"used": {}, "explorable": []}

        # 1. Extract Used Reference Names from Plan
        # Map: RefName -> ModuleUserTerm
        used_ref_map = {} 
        
        module_details = arch_plan["module_details"]
        vals = module_details.values() if isinstance(module_details, dict) else module_details
        
        for mod in vals:
            m_name = mod.get("name", "Unknown")
            ref_str = mod.get("references", "")
            # LLM may return references as a list instead of a comma-separated string
            if isinstance(ref_str, list):
                ref_str = ", ".join(str(r) for r in ref_str)
            if ref_str and str(ref_str).lower() != "none":
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


def _rewire_ports(modules, rename_map):
    """In-place rewrite of every source_mod/target_mod reference from an old node
    name to its new name (used after a rename so edges don't dangle). Handles both
    the expanded dict port format and the compact "ModName->label" string format."""
    if not rename_map:
        return
    for m in modules:
        if not isinstance(m, dict):
            continue
        ports = m.get("ports")
        if not isinstance(ports, dict):
            continue
        for key in ("incoming", "outgoing"):
            edges = ports.get(key)
            if not isinstance(edges, list):
                continue
            new_edges = []
            for e in edges:
                if isinstance(e, dict):
                    for fld in ("source_mod", "target_mod"):
                        if e.get(fld) in rename_map:
                            e[fld] = rename_map[e[fld]]
                    new_edges.append(e)
                elif isinstance(e, str):
                    head, sep, rest = e.partition("->")
                    head = head.strip()
                    new_edges.append(f"{rename_map[head]}{sep}{rest}" if head in rename_map else e)
                else:
                    new_edges.append(e)
            ports[key] = new_edges


class EditorAgent:
    def __init__(self, client):
        self.client = client
        # self.model = "gpt-4o-2024-11-20"
        self.model = CHAT_MODEL
        self._solver = None
        self._integrator = None

    def _get_solver(self):
        if self._solver is None:
            self._solver = ConflictSolver()
        return self._solver

    def _get_integrator(self):
        if self._integrator is None:
            self._integrator = IntegratorAgent(self.client)
        return self._integrator

    def _finalize(self, result, current_plan, user_instruction, intent_analysis, prior_memory=None):
        merged = self._merge_plan(result, current_plan, user_instruction, intent_analysis)
        merged = self._conflict_hook(merged, user_instruction)
        merged = self._quality_reflect(merged, intent_analysis, user_instruction)
        merged = self._update_memory(merged, prior_memory, user_instruction, intent_analysis)
        return merged
    
    def critique_plan(self, result, current_plan, user_instruction):
        if not isinstance(result, dict): return {}
        intent_analysis = result.get("intent_analysis") or {}
        self._llm_critique(result, intent_analysis, current_plan, user_instruction)
        return result.get("llm_critique") or {}

    def _update_memory(self, result, prior_memory, user_instruction, intent_analysis):
        """Sink the structured signals this turn already produced into editor memory."""
        try:
            import editor_memory  # local import to avoid a hard load at module import time
            revised_plan = result.get("revised_plan") if isinstance(result, dict) else None
            modification_intent = result.get("modification_intent") if isinstance(result, dict) else None
            unhandled = result.get("unhandled_intents") if isinstance(result, dict) else None
            new_mem = editor_memory.update_memory(
                memory=prior_memory,
                user_instruction=user_instruction,
                intent_analysis=intent_analysis,
                modification_intent=modification_intent,
                revised_plan=revised_plan,
                unhandled_intents=unhandled,
                client=self.client,
                model=self.model,
            )
            if isinstance(result, dict):
                result["editor_memory"] = new_mem
        except Exception as exc:
            print(f"[editor] memory update failed (non-fatal): {exc}")
        return result

    def _conflict_hook(self, result, user_instruction):
        """Detect → optional LLM self-fix → resolve fallback. Never raises."""
        if not isinstance(result, dict):
            return result
        if result.get("output_type") != "plan_update":
            return result
        revised = result.get("revised_plan")
        if not isinstance(revised, dict) or not revised.get("module_details"):
            return result

        try:
            solver = self._get_solver()

            is_valid, report = solver.detect_conflicts(revised)
            if is_valid:
                print("🛡️  Conflict hook: plan passed integrity check, no cleanup needed.")
                result["conflict_report"] = {
                    "was_valid_before_cleanup": True,
                    "self_fix_attempted": False,
                    "detect_report": report,
                }
                return result

            print(f"🛡️  Conflict hook: structural issues detected, attempting LLM self-fix.\n{report}")

            refined_plan = revised
            self_fix_error = None
            try:
                refined_str = self._get_integrator().refine_arch(
                    original_query=user_instruction,
                    current_arch=json.dumps(revised, ensure_ascii=False),
                    error_report=report,
                )
                parsed = json.loads(refined_str) if isinstance(refined_str, str) else refined_str
                if isinstance(parsed, dict) and parsed.get("module_details"):
                    refined_plan = parsed
            except Exception as exc:
                self_fix_error = str(exc)
                print(f"[editor] conflict hook self-fix failed (non-fatal): {exc}")

            # Skip the redundant detect when self-fix errored — refined_plan is unchanged.
            if self_fix_error:
                is_valid_2, report_2 = is_valid, report
            else:
                is_valid_2, report_2 = solver.detect_conflicts(refined_plan)

            if is_valid_2:
                print("✅ Conflict hook: LLM self-fix produced a clean plan.")
                result["revised_plan"] = refined_plan
            else:
                print(f"⚠️  Conflict hook: self-fix incomplete, applying mechanical cleanup.\n{report_2}")
                resolved_str = solver.resolve(refined_plan)
                cleaned = json.loads(resolved_str) if isinstance(resolved_str, str) else resolved_str
                if isinstance(cleaned, dict) and cleaned.get("module_details"):
                    result["revised_plan"] = cleaned

            result["conflict_report"] = {
                "was_valid_before_cleanup": False,
                "self_fix_attempted": True,
                "self_fix_succeeded": bool(is_valid_2),
                "self_fix_error": self_fix_error,
                "detect_report_before": report,
                "detect_report_after": report_2,
            }
        except Exception as exc:
            print(f"[editor] conflict hook failed (non-fatal): {exc}")
            result["conflict_report"] = {"error": str(exc)}
        return result

    @traceable(name="classify_intent", run_type="chain")
    def _classify_intent(self, user_instruction, current_plan, history_summary, focused_modules=None):
        """
        Decompose the user instruction into structured intent slots BEFORE the main
        edit loop runs. Handles MIXED intents (e.g. "delete node A AND look at other-domain inspirations"):
        the main LLM is otherwise prone to answer only the loudest component.

        Returns:
          {
            "primary_intent": "structural_edit | exploration | question | mixed",
            "structural_edits": [{"action": "add|delete|modify|rewire|rename", "target": "...", "detail": "..."}],
            "exploration_topics": ["..."],
            "questions": ["..."],
            "rationale": "..."
          }
        Falls back to an empty-but-valid intent on failure so the main loop still works.
        """
        try:
            module_names = []
            if isinstance(current_plan, dict):
                mds = current_plan.get("module_details", [])
                if isinstance(mds, dict):
                    mds = list(mds.values())
                module_names = [m.get("name", "") for m in mds if isinstance(m, dict) and m.get("name")]

            hist_brief = ""
            if history_summary:
                hist_brief = "; ".join(
                    f"step{item.get('step', '?')}:{(item.get('user_instruction') or item.get('perspective') or '')[:40]}"
                    for item in history_summary[-3:]
                )

            sys_prompt = (
                "You are an intent parser for an architecture-editing assistant. "
                "Parse the user's instruction into structured slots.\n"
                "CRITICAL: a single instruction may contain MULTIPLE intent components "
                "(e.g. a hard structural edit AND an open-ended exploration request). "
                "Extract EVERY component separately — never collapse them into one.\n\n"
                "Slots:\n"
                "- structural_edits: concrete graph mutations (add/delete/modify/rewire/rename a module or sub_node). "
                "Each item must be {action, target, detail}.\n"
                "- exploration_topics: open-ended inspiration / cross-domain reference / brainstorming asks.\n"
                "- questions: pure information questions about the current plan (no mutation, no exploration).\n"
                "- primary_intent: one of structural_edit | exploration | question | mixed.\n"
                "- rationale: one short sentence.\n"
                "SCOPE RULE: if a SELECTED module is given below and the instruction uses a deictic/singular "
                "reference ('this module', '这个模块', 'it', '它'), the structural_edits MUST target ONLY the "
                "selected module(s) — do NOT expand the edit to every module in the plan.\n"
                "Output STRICT JSON only."
            )
            focus_line = ""
            if focused_modules:
                _fm = focused_modules if isinstance(focused_modules, list) else [focused_modules]
                _fm = [str(n) for n in _fm if n]
                if _fm:
                    focus_line = f"Selected module(s) on canvas (referent of 'this module'): {_fm}\n"
            user_prompt = (
                f"Existing module names: {module_names}\n"
                f"{focus_line}"
                f"Recent history: {hist_brief or 'none'}\n"
                f"User instruction: {user_instruction}"
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            intent = json.loads(resp.choices[0].message.content or "{}")
            intent.setdefault("structural_edits", [])
            intent.setdefault("exploration_topics", [])
            intent.setdefault("questions", [])
            if not intent.get("primary_intent"):
                filled = sum(1 for k in ("structural_edits", "exploration_topics", "questions") if intent.get(k))
                if filled >= 2:
                    intent["primary_intent"] = "mixed"
                elif intent["structural_edits"]:
                    intent["primary_intent"] = "structural_edit"
                elif intent["exploration_topics"]:
                    intent["primary_intent"] = "exploration"
                elif intent["questions"]:
                    intent["primary_intent"] = "question"
                else:
                    intent["primary_intent"] = "structural_edit"
            print(
                f"🎯 Intent: {intent['primary_intent']} | "
                f"edits={len(intent['structural_edits'])} "
                f"explore={len(intent['exploration_topics'])} "
                f"q={len(intent['questions'])}"
            )
            return intent
        except Exception as exc:
            print(f"[editor] intent classification failed, falling back: {exc}")
            return {
                "primary_intent": "structural_edit",
                "structural_edits": [],
                "exploration_topics": [],
                "questions": [],
                "rationale": "fallback (parser unavailable)",
            }

    @staticmethod
    def _format_intent_block(intent):
        """Render the intent dict as a prompt section the main LLM must address point-by-point."""
        lines = ["## Parsed User Intent (you MUST address every component below)"]
        lines.append(f"- primary_intent: {intent.get('primary_intent', 'structural_edit')}")
        if intent.get("rationale"):
            lines.append(f"- rationale: {intent['rationale']}")
        edits = intent.get("structural_edits") or []
        if edits:
            lines.append("- structural_edits (HARD requirements — must be applied to the plan):")
            for i, e in enumerate(edits, 1):
                lines.append(
                    f"    {i}. action={e.get('action', '?')} | "
                    f"target={e.get('target', '?')} | detail={e.get('detail', '')}"
                )
        explore = intent.get("exploration_topics") or []
        if explore:
            lines.append("- exploration_topics (SOFT inspiration — surface options, do not silently rewrite the whole plan):")
            for i, t in enumerate(explore, 1):
                lines.append(f"    {i}. {t}")
        qs = intent.get("questions") or []
        if qs:
            lines.append("- questions (pure Q&A — answer without mutating the plan):")
            for i, q in enumerate(qs, 1):
                lines.append(f"    {i}. {q}")
        lines.append(
            "Routing rule: if structural_edits is non-empty, output_type MUST be plan_update "
            "AND every listed edit must be reflected in module_details. "
            "If only exploration_topics/questions exist, prefer evidence_answer or direct_answer "
            "and DO NOT mutate the plan unless the user explicitly asked to."
        )
        return "\n        ".join(lines)
    
    @traceable(name="Editor.edit_plan", run_type="chain")
    def edit_plan(self, current_plan, user_instruction, history_summary, retriever, referenced_modules=None, memory=None, focused_modules=None):
        """
        Analyzes the user's instruction, retrieves necessary information, and revises the architecture.
        Maintains modification history for context.

        `memory` is the new tiered memory snapshot (see editor_memory.py). When omitted,
        we bootstrap one from the legacy `history_summary` list so older callers keep working.
        `focused_modules` is the list of module name(s) the user has SELECTED on the canvas;
        it disambiguates deictic references ("this module") so the edit stays scoped.
        """
        print(f"🧠 Editor: Processing modification request with lineage history...")

        # 0. Intent classification — decompose mixed intents up front
        intent_analysis = self._classify_intent(user_instruction, current_plan, history_summary, focused_modules=focused_modules)
        intent_block = self._format_intent_block(intent_analysis)

        # RC5: only offer retrieval tools when the edit plausibly needs EXTERNAL knowledge.
        # Pure rename/delete/reorganize or completing descriptions of EXISTING modules needs
        # no retrieval — skipping it avoids slow, useless tool round-trips.
        _edits = intent_analysis.get("structural_edits") or []
        _explore = intent_analysis.get("exploration_topics") or []
        _questions = intent_analysis.get("questions") or []
        _has_add = any((e.get("action") or "").lower() == "add" for e in _edits if isinstance(e, dict))
        allow_tools = not (
            intent_analysis.get("primary_intent") == "structural_edit"
            and not _has_add and not _explore and not _questions
        )
        print(f"🔧 Editor retrieval tools: {'enabled' if allow_tools else 'disabled (pure structural edit — no new knowledge needed)'}")

        # Bootstrap memory if caller didn't supply one (legacy path).
        import editor_memory
        if memory is None:
            memory = editor_memory.memory_from_history_summary(history_summary)

        # 1. Prepare Tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_knowledge",
                    "description": "Retrieve domain knowledge or technical details from the knowledge graph. Use it when the user's request involves specific concepts, technologies, or complex workflows that are not included in the plan and not fully understood or need external validation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant modules or patterns."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_topology",
                    "description": "Retrieve structural connection paths between two modules from the knowledge graph. Use this when you need to understand how two components are typically connected in literature, or when you are adding/reconnecting a module and need to know its upstream/downstream neighbors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Name or description of the source module."
                            },
                            "target": {
                                "type": "string",
                                "description": "Name or description of the target module."
                            }
                        },
                        "required": ["source", "target"]
                    }
                }
            }
        ]

        # 2. Prepare Context & Memory-rendered History
        in_focus_modules = editor_memory.compute_in_focus(intent_analysis, current_plan if isinstance(current_plan, dict) else None)
        history_text = editor_memory.format_for_prompt(memory, in_focus_modules)

        # Ensure current plan is a dict
        if isinstance(current_plan, str):
            try:
                current_plan = json.loads(current_plan)
            except:
                pass

        # Slim down the plan for the editor prompt: drop heavy fields not needed for decision-making
        def _ports_to_compact(ports):
            """Convert expanded port objects back to compact string format for the LLM prompt."""
            if not isinstance(ports, dict):
                return ports
            compact = {}

            incoming = []
            for edge in ports.get("incoming", []):
                if isinstance(edge, str):
                    incoming.append(edge)
                    continue
                src = edge.get("source_mod", "")
                src_node = edge.get("source_node") or edge.get("connects_to_internal_node")
                label = edge.get("edge_label", "")
                prefix = f"{src}.{src_node}" if src_node else src
                incoming.append(f"{prefix}->{label}" if label else prefix)
            if incoming:
                compact["incoming"] = incoming

            outgoing = []
            for edge in ports.get("outgoing", []):
                if isinstance(edge, str):
                    outgoing.append(edge)
                    continue
                tgt = edge.get("target_mod", "")
                tgt_node = edge.get("target_node") or edge.get("originates_from_internal_node")
                label = edge.get("edge_label", "")
                prefix = f"{tgt}.{tgt_node}" if tgt_node else tgt
                outgoing.append(f"{prefix}->{label}" if label else prefix)
            if outgoing:
                compact["outgoing"] = outgoing

            internal = []
            for flow in ports.get("internal_flow", []):
                if isinstance(flow, str):
                    internal.append(flow)
                    continue
                src_n = flow.get("source_node", "")
                tgt_n = flow.get("target_node", "")
                label = flow.get("edge_label", "")
                internal.append(f"{src_n}->{tgt_n}->{label}" if label else f"{src_n}->{tgt_n}")
            if internal:
                compact["internal_flow"] = internal

            return compact

        def _slim_plan(plan, in_focus=None):
            if not isinstance(plan, dict):
                return plan
            slim = {k: v for k, v in plan.items() if k != "module_details"}
            mds = plan.get("module_details", [])
            if isinstance(mds, dict):
                mds = list(mds.values())
            # Focus-aware: only drop heavy fields when the canvas is large enough
            # AND the focus set genuinely narrows things down.
            use_focus = (
                in_focus is not None
                and len(mds) > editor_memory.FOCUS_AWARE_THRESHOLD
                and 0 < len(in_focus) < len(mds)
            )
            slim_mds = []
            for m in mds:
                name = m.get("name", "")
                if use_focus and name not in in_focus:
                    entry = {k: m[k] for k in ("name", "role") if k in m}
                else:
                    entry = {k: m[k] for k in ("name", "brief_description", "role", "sub_nodes") if k in m}
                if "ports" in m:
                    entry["ports"] = _ports_to_compact(m["ports"])
                slim_mds.append(entry)
            slim["module_details"] = slim_mds
            return slim

        slim_plan = _slim_plan(current_plan, in_focus_modules)

        # Build referenced modules block before f-string (avoid nested f-string syntax error)
        if referenced_modules:
            ref_lines = "\n".join(
                f"- [{m.get('plan_name', 'Other Plan')}] {m.get('module_name', '')}:\n  {json.dumps(m.get('module_data', {}), ensure_ascii=False)}"
                for m in referenced_modules
            )
            referenced_modules_section = f"""
        ## Referenced Modules from Other Plans (User-Selected Context)
        The user has explicitly referenced the following modules from alternative design plans as inspiration or comparison. When addressing the user's instruction, consider these modules — you may borrow their structure, logic, or sub-nodes where appropriate, while adapting them to fit the current architecture's naming conventions and domain context.
        {ref_lines}
                """
        else:
            referenced_modules_section = ""

        # Focused-selection block: scope deictic references to the selected node(s).
        if focused_modules:
            _focus_names = focused_modules if isinstance(focused_modules, list) else [focused_modules]
            _focus_names = [str(n) for n in _focus_names if n]
            focused_block = (
                "## Focused Module(s) — User Selection\n"
                f"The user currently has these module(s) SELECTED on the canvas: {_focus_names}.\n"
                "If the instruction uses a deictic/singular reference ('这个模块', '这个节点', 'this module', "
                "'it', '它', 'the module') and does NOT clearly describe a plan-wide operation, then it is about "
                "the SELECTED module(s): restrict ALL structural_edits to ONLY those module(s) and mark every "
                "other module 'unchanged' — do NOT rename or modify unrelated modules. "
                "If the instruction is explicitly plan-wide (e.g. names other modules, or says 'all modules' / "
                "'keep only ...'), ignore this selection and follow the instruction as written.\n"
            ) if _focus_names else ""
        else:
            focused_block = ""

        system_prompt = f"""
        You are an expert System Architect. You are collaborating with a user to modify a system architecture design.
        
        ## Your Capabilities
        1. **Analyze**: Understand the user's intent and the current architecture.
        2. **Retrieve Knowledge**: If the user asks for something that needs specific knowledge or technical details (e.g., "Add a Kafka stream", "Use a Transformer model"), use the `retrieve_knowledge` tool to get technical details.
        3. **Retrieve Topology**: If you are adding, reconnecting, or repositioning a module and need to know how it typically connects to other components in literature, use the `retrieve_topology` tool with the source and target module names. This returns validated structural paths from the knowledge graph.
        4. **Modify**: Update the architecture JSON to reflect the changes.
        
        ## Evolution Context
        This is the current state of the plan. Here is the history of how it evolved:
        {history_text}

        ## Current Architecture
        {json.dumps(slim_plan, ensure_ascii=False)}
        {referenced_modules_section}

        {focused_block}
        {intent_block}

        ## Task
        The user has provided an instruction. You must:
        1. Address EVERY component listed in "Parsed User Intent" above — never silently drop a structural_edit because exploration_topics looked more interesting.
        2. Stay consistent with the history (avoid repeating undone changes, or building on top of recent changes).
        3. If you need external knowledge (e.g., technical specs, similar modules), call `retrieve_knowledge`.
        4. If you are adding or reconnecting a module and need to know its typical upstream/downstream neighbors, call `retrieve_topology`.
        5. Pick the response mode that matches the parsed intent:
           - plan_update: any structural_edits requested → mutate the architecture JSON
           - direct_answer: plain answer, no plan change, no external evidence required
           - evidence_answer: answer with citations/evidence, no plan change
           - plan_explain: explain the current plan without changing it
        6. Output a FINAL JSON response matching the required format below.

        ## Language Policy
        - Understand the user instruction as-is (Chinese or English) without changing intent.
        - Do not alter plan structure rules or retrieval behavior because of language choice.
        - In final output, user-facing textual fields should be primarily in Chinese.
        - Proper nouns and indispensable technical terms may remain in English.

        **Semantics Rule for Module and Subnodes**:
            - **Node = noun-form component with single responsibility**. Sentence test: "X is a ____ component that takes A and produces B" must fill with a domain noun. Good: "fruit-maturity recognition" ✓; Bad: "weighted re-ranking of maturity scores" ✗ (verb-flavored, describes a method). No structural suffixes (module/unit/engine/system/component/layer). No algorithm names in node names ever.
            - **Three sequential tests for any new or renamed node / sub_node**:
              [T1] **Substitutability**: swap the candidate's internal implementation with a totally different technical approach — does its position in the diagram and its connections to neighbors stay the same? Stays → function (node candidate); changes → algorithm/implementation, fold into parent's `technical_methods`. Example: "lighting estimation" — swap an HSV-based scheme with a neural exposure-correction model; diagram unchanged → it's an implementation of the parent "robust recognition" module, fold.
              [T2] **Domain language**: can a non-technical domain stakeholder name this with their own goal-language, no technical vocabulary? Yes → function; no → leaks implementation, fold.
              [T3] **Sibling same-altitude (sub_nodes only)**: are all sub_nodes in one parent at the same abstraction level (all functional sub-goals at user-visible granularity)? If one sibling looks more technique-y or scope-y, level is mixed — fold the outlier or lift the whole group.
            - **Tie-breaker**: when tests are ambiguous, judge by information density — distinct scenario-meaningful behavior worth its own node? Keep. Thin wrapper around a technique? Fold. Do NOT pre-compress for parsimony.
            - **Hard exclusions** (always fail tests; never become nodes or sub_nodes):
              * Modalities/data formats (video stream / IMU / RGB / voice / depth map) → edge_label on incoming/outgoing edges
              * Algorithms/techniques (weighting / re-ranking / filtering / normalization / Kalman / PID / softmax) → sentences inside parent's `technical_methods`
              * Deployment/team/device groupings (cloud-side / on-device / team-A-owned) → not in this system's output
            - **Sub_node = functional sub-goal of parent's single responsibility**. Example: parent "teach plant recognition" → sub_nodes "identify the plant / explain the knowledge / help the user remember", three sub-goals at the same altitude. NOT processing stages, NOT modalities, NOT algorithms.
            - **function field**: IPO format — "Receives [specific input data], performs [single responsibility], outputs [specific result]". Inputs/outputs are data-type names matching edge_labels.
            - **technical_methods field**: ≥2 sentences. Algorithms and techniques live HERE — name technique + how implemented in this domain + why chosen. NOT processing steps (an "A then B" sequence belongs in `function`).
            - Ensure the added nodes from the divergent plans are integrated meaningfully.
            - **Ground both fields in retrieved info; do NOT fabricate.**
        **Connections**:
            - Build a clear primary end-to-end flow with MINIMUN NECESSARY connections.
            - Hard limit: at most **2 outgoing** and **2 incoming** ports per module. At most **1** feedback loop in the entire diagram.


        ## Modification Rules
        1. **Locality**: Only change what is requested.
           - In single-node mode, modify only the target module, its `sub_nodes`, and minimally required adjacent edges.
           - Keep all other modules unchanged unless a strict consistency fix is necessary.
        2. **Preservation**: Keep existing modules unless asked to remove them.
        3. **Consistency**: Ensure the graph remains connected and logical.
        4. **Integrity**: Ensure all `source_mod` and `target_mod` in ports refer to valid keys in `module_details`.
        5. **Naming Rule**: When adding or renaming nodes, apply the three tests (T1 substitutability, T2 domain-language, T3 sibling same-altitude) from Semantics Rule above. Names are noun-form components scoped to the user's domain. Reject candidates that are algorithms (weighting / re-ranking / filtering), modalities (video stream / IMU / voice), or deployment groupings — fold them into `technical_methods` or onto edge_labels. Examples: "priority explanation" not "priority explanation display unit"; "harvest-task dynamic scheduling" not "dynamic priority and resource decision module"; "fruit feature extraction" not "task feature extraction unit".
        
        
        ## Final Output Format
        If you choose plan_update, output:
        {{
            "output_type": "plan_update",
            "modification_intent": "A concise summary of what you changed (e.g., 'Added Authentication Module and connected it to User Interface')",
            "chat_response": {{
                "text": "User-facing chat reply (Chinese, per Language Policy). MUST explicitly respond to every exploration_topic and every question in Parsed User Intent. When only structural_edits are present, a brief confirmation of the change is enough.",
                "addressed_explorations": [
                    {{"topic": "the matching exploration_topic, verbatim", "response": "brief note on what you borrowed / alternative ideas offered", "applied_to_plan": true}}
                ],
                "addressed_questions": [
                    {{"question": "the matching question, verbatim", "answer": "direct answer, no hedging"}}
                ],
                "citations": [
                    {{"title": "...", "snippet": "...", "source": "..."}}
                ]
            }},
            "revised_plan": {{
                "overall_description": "The overall description of the ENTIRE revised system architecture. Update the original description to reflect your changes, but DO NOT just write your modification intent here.",
                "module_details": [
                    {{
                        "name": "NewOrModifiedModuleA",
                        "change_type": "added | modified",
                        "brief_description": "...",
                        "function": "Detailed explanation of the module's function, its core responsibilities, and its crucial role and importance within the overall system architecture.",
                        "technical_methods": "Detailed description of the specific techniques, algorithms, equipments or tools used as **implementation means** — NOT processing steps or execution logic. Explain HOW each technique implements the module's capability and WHY it was chosen. You MUST abstract away the original paper context.",
                        "role": "choose from ['input', 'output', 'processing', 'data']",
                        "sub_nodes": [],
                        "ports": {{}}
                    }},
                    {{
                        "name": "ExistingModuleWhosePortsChanged",
                        "change_type": "ports_only",
                        "ports": {{}}
                    }},
                    {{
                        "name": "NewModuleName",
                        "old_name": "CurrentModuleName",
                        "change_type": "modified",
                        "brief_description": "...",
                        "function": "...",
                        "technical_methods": "...",
                        "role": "...",
                        "sub_nodes": [],
                        "ports": {{}}
                    }},
                    {{
                        "name": "ModuleToRemove",
                        "change_type": "deleted"
                    }},
                    {{
                        "name": "UnchangedModuleC",
                        "change_type": "unchanged"
                    }}
                ]
            }}
        }}

        **Mixed-intent rule (CRITICAL)**: when Parsed User Intent contains exploration_topics or questions
        alongside structural_edits, `chat_response.text` MUST contain a full user-facing reply that
        addresses every exploration and question item. Do NOT swallow them into modification_intent —
        modification_intent is for the change log, chat_response is for the user.

        If you choose direct_answer, evidence_answer, or plan_explain, output:
        {{
            "output_type": "direct_answer | evidence_answer | plan_explain",
            "answer": {{
                "text": "Your response to the user.",
                "citations": [
                    {{
                        "title": "Source title or module name",
                        "snippet": "Short evidence snippet",
                        "source": "Origin if available"
                    }}
                ]
            }}
        }}

        **CRITICAL REQUIREMENT:**
        1. `overall_description` MUST describe the full, comprehensive system workflow after your changes.
        2. `module_details` MUST be a LIST containing ALL nodes using the diff format above:
           - Newly added or content-modified nodes: include ALL fields with `"change_type": "added"` or `"change_type": "modified"`.
           - Nodes whose only change is connection rewiring: include ONLY `name`, `"change_type": "ports_only"`, and the new `ports` object.
           - Completely unmodified nodes: include ONLY `name` and `"change_type": "unchanged"`. Do NOT omit them.
           - **To RENAME a node**: use `"change_type": "modified"` and include BOTH `"old_name"` (its CURRENT name) and `"name"` (the NEW name). NEVER emit a renamed node as a brand-new entry without `old_name` — that creates a duplicate of the old node. Connections to the old name are rewired automatically.
           - **To DELETE a node**: emit ONLY `{{"name": "<current name>", "change_type": "deleted"}}`. Do NOT express deletion by silently omitting the node — omitted nodes are treated as accidental drops and restored.
        3. **Scope discipline**: only emit `added`/`modified`/`ports_only`/`deleted` for the nodes the user's Parsed User Intent actually targets. If the user points at ONE module ("this module"), change ONLY that module and mark every other node `unchanged`. Do NOT rename or rewrite unrelated modules.
        4. When adding a new module that connects to existing modules, you MUST also output `ports_only` entries for those existing modules to reflect the new connections.
        5. For non-plan_update modes, do NOT output revised_plan and do NOT alter the plan.
        """

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_instruction})

        # 3. Execution Loop
        max_steps = 5
        current_step = 0
        
        while current_step < max_steps:
            current_step += 1

            try:
                # Only pass response_format when not using tools, as some APIs
                # do not support tools + response_format=json_object simultaneously.
                # Retry up to 2 times on timeout — the relay API has occasional timeouts.
                response = None
                last_exc = None
                _create_kwargs = {"model": self.model, "messages": messages}
                if allow_tools:
                    _create_kwargs["tools"] = tools
                    _create_kwargs["tool_choice"] = "auto"
                for _attempt in range(2):
                    try:
                        response = self.client.chat.completions.create(**_create_kwargs)
                        break
                    except Exception as exc:
                        last_exc = exc
                        print(f"[editor] LLM attempt {_attempt+1} failed: {exc}")
                if response is None:
                    raise last_exc

                msg = response.choices[0].message

                # Check for tool calls
                if msg.tool_calls:
                    messages.append(msg) # Add assistant's message with tool calls
                    
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == "retrieve_knowledge":
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query")
                            print(f"🔍 Editor calling tool: retrieve_knowledge('{query}')")

                            # Execute retrieval using the passed retriever instance
                            # We'll use a simplified retrieval wrapper here
                            # Create fake anchors for the retriever
                            search_anchors = [{"user_term": query, "description": "User requested enhancement"}]
                            retrieved_map = retriever.retrieve_for_module(search_anchors, top_k=3, include_topology=True)

                            # Format results
                            context_text = f"### Retrieval Results for '{query}':\n"
                            found_any = False
                            for term, nodes in retrieved_map.items():
                                for n in nodes:
                                    found_any = True
                                    title = n.node.metadata.get("name", "Unknown")
                                    source = n.node.metadata.get("source", "")
                                    snippet = n.node.text[:500]
                                    context_text += f"- Title: {title}\n"
                                    context_text += f"  Snippet: {snippet}...\n"
                                    context_text += f"  Source: {source}\n"

                            if not found_any:
                                context_text += "No relevant modules found."

                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": "retrieve_knowledge",
                                "content": context_text
                            })

                        elif tool_call.function.name == "retrieve_topology":
                            args = json.loads(tool_call.function.arguments)
                            source = args.get("source")
                            target = args.get("target")
                            print(f"🔍 Editor calling tool: retrieve_topology('{source}' -> '{target}')")

                            # Execute structural path retrieval
                            user_relations = [{"source": source, "target": target, "type": "FLOWS_TO"}]
                            structural_paths = retriever.retrieve_structure_paths(user_relations)

                            # Format results similar to IntegratorAgent.format_hints
                            context_text = f"### Structural Paths for '{source}' -> '{target}':\n"
                            if structural_paths:
                                for i, path in enumerate(structural_paths):
                                    nodes = path.get('path_nodes', [])
                                    rels = path.get('relations', [])
                                    rel_props = path.get('rel_props', [])

                                    chain_desc = []
                                    if nodes:
                                        start_node_name = nodes[0].get('name', 'Unknown')
                                        chain_desc.append(f"[{start_node_name}]")

                                        for j, rel in enumerate(rels):
                                            if j + 1 < len(nodes):
                                                target_node = nodes[j+1].get('name', 'Unknown')
                                                func = nodes[j+1].get('function', '')
                                                if func:
                                                    target_node += f" ({func[:100]}...)"

                                                data_content = ""
                                                if j < len(rel_props) and rel_props[j] and 'edge_label' in rel_props[j]:
                                                    lbl = rel_props[j]['edge_label']
                                                    if lbl and lbl != "null":
                                                        data_content = f": {lbl}"

                                                if rel == "FLOWS_TO":
                                                    chain_desc.append(f" --({data_content})--> [{target_node}]")
                                                else:
                                                    chain_desc.append(f" --({rel}{data_content})--> [{target_node}]")

                                    context_text += f"  Path {i+1}: {''.join(chain_desc)}\n"
                            else:
                                context_text += "No structural paths found in literature. You may need to infer the connection based on domain knowledge."

                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": "retrieve_topology",
                                "content": context_text
                            })
                    
                    # Continue loop to let model use the info
                    continue
                
                # If no tool calls, this is the final response.
                # Re-request without tools but with response_format=json_object to
                # guarantee valid JSON output (tools + response_format causes timeouts
                # on this relay API, so we separate the two calls).
                # Pass the draft content back so the model only reformats, not re-reasons.
                format_messages = messages + [
                    {"role": "assistant", "content": msg.content or ""},
                    {"role": "user", "content": "Output exactly the above as valid JSON only, no markdown, no extra text."},
                ]
                final_response = None
                last_final_exc = None
                for _attempt in range(2):
                    try:
                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=format_messages,
                            response_format={"type": "json_object"},
                        )
                        break
                    except Exception as exc:
                        last_final_exc = exc
                        print(f"[editor] final LLM attempt {_attempt+1} failed: {exc}")
                if final_response is None:
                    raise last_final_exc
                content = final_response.choices[0].message.content
                try:
                    import re
                    # Try direct JSON parsing
                    try:
                        result = json.loads(content)
                        return self._finalize(result, current_plan, user_instruction, intent_analysis, prior_memory=memory)
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON directly. Attempting regex extraction...")
                        # Fallback: extract JSON from markdown code blocks
                        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(1))
                                return self._finalize(result, current_plan, user_instruction, intent_analysis, prior_memory=memory)
                            except json.JSONDecodeError:
                                pass

                        # Fallback: extract outermost JSON object
                        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(1))
                                return self._finalize(result, current_plan, user_instruction, intent_analysis, prior_memory=memory)
                            except json.JSONDecodeError:
                                pass

                        print(f"Error parsing final JSON even after extraction. Retrying...")
                        # Append a user message asking to fix the JSON format
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": "The JSON you provided is invalid. Please ensure it is strictly valid JSON without any // comments, trailing commas, or markdown outside the valid JSON object format."})
                        continue

                except Exception as e:
                    print(f"Unexpected error when parsing content: {e}")
                    continue

            except Exception as e:
                print(f"Error in edit_plan loop: {e}")
                break

        return None

    def _merge_plan(self, result, original_plan, user_instruction, intent_analysis=None):
        """
        Post-edit reflection: merge diff output back into the full plan.
        Nodes with change_type='unchanged' are restored from original_plan.
        Nodes with change_type='ports_only' get only their ports updated.
        Nodes with change_type='added'/'modified' are taken as-is.
        Nodes with change_type='deleted' (or named as a delete target in
        intent_analysis) are removed and NEVER auto-restored.
        Renames carry old_name -> name: identity is preserved and edges referencing
        the old name are rewired.

        The "silently dropped -> restore" safety net only fires for nodes the user
        did NOT intend to touch, so deletes and renames are no longer undone.
        """
        if not isinstance(result, dict) or result.get("output_type") != "plan_update":
            return result

        revised_plan = result.get("revised_plan")
        if not revised_plan:
            return result

        def get_module_list(plan):
            if not isinstance(plan, dict):
                return []
            md = plan.get("module_details", [])
            if isinstance(md, list):
                return md
            if isinstance(md, dict):
                return list(md.values())
            return []

        orig_modules = get_module_list(original_plan if isinstance(original_plan, dict) else {})
        orig_lookup  = {m.get("name", ""): m for m in orig_modules if m.get("name")}
        diff_list    = get_module_list(revised_plan)

        # Check if diff format is being used (any entry has change_type)
        uses_diff = any("change_type" in m for m in diff_list if isinstance(m, dict))

        # Intentional-removal / edit-scope signals from the upfront intent parse, so the
        # restore safety net below does NOT resurrect nodes the user asked to delete or
        # rename away (the root cause of failed deletes and rename duplication).
        intent_edits = (intent_analysis or {}).get("structural_edits") or []
        intent_delete_targets = {
            (e.get("target") or "").strip()
            for e in intent_edits
            if isinstance(e, dict) and e.get("action") == "delete" and (e.get("target") or "").strip()
        }
        intent_targets = {
            (e.get("target") or "").strip()
            for e in intent_edits
            if isinstance(e, dict) and (e.get("target") or "").strip()
        }

        if uses_diff:
            merged = []
            diff_names = set()
            deleted_names = set(intent_delete_targets)
            consumed_originals = set()   # originals renamed away (old_name -> new name)
            rename_map = {}              # old_name -> new_name, for edge rewiring

            for entry in diff_list:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                if not name:
                    continue
                ct = entry.get("change_type", "modified")

                # Explicit deletion: drop it, and remember so we never restore it.
                if ct == "deleted":
                    deleted_names.add(name)
                    if entry.get("old_name"):
                        deleted_names.add(entry["old_name"])
                    continue

                diff_names.add(name)

                # Rename: old_name (current) -> name (new). Preserve identity + rewire edges.
                old_name = entry.get("old_name")
                if old_name and old_name != name:
                    rename_map[old_name] = name
                    consumed_originals.add(old_name)

                src_node = orig_lookup.get(name) or (orig_lookup.get(old_name) if old_name else None)

                if ct == "unchanged":
                    if src_node is not None:
                        node = dict(src_node)
                        node["name"] = name
                        merged.append(node)
                    else:
                        print(f"⚠️  Reflection: unchanged node '{name}' not found in original, skipping.")
                elif ct == "ports_only":
                    # Take original node but overwrite ports.
                    # LLM may output compact string format ("ModA->label") — expand first.
                    raw_ports_entry = json.dumps({"module_details": [{"ports": entry.get("ports", {})}]}, ensure_ascii=False)
                    expanded_str = _expand_compact_ports(raw_ports_entry)
                    try:
                        expanded_ports = json.loads(expanded_str)["module_details"][0]["ports"]
                    except Exception:
                        expanded_ports = entry.get("ports", {})
                    if src_node is not None:
                        restored = dict(src_node)
                        restored["name"] = name
                        restored["ports"] = expanded_ports
                        merged.append(restored)
                    else:
                        # New node with only ports? Keep as-is (shouldn't happen)
                        merged.append(entry)
                else:
                    # added or modified: expand compact ports before using
                    if "ports" in entry and entry["ports"]:
                        raw = json.dumps({"module_details": [{"ports": entry["ports"]}]}, ensure_ascii=False)
                        try:
                            expanded_ports = json.loads(_expand_compact_ports(raw))["module_details"][0]["ports"]
                            entry = {**entry, "ports": expanded_ports}
                        except Exception:
                            pass
                    merged.append(entry)

            # Safety net: restore originals missing from the diff — BUT never resurrect a
            # node the user intentionally deleted, renamed away, or otherwise targeted.
            instr_lower = user_instruction.lower()
            for name, orig_node in orig_lookup.items():
                if name in diff_names or name in deleted_names or name in consumed_originals:
                    continue
                if name in intent_targets:
                    # The user explicitly asked to act on this node and the model did not
                    # emit it — trust that membership decision (it was transformed/renamed/
                    # removed) instead of blindly restoring it (prevents duplication).
                    print(f"⚠️  Reflection: edit target '{name}' absent from diff — treating as intentionally transformed/removed, not restoring.")
                    continue
                if name.lower() in instr_lower:
                    continue
                print(f"⚠️  Reflection: silently dropped node '{name}', restoring.")
                merged.append(orig_node)

            # Hard-drop anything marked deleted that slipped through, then rewire renamed edges.
            if deleted_names:
                merged = [m for m in merged if isinstance(m, dict) and m.get("name") not in deleted_names]
            if rename_map:
                _rewire_ports(merged, rename_map)

            # Strip the internal rename marker so it doesn't leak to the frontend.
            for m in merged:
                if isinstance(m, dict):
                    m.pop("old_name", None)

            revised_plan["module_details"] = merged
            result["revised_plan"] = revised_plan
            print(f"✅ Reflection (diff mode): merged {len(merged)} nodes "
                  f"(deleted={len(deleted_names)}, renamed={len(rename_map)}).")
            return result

        # Legacy full-output mode: check for silently dropped nodes
        revised_list  = diff_list
        revised_names = {m.get("name", "") for m in revised_list if m.get("name")}
        dropped       = set(orig_lookup.keys()) - revised_names

        if not dropped:
            print("✅ Reflection: No nodes were silently dropped.")
            return result

        instr_lower = user_instruction.lower()
        # Honor intentional deletes/renames here too — don't restore what the user
        # asked to remove or transform.
        to_restore = [
            name for name in dropped
            if name.lower() not in instr_lower
            and name not in intent_delete_targets
            and name not in intent_targets
        ]
        truly_dropped = dropped - set(to_restore)

        if truly_dropped:
            print(f"✅ Reflection: {truly_dropped} appear intentionally removed (mentioned in instruction).")

        if not to_restore:
            print("✅ Reflection: All dropped nodes were legitimately removed.")
            return result

        print(f"⚠️  Reflection: Restoring {len(to_restore)} silently dropped node(s): {to_restore}")
        for name in to_restore:
            revised_list.append(orig_lookup[name])

        revised_plan["module_details"] = revised_list
        result["revised_plan"] = revised_plan
        return result

    @traceable(name="quality_reflect", run_type="chain")
    def _quality_reflect(self, result, intent_analysis, user_instruction):
        """
        Post-output reflection: verify the LLM actually addressed every intent component.
        For mixed instructions, the main LLM often executes the loudest intent and silently
        drops the others. This pass checks coverage and attaches a self-critique block
        (and a 'unhandled_intents' marker) so the frontend can surface follow-ups instead
        of silently losing user requests.

        This DOES NOT mutate the plan — it only annotates the result. Triggering an
        automatic re-edit would risk runaway loops; we surface gaps to the user instead.
        """
        if not isinstance(result, dict):
            return result
        if not isinstance(intent_analysis, dict):
            return result

        edits = intent_analysis.get("structural_edits") or []
        explore = intent_analysis.get("exploration_topics") or []
        questions = intent_analysis.get("questions") or []
        if not (edits or explore or questions):
            return result

        output_type = result.get("output_type", "")
        unhandled = []

        # Rule 1: structural_edits requested but output_type is not plan_update.
        if edits and output_type != "plan_update":
            unhandled.append({
                "kind": "structural_edits_skipped",
                "detail": f"User requested {len(edits)} structural edit(s) but output_type='{output_type}' did not mutate the plan.",
                "items": edits,
            })

        # Rule 2: plan_update mode — check each requested edit shows up in modification_intent
        # or in the diff list. Lightweight string match: target name appears in changed entries.
        if edits and output_type == "plan_update":
            mod_intent_text = (result.get("modification_intent") or "").lower()
            revised = result.get("revised_plan", {}) or {}
            mds = revised.get("module_details", [])
            if isinstance(mds, dict):
                mds = list(mds.values())
            changed_names = {
                (m.get("name") or "").lower()
                for m in mds
                if isinstance(m, dict) and m.get("change_type") in ("added", "modified", "ports_only")
            }
            missing = []
            for e in edits:
                target = (e.get("target") or "").strip().lower()
                if not target:
                    continue
                hit = (
                    target in mod_intent_text
                    or any(target in n for n in changed_names if n)
                    or any(n and n in target for n in changed_names)
                )
                if not hit:
                    missing.append(e)
            if missing:
                unhandled.append({
                    "kind": "structural_edits_partially_applied",
                    "detail": f"{len(missing)} structural edit(s) were not visible in the diff or modification_intent.",
                    "items": missing,
                })

        # Rule 3: questions raised but no answer surfaced anywhere user-visible.
        if questions and output_type == "plan_update":
            chat = result.get("chat_response") or {}
            chat_text = (chat.get("text") or "").lower()
            addressed_qs = chat.get("addressed_questions") or []
            answered = set()
            for aq in addressed_qs:
                q_text = (aq.get("question") or "").strip().lower()
                if q_text and aq.get("answer"):
                    answered.add(q_text)
            missing_q = []
            for q in questions:
                q_low = q.strip().lower()
                if q_low in answered:
                    continue
                # Loose fallback: check if any 6+-char fragment shows up in chat_text.
                fragments = [w for w in re.split(r"[\s，。,.;；?？]+", q_low) if len(w) >= 6]
                if fragments and any(f in chat_text for f in fragments):
                    continue
                missing_q.append(q)
            if missing_q:
                unhandled.append({
                    "kind": "questions_unanswered",
                    "detail": f"{len(missing_q)} question(s) raised by the user were not answered in chat_response.",
                    "items": missing_q,
                })

        # Rule 4: exploration topics raised but neither surfaced in chat nor reflected in modification_intent.
        if explore and output_type == "plan_update":
            chat = result.get("chat_response") or {}
            chat_text = (chat.get("text") or "").lower()
            addressed_ex = chat.get("addressed_explorations") or []
            covered_topics = set()
            for ax in addressed_ex:
                t = (ax.get("topic") or "").strip().lower()
                if t and ax.get("response"):
                    covered_topics.add(t)
            mod_intent_text = (result.get("modification_intent") or "").lower()
            missing_ex = []
            for t in explore:
                t_low = t.strip().lower()
                if t_low in covered_topics:
                    continue
                fragments = [w for w in re.split(r"[\s，。,.;；]+", t_low) if len(w) >= 4]
                if fragments and any(f in chat_text or f in mod_intent_text for f in fragments):
                    continue
                missing_ex.append(t)
            if missing_ex:
                unhandled.append({
                    "kind": "exploration_not_surfaced",
                    "detail": f"{len(missing_ex)} exploration topic(s) had no visible response in chat_response or modification_intent.",
                    "items": missing_ex,
                })

        result["intent_analysis"] = intent_analysis
        if unhandled:
            result["unhandled_intents"] = unhandled
            print(f"⚠️  Quality reflection: {len(unhandled)} unhandled intent group(s).")
            for u in unhandled:
                print(f"   - {u['kind']}: {u['detail']}")
        else:
            print("✅ Quality reflection: all intent components addressed.")
        return result

    @traceable(name="llm_critique", run_type="chain")
    def _llm_critique(self, result, intent_analysis, original_plan, user_instruction):
        """
        LLM-based self-critique. Unlike _quality_reflect (rule-based string match),
        this asks a Critic LLM to read (original plan, intent, new plan) and judge:
          - intent_coverage: did each parsed intent component get a meaningful response?
          - structural_validity: are connections still coherent (no orphan refs, no broken flows)?
          - hallucination: did the editor invent modules/tech not asked for?
          - quality_score: 1–5 overall
          - critique: concise points
          - suggested_fixes: what a follow-up edit should change (informational only — we
            do NOT auto-revise here; that would risk runaway latency).
        Attaches output as result['llm_critique']. Failures are non-fatal.
        """
        if not isinstance(result, dict):
            return result
        try:
            # Compact the new plan for the critic to keep tokens reasonable.
            new_plan_view = result.get("revised_plan") if result.get("output_type") == "plan_update" else None
            if new_plan_view is None:
                # Non-mutating modes: critic still evaluates whether the answer addressed the intent.
                new_plan_view = {"answer": result.get("answer"), "output_type": result.get("output_type")}

            def _compact(plan):
                if not isinstance(plan, dict):
                    return plan
                mds = plan.get("module_details", [])
                if isinstance(mds, dict):
                    mds = list(mds.values())
                slim_mds = []
                for m in mds:
                    if not isinstance(m, dict):
                        continue
                    slim_mds.append({
                        "name": m.get("name"),
                        "change_type": m.get("change_type"),
                        "brief_description": (m.get("brief_description") or "")[:120],
                        "ports": m.get("ports"),
                    })
                return {**{k: v for k, v in plan.items() if k != "module_details"}, "module_details": slim_mds}

            sys_prompt = (
                "You are a strict Critic Agent reviewing an architecture editor's output. "
                "Compare the original plan, the user's parsed intent, and the editor's new plan. "
                "Do NOT rewrite the plan. Output a structured judgment ONLY.\n\n"
                "JSON schema (strict):\n"
                "{\n"
                "  \"quality_score\": 1-5 integer (5=perfect, 1=mostly wrong),\n"
                "  \"intent_coverage\": [{\"intent\": \"...\", \"covered\": true|false, \"evidence\": \"...\"}],\n"
                "  \"structural_issues\": [\"orphan ref / broken flow / dangling edge — be specific\"],\n"
                "  \"hallucinations\": [\"fabricated module or tech not asked for\"],\n"
                "  \"critique\": \"<= 3 short sentences, plain Chinese\",\n"
                "  \"suggested_fixes\": [\"actionable items for a follow-up edit\"],\n"
                "  \"needs_revision\": true|false\n"
                "}\n"
                "Be concise. If everything is fine, return high score and empty arrays."
            )
            critic_input = {
                "user_instruction": user_instruction,
                "parsed_intent": intent_analysis,
                "original_plan": _compact(original_plan if isinstance(original_plan, dict) else {}),
                "new_plan_or_answer": _compact(new_plan_view) if isinstance(new_plan_view, dict) and "module_details" in new_plan_view else new_plan_view,
                "modification_intent": result.get("modification_intent"),
                "chat_response": result.get("chat_response"),
                "rule_based_unhandled": result.get("unhandled_intents", []),
            }
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(critic_input, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            critique = json.loads(resp.choices[0].message.content or "{}")
            # Normalize
            critique.setdefault("quality_score", 0)
            critique.setdefault("intent_coverage", [])
            critique.setdefault("structural_issues", [])
            critique.setdefault("hallucinations", [])
            critique.setdefault("suggested_fixes", [])
            critique.setdefault("needs_revision", False)
            result["llm_critique"] = critique
            score = critique.get("quality_score", 0)
            uncovered = sum(1 for c in critique.get("intent_coverage", []) if not c.get("covered"))
            print(
                f"🧪 Critic: score={score}/5 | uncovered={uncovered} | "
                f"struct_issues={len(critique['structural_issues'])} "
                f"halluc={len(critique['hallucinations'])} "
                f"needs_revision={critique['needs_revision']}"
            )
            if critique.get("critique"):
                print(f"   ↳ {critique['critique']}")
        except Exception as exc:
            print(f"[editor] LLM critique failed (non-fatal): {exc}")
            result["llm_critique"] = {"error": str(exc), "needs_revision": False}
        return result
# TODO: add memory for the editor

class ConflictSolver:
    def __init__(self):
        pass

    @staticmethod
    def _normalize(arch_json_str):
        """Parse and normalize module_details to a dict keyed by name."""
        data = json.loads(arch_json_str) if isinstance(arch_json_str, str) else arch_json_str
        md = data.get("module_details", {})
        if isinstance(md, list):
            md = {m.get("name", f"__unknown_{i}"): m for i, m in enumerate(md)}
        return data, md

    @staticmethod
    def _sub_ids(m_info):
        """Return the set of sub-node names declared in a module."""
        return {s.get("name") for s in m_info.get("sub_nodes", []) if s.get("name")}

    @staticmethod
    def _edge_key(src_mod, tgt_mod, src_node, tgt_node):
        """Canonical dedup key for an edge (ignores label)."""
        return (src_mod or "", tgt_mod or "", src_node or "", tgt_node or "")

    def detect_conflicts(self, arch_json_str):
        """
        Inspect the JSON for hard errors and soft warnings.

        Hard errors (is_valid=False):
          - Dangling edge: invalid module reference in incoming/outgoing
          - Invalid sub-node reference in internal_flow, connects_to_internal_node,
            originates_from_internal_node, or target_node
          - Truly isolated module: no connections AND no meaningful content

        Soft warnings (printed only, do not affect is_valid):
          - Sub-node not referenced by any port
          - Duplicate edges
        """
        try:
            data, module_details = self._normalize(arch_json_str)
        except Exception as e:
            return False, f"CRITICAL: Invalid JSON format. ({e})"

        if not module_details:
            return False, "CRITICAL: 'module_details' field is missing or empty."

        hard_issues = []
        valid_mod_ids = set(module_details.keys())
        connected_modules = set()

        for m_id, m_info in module_details.items():
            ports = m_info.get("ports", {})
            sub_ids = self._sub_ids(m_info)

            # ── Incoming ────────────────────────────────────────────────────
            for edge in ports.get("incoming", []):
                src = edge.get("source_mod")
                if src and src not in valid_mod_ids:
                    hard_issues.append(
                        f"- Dangling incoming edge: Module '{m_id}' references undefined source '{src}'."
                    )
                else:
                    connected_modules.update(filter(None, [m_id, src]))
                cti = edge.get("connects_to_internal_node")
                if cti and cti not in sub_ids:
                    hard_issues.append(
                        f"- Invalid sub-node ref: 'connects_to_internal_node'='{cti}' in '{m_id}' "
                        f"not found in its sub_nodes."
                    )

            # ── Outgoing ────────────────────────────────────────────────────
            for edge in ports.get("outgoing", []):
                tgt = edge.get("target_mod")
                if tgt and tgt not in valid_mod_ids:
                    hard_issues.append(
                        f"- Dangling outgoing edge: Module '{m_id}' references undefined target '{tgt}'."
                    )
                else:
                    connected_modules.update(filter(None, [m_id, tgt]))
                ofin = edge.get("originates_from_internal_node")
                if ofin and ofin not in sub_ids:
                    hard_issues.append(
                        f"- Invalid sub-node ref: 'originates_from_internal_node'='{ofin}' in '{m_id}' "
                        f"not found in its sub_nodes."
                    )
                tgt_node = edge.get("target_node")
                if tgt_node and tgt in valid_mod_ids:
                    tgt_subs = self._sub_ids(module_details[tgt])
                    if tgt_subs and tgt_node not in tgt_subs:
                        hard_issues.append(
                            f"- Invalid sub-node ref: 'target_node'='{tgt_node}' in outgoing of '{m_id}' "
                            f"not found in target module '{tgt}'."
                        )

            # ── Internal flow ───────────────────────────────────────────────
            for flow in ports.get("internal_flow", []):
                for field in ("source_node", "target_node"):
                    node_ref = flow.get(field)
                    if node_ref and node_ref not in sub_ids:
                        hard_issues.append(
                            f"- Invalid internal_flow ref: '{field}'='{node_ref}' in '{m_id}' "
                            f"not found in its sub_nodes."
                        )

        # ── Isolated modules ────────────────────────────────────────────────
        if len(valid_mod_ids) > 1:
            for m_id, m_info in module_details.items():
                if m_id not in connected_modules:
                    has_content = bool(
                        m_info.get("function") or
                        m_info.get("brief_description") or
                        m_info.get("sub_nodes")
                    )
                    if has_content:
                        print(f"  ⚠️  Soft warning: '{m_id}' has no connections but has content — needs review.")
                    else:
                        hard_issues.append(
                            f"- Isolated empty module: '{m_id}' has no connections and no meaningful content."
                        )

        # ── Soft: unreferenced sub-nodes (warn only) ────────────────────────
        for m_id, m_info in module_details.items():
            sub_ids = self._sub_ids(m_info)
            if not sub_ids:
                continue
            ports = m_info.get("ports", {})
            referenced = set()
            for edge in ports.get("incoming", []):
                if edge.get("connects_to_internal_node"):
                    referenced.add(edge["connects_to_internal_node"])
            for edge in ports.get("outgoing", []):
                if edge.get("originates_from_internal_node"):
                    referenced.add(edge["originates_from_internal_node"])
            for flow in ports.get("internal_flow", []):
                referenced.update(filter(None, [flow.get("source_node"), flow.get("target_node")]))
            for sub_name in sub_ids - referenced:
                print(f"  ⚠️  Soft warning: sub-node '{sub_name}' in '{m_id}' not referenced by any port.")

        if not hard_issues:
            return True, "No conflicts found."
        return False, "\n".join(hard_issues)

    def resolve(self, arch_json_str):
        """
        Hard-clean the architecture JSON:
          1. Remove dangling edges (invalid module refs).
          2. Clear invalid sub-node refs in connects_to / originates_from /
             target_node / internal_flow (set to None rather than crashing).
          3. Deduplicate exact-duplicate edges (same src_mod, tgt_mod,
             src_node, tgt_node — ignoring label).
          4. Remove modules that are both unconnected AND content-empty.
          5. Output module_details as a list.
        """
        try:
            data, module_details = self._normalize(arch_json_str)
        except Exception:
            print("❌ ConflictSolver: Invalid JSON string.")
            return arch_json_str

        if not module_details:
            return json.dumps(arch_json_str if isinstance(arch_json_str, dict) else {})

        print("🕵️ ConflictSolver: Resolving topology...")

        # Clean malformed sub_nodes (remove 'ports' field if present)
        for m_info in module_details.values():
            sub_nodes = m_info.get("sub_nodes", [])
            for sub in sub_nodes:
                if "ports" in sub:
                    del sub["ports"]
                    print(f"  Cleaned invalid 'ports' field from sub_node '{sub.get('name')}'")

        valid_mod_ids = set(module_details.keys())
        sub_id_map = {m_id: self._sub_ids(m_info) for m_id, m_info in module_details.items()}
        connected_modules = set()
        removed_edges = 0
        deduped = 0

        for m_id, m_info in module_details.items():
            ports = m_info.get("ports", {})
            sub_ids = sub_id_map[m_id]

            # ── Incoming ────────────────────────────────────────────────────
            new_incoming, seen = [], set()
            for edge in ports.get("incoming", []):
                src = edge.get("source_mod")
                if src and src not in valid_mod_ids:
                    print(f"  Removed dangling incoming: '{src}' → '{m_id}'")
                    removed_edges += 1
                    continue
                cti = edge.get("connects_to_internal_node")
                if cti and cti not in sub_ids:
                    print(f"  Cleared invalid connects_to_internal_node='{cti}' in '{m_id}'")
                    edge = {**edge, "connects_to_internal_node": None}
                key = self._edge_key(src, m_id, edge.get("source_node"), edge.get("connects_to_internal_node"))
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                new_incoming.append(edge)
                connected_modules.update(filter(None, [m_id, src]))

            # ── Outgoing ────────────────────────────────────────────────────
            new_outgoing, seen = [], set()
            for edge in ports.get("outgoing", []):
                tgt = edge.get("target_mod")
                if tgt and tgt not in valid_mod_ids:
                    print(f"  Removed dangling outgoing: '{m_id}' → '{tgt}'")
                    removed_edges += 1
                    continue
                ofin = edge.get("originates_from_internal_node")
                if ofin and ofin not in sub_ids:
                    print(f"  Cleared invalid originates_from_internal_node='{ofin}' in '{m_id}'")
                    edge = {**edge, "originates_from_internal_node": None}
                tgt_node = edge.get("target_node")
                if tgt_node and tgt in valid_mod_ids:
                    tgt_subs = sub_id_map.get(tgt, set())
                    if tgt_subs and tgt_node not in tgt_subs:
                        print(f"  Cleared invalid target_node='{tgt_node}' in outgoing of '{m_id}'")
                        edge = {**edge, "target_node": None}
                key = self._edge_key(m_id, tgt, edge.get("originates_from_internal_node"), edge.get("target_node"))
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                new_outgoing.append(edge)
                connected_modules.update(filter(None, [m_id, tgt]))

            # ── Internal flow ───────────────────────────────────────────────
            new_internal, seen = [], set()
            for flow in ports.get("internal_flow", []):
                src_n, tgt_n = flow.get("source_node"), flow.get("target_node")
                if (src_n and src_n not in sub_ids) or (tgt_n and tgt_n not in sub_ids):
                    print(f"  Removed invalid internal_flow '{src_n}→{tgt_n}' in '{m_id}'")
                    removed_edges += 1
                    continue
                key = (src_n or "", tgt_n or "")
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                new_internal.append(flow)

            m_info["ports"] = {
                "incoming": new_incoming,
                "outgoing": new_outgoing,
                "internal_flow": new_internal,
            }

        # ── Remove unconnected + content-empty modules ──────────────────────
        final_modules = []
        removed_mods = 0
        for m_id, m_info in module_details.items():
            if m_id not in connected_modules:
                has_content = bool(
                    m_info.get("function") or
                    m_info.get("brief_description") or
                    m_info.get("sub_nodes")
                )
                if not has_content:
                    print(f"  Removed isolated empty module: '{m_id}'")
                    removed_mods += 1
                    continue
                print(f"  Kept isolated module '{m_id}' (has content — review recommended)")
            final_modules.append(m_info)

        data["module_details"] = final_modules
        print(f"✅ ConflictSolver: removed {removed_edges} dangling edges, "
              f"deduped {deduped} duplicate edges, removed {removed_mods} empty isolated modules.")
        return json.dumps(data, ensure_ascii=False)
    

if __name__ == "__main__":
    vector_store = get_vector_store()
    graph_store = get_graph_store()
    planner = PlannerAgent(get_client(), graph_store)
    user_query = "Design an LLM-based code education system for children"
    base_anchors = []
    divergent_plans = planner.divergent_planning(
            user_query=user_query,
            base_anchors=base_anchors,
            num_outputs=1
    )
    print(divergent_plans)
