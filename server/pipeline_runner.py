import sys
import os
import json
import concurrent.futures
from langsmith import tracing_context

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from config import *
from agents import (
    PlannerAgent,
    RetrieverAgent,
    IntegratorAgent,
    ConflictSolver,
    EditorAgent,
    get_client,
)
from retriever_neo4j import get_global_stores
from session_manager import create_initial_plan, create_modified_plan, get_session_tree, get_plan_lineage
import editor_memory

# === LangSmith startup diagnostic (one-shot) ===
# If TRACING != 'true' here, every @traceable in this process is a no-op for the
# rest of its lifetime — restart the server after editing config.py.
print(
    f"🔭 LangSmith @ pipeline_runner import: "
    f"TRACING={os.environ.get('LANGSMITH_TRACING', '<UNSET>')}, "
    f"PROJECT={os.environ.get('LANGSMITH_PROJECT', '<UNSET>')}, "
    f"KEY={'set' if os.environ.get('LANGSMITH_API_KEY') else '<UNSET>'}"
)

def run_pipeline(user_query, base_anchors, user_relations, num_outputs=1, divergence=True, session_id=None):
    vector_store, graph_store = get_global_stores()
    planner = PlannerAgent(get_client(), graph_store)
    retriever = RetrieverAgent(vector_store, graph_store, get_client())
    integrator = IntegratorAgent(get_client())
    solver = ConflictSolver() # Initialize Solver
    
    yield json.dumps({"type": "progress", "message": "Searching knowledge base...", "percent": 15}) + "\n"

    # === Parallelize ALL independent retrieval tasks upfront ===
    # structural paths, divergent planning, AND base anchor retrieval have no mutual dependency
    print(f"\n🎯 Launching parallel retrieval: structure paths + divergent planning + base anchors...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _ex:
        _fut_hints = _ex.submit(retriever.retrieve_structure_paths, user_relations)
        _fut_plans = _ex.submit(
            planner.divergent_planning, user_query, base_anchors, num_outputs
        ) if divergence else None
        _fut_base  = _ex.submit(retriever.retrieve_for_module, base_anchors, 6, False)
        structural_hints = _fut_hints.result()
        divergent_plans  = _fut_plans.result() if _fut_plans else None
        base_nodes_map   = _fut_base.result()

    if structural_hints:
        print(integrator.format_hints(structural_hints))

    # === KG health check — surface silent degradation to the user ===
    # If divergent planning AND base-anchor retrieval both came back empty when we
    # actually expected something, the embedding API is likely down. Tell the user
    # so they don't get a "fine-looking but ungrounded" plan and wonder why.
    expected_planning = bool(divergence)
    expected_retrieval = bool(base_anchors)
    planning_failed = expected_planning and not divergent_plans
    retrieval_failed = expected_retrieval and not base_nodes_map
    if planning_failed or retrieval_failed:
        warn_parts = []
        if planning_failed:
            warn_parts.append("发散规划")
        if retrieval_failed:
            warn_parts.append("基础锚点检索")
        warn_msg = (
            f"⚠️ 知识库检索未返回结果（{', '.join(warn_parts)}）。"
            "可能是 embedding 接口不可用，本次方案将完全由 LLM 自主生成，"
            "缺少文献支撑，请谨慎对待创新性与可行性。"
        )
        print(warn_msg, flush=True)
        yield json.dumps({"type": "warning", "message": warn_msg}) + "\n"

    # Emit divergent perspectives so frontend can display them immediately
    if divergent_plans:
        perspectives_data = [
            {
                "perspective": p.get("perspective", ""),
                "rationale": p.get("rationale", ""),
                "additional_anchors": [
                    {"user_term": a.get("user_term", "")}
                    for a in p.get("additional_anchors", [])
                ]
            }
            for p in divergent_plans
        ]
        yield json.dumps({"type": "divergent_perspectives", "perspectives": perspectives_data}) + "\n"

    yield json.dumps({"type": "progress", "message": "Extracting domain knowledge and base nodes...", "percent": 35}) + "\n"
    print(f"\n🎯 Base anchor retrieval complete ({len(base_nodes_map)} anchors).")
    
    # Process base nodes into standard format
    base_ref_info = {}
    for anchor_name, retrieved_list in base_nodes_map.items():
        anchor_nodes = []
        for node_with_score in retrieved_list:
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

    # === Pre-fetch additional anchors for ALL divergent plans in parallel ===
    # This avoids a serial retrieval bottleneck inside the generation loop
    additional_maps = {}  # index -> {anchor_name: [nodes]}
    if divergent_plans:
        def _fetch_additional(idx_plan):
            idx, plan = idx_plan
            anchors = plan.get("additional_anchors", [])
            if not anchors:
                return idx, {}
            raw = retriever.retrieve_for_module(anchors, top_k=3, include_topology=True)
            result = {}
            for anchor_name, retrieved_list in raw.items():
                result[anchor_name] = []
                for node_with_score in retrieved_list:
                    node = node_with_score.node
                    result[anchor_name].append({
                        'id': node.id_,
                        'text': node.text,
                        'score': node_with_score.score,
                        'structural_path': node.metadata.get('topology', None),
                        'metadata': dict(node.metadata)
                    })
            return idx, result

        plans_to_fetch = [(i, p) for i, p in enumerate(divergent_plans)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(plans_to_fetch) or 1) as _ex:
            for idx, result in _ex.map(_fetch_additional, plans_to_fetch):
                additional_maps[idx] = result
        print(f"✅ Pre-fetched additional anchors for {len(additional_maps)} divergent plans.")

    # === Generate Options ===
    target_iterations = num_outputs if divergence else 1

    for i in range(target_iterations):

        current_ref_info = base_ref_info.copy()

        d_plan = None
        # Apply Divergent Plan (if available)
        if divergent_plans and i < len(divergent_plans):
            d_plan = divergent_plans[i]
            print(f"\n{'='*60}")
            print(f"{d_plan}\n")

        if d_plan:
            additional_anchors = d_plan.get("additional_anchors", [])
            perspective = d_plan.get("perspective", "Enhanced")
            rationale = d_plan.get("rationale", "")

            print(f"Applying Divergent Perspective: {perspective}")
            print(f"Rationale: {rationale}")

            additional_desc = f"**Perspective**: {perspective}\n**Rationale**: {rationale}\n\n"

            if additional_anchors:
                # Use pre-fetched results
                for anchor_name, nodes in additional_maps.get(i, {}).items():
                    current_ref_info[anchor_name] = nodes
            else:
                print(f"No additional anchors suggested for this plan")
        else:
            # Fallback / Standard mode
            additional_anchors = []
            perspective = "Standard"
            additional_desc = ""
            print(f"\n{'='*60}")
            print(f"Generating Standard Plan (Option {i+1})\n")
        
        print(f"\n📦 Retrieved modules summary for Option {i+1}:")
        print(f"Base Anchors: {len(base_ref_info)} keys")
        print(f"Total Anchors: {len(current_ref_info)} keys")
        
        # Integrate query context
        full_query = user_query
        if d_plan:
             full_query += "; Perspective: "+d_plan.get("perspective", "")+"; Rationale: "+d_plan.get("rationale", "")

        # Generate architecture (Initial)
        base_percent = 35 + (60 / target_iterations) * i
        yield json.dumps({"type": "progress", "message": f"Generating architecture for option {i+1}...", "percent": int(base_percent + 5)}) + "\n"
        current_arch_json = integrator.integrate_arch(
            query=full_query, 
            retrieved_info=current_ref_info, 
            structural_hints=structural_hints,
            divergent_plan=d_plan,
            base_anchors=base_anchors
        )
        
        # === SINGLE-PASS VALIDATION ===
        yield json.dumps({"type": "progress", "message": f"Validating option {i+1}...", "percent": int(base_percent + 15)}) + "\n"

        is_valid, report = solver.detect_conflicts(current_arch_json)
        if not is_valid:
            print(f"⚠️ Conflicts detected (Attempt 0):\n{report}")
            # Single LLM refinement attempt
            current_arch_json = integrator.refine_arch(full_query, current_arch_json, report)
            is_valid, report = solver.detect_conflicts(current_arch_json)
            if is_valid:
                print(f"✅ Architecture passed after refinement.")
            else:
                print(f"⚠️ Still has conflicts, will apply automatic cleanup.")
        else:
            print(f"✅ Architecture passed integrity check.")

        if current_arch_json:
            # Apply automatic cleanup for any remaining issues
            validated_arch_json_str = solver.resolve(current_arch_json)
            
            # Skip mermaid generation to save extreme time since frontend natively renders the JSON now
            mermaid_code = "graph TD\n  A[Result rendered directly from JSON using ReactFlow] --> B[No Mermaid needed!]"
            
            # Add perspective info to description
            try:
                arch_json = json.loads(validated_arch_json_str) if isinstance(validated_arch_json_str, str) else validated_arch_json_str
                if not isinstance(arch_json, dict):
                    raise ValueError(f"Expected dict from arch JSON, got {type(arch_json)}")
                module_details = arch_json.get("module_details", {})
                raw_desc = arch_json.get("overall_description", "")
                
                # Collect References
                ref_data = integrator.collect_references(arch_json, current_ref_info)
                used_ref = ref_data.get("used", {})
                explorable_ref = ref_data.get("explorable", [])
                
                # Attach reference details directly to module_details
                if isinstance(module_details, list):
                    for details in module_details:
                        if details.get("brief_description"):
                            details["function"] = details["brief_description"]
                        mod_name = details.get("name", "Unknown")
                        if mod_name in used_ref and used_ref[mod_name]:
                            first_ref = used_ref[mod_name][0]
                            details["reference"] = {
                                "name": first_ref.get("name", "Unknown"),
                                "technical_details": first_ref.get("text", ""),
                                "source": first_ref.get("source", "")
                            }
                        else:
                            details["reference"] = None
                else:
                    for mod_name, details in module_details.items():
                        if mod_name in used_ref and used_ref[mod_name]:
                            # Pass the first used reference object associated with the module
                            first_ref = used_ref[mod_name][0]
                            details["reference"] = {
                                "name": first_ref.get("name", "Unknown"),
                                "technical_details": first_ref.get("text", ""),
                                "source": first_ref.get("source", "")
                            }
                        else:
                            details["reference"] = None
                        
            except Exception as _parse_err:
                import traceback
                print(f"❌ pipeline parse error: {_parse_err}")
                traceback.print_exc()
                module_details = {}
                raw_desc = ""
                used_ref = {}
                explorable_ref = []

            final_desc = raw_desc

            # === Save to Session Tree ===
            try:
                plan_id = create_initial_plan(
                    session_id=session_id,
                    original_query=full_query,
                    perspective=perspective,
                    description=final_desc,
                    arch_json=arch_json
                )
                print(f"✅ Plan node saved to Session Tree with ID: {plan_id}")
            except Exception as e:
                print(f"❌ Failed to save plan to Session Tree: {e}")
                plan_id = "error_saving_db"
                
            plan_data = {
                "plan_id": plan_id,
                "option_id": i + 1,
                "perspective": perspective,
                "mermaid_code": mermaid_code,
                "description": final_desc,
                "node_details": module_details,
                "used_references": used_ref,
                "explorable_references": explorable_ref
            }
            yield json.dumps({"type": "plan", "plan_data": plan_data}) + "\n"
            print(f"\n--- Mermaid Code (Option {i+1}) ---\n{mermaid_code}\n")
        else:
            print(f"❌ Failed to generate architecture for option {i+1}")
            yield json.dumps({"type": "error", "message": f"Failed to generate architecture for option {i+1}"}) + "\n"

    yield json.dumps({"type": "progress", "message": "所有方案生成完毕！", "percent": 100}) + "\n"

def run_edit_pipeline(plan_id, current_plan_json, user_instruction, original_query, session_id=None, referenced_modules=None, focused_modules=None):
    # Initialize agents
    integrator = IntegratorAgent(get_client())
    solver = ConflictSolver()
    editor = EditorAgent(get_client())

    # Initialize retriever for tool use
    vector_store, graph_store = get_global_stores()
    retriever = RetrieverAgent(vector_store, graph_store, get_client())

    if plan_id:
        print(f"User requested modification on parent node: {plan_id}")

    # Get Lineage History (kept as a fallback bootstrap source for memory)
    history_summary = []
    if session_id and plan_id:
        try:
            lineage = get_plan_lineage(session_id, plan_id)
            for i, node in enumerate(lineage):
                role_desc = "Initial Plan" if i == 0 else "Modification"

                # Extract key info
                instruction = node.get("user_instruction", "None (Base Plan)")
                desc = node.get("description", "")
                perspective = node.get("perspective", "")

                summary_entry = {
                    "step": i,
                    "type": role_desc,
                    "perspective": perspective,
                    "user_instruction": instruction,
                    "result_description": desc
                }
                history_summary.append(summary_entry)

            print(f"📜 Constructed history summary with {len(history_summary)} steps.")
        except Exception as e:
            print(f"Warning: Could not load lineage for plan {plan_id}: {e}")

    # Load tiered editor memory (branch-aware via parent-chain walk).
    # Falls back to bootstrapping from history_summary if no ancestor has a snapshot yet.
    prior_memory = editor_memory.load_memory(session_id, plan_id)
    if not prior_memory.get("recent_actions") and not prior_memory.get("running_summary") and history_summary:
        prior_memory = editor_memory.memory_from_history_summary(history_summary)
        print(f"📜 Bootstrapped editor memory from legacy history ({len(prior_memory.get('recent_actions') or [])} entries).")
    else:
        print(
            f"📜 Loaded editor memory: turn={prior_memory.get('turn_index', 0)}, "
            f"recent={len(prior_memory.get('recent_actions') or [])}, "
            f"decisions={len(prior_memory.get('key_decisions') or [])}, "
            f"open_q={len(prior_memory.get('open_questions') or [])}."
        )

    yield json.dumps({"type": "progress", "message": "Analyzing and modifying...", "percent": 20}) + "\n"

    # Call Editor Agent (One-shot with tools)
    with tracing_context(metadata={"session_id": session_id, "plan_id": plan_id}):
        edit_result = editor.edit_plan(current_plan_json, user_instruction, history_summary, retriever, referenced_modules=referenced_modules, memory=prior_memory, focused_modules=focused_modules)
    
    # Handle cases where edit_result might not be a dictionary (e.g., if JSON decoding failed inside editor)
    if not isinstance(edit_result, dict):
        yield json.dumps({"type": "error", "message": "Failed to generate valid editor output."}) + "\n"
        return

    output_type = edit_result.get("output_type")
    modified_arch_json = edit_result.get("revised_plan")
    modification_intent = edit_result.get("modification_intent", "User modified plan")
    answer = edit_result.get("answer", {}) if isinstance(edit_result.get("answer"), dict) else {}
    chat_response = edit_result.get("chat_response", {}) if isinstance(edit_result.get("chat_response"), dict) else {}
    intent_analysis = edit_result.get("intent_analysis") or {}
    unhandled_intents = edit_result.get("unhandled_intents") or []
    llm_critique = {}
    has_plan = bool(modified_arch_json)
    has_answer = bool(answer.get("text") or answer.get("citations"))
    has_chat = bool(chat_response.get("text") or chat_response.get("addressed_explorations") or chat_response.get("addressed_questions"))
    allowed_modes = {"plan_update", "direct_answer", "evidence_answer", "plan_explain"}

    if output_type not in allowed_modes:
        output_type = "plan_update" if has_plan else ("direct_answer" if has_answer else "plan_update")

    if has_plan:
        output_type = "plan_update"
    elif has_answer:
        output_type = output_type if output_type in {"direct_answer", "evidence_answer", "plan_explain"} else "direct_answer"

    if output_type in {"direct_answer", "evidence_answer", "plan_explain"}:
        message = answer.get("text", "")
        citations = answer.get("citations", [])
        yield json.dumps({
            "type": "assistant",
            "mode": output_type,
            "message": message,
            "citations": citations,
            "intent_analysis": intent_analysis,
            "unhandled_intents": unhandled_intents,
        }) + "\n"
        # Answer is on screen — run the deferred ~48s critique and push it as a follow-up event.
        try:
            with tracing_context(metadata={"session_id": session_id, "plan_id": plan_id}):
                critique = editor.critique_plan(edit_result, current_plan_json, user_instruction)
            yield json.dumps({"type": "critique", "plan_id": plan_id, "llm_critique": critique}) + "\n"
        except Exception as e:
            print(f"[edit] deferred critique failed (non-fatal): {e}")
        yield json.dumps({"type": "progress", "message": "Modification complete!", "percent": 100}) + "\n"
        return

    if not has_plan:
        yield json.dumps({"type": "error", "message": "Failed to generate modified plan structure."}) + "\n"
        return

    print(f"✅ Editor Intent: {modification_intent}")

    # Surface chat_response BEFORE plan event so the chat bubble appears alongside the diagram update.
    # Mixed-intent edits (delete X + ask about Y + explore Z) need the user-facing reply rendered too.
    if has_chat:
        yield json.dumps({
            "type": "assistant",
            "mode": "plan_update_chat",
            "message": chat_response.get("text", ""),
            "citations": chat_response.get("citations", []),
            "addressed_explorations": chat_response.get("addressed_explorations", []),
            "addressed_questions": chat_response.get("addressed_questions", []),
            "intent_analysis": intent_analysis,
            "unhandled_intents": unhandled_intents,
        }) + "\n"

    # 2. Quick validation (skip slow LLM refinement loop)
    current_arch_json = modified_arch_json
    yield json.dumps({"type": "progress", "message": "Validating architecture...", "percent": 60}) + "\n"

    is_valid, report = solver.detect_conflicts(current_arch_json)
    if not is_valid:
        print(f"⚠️ Conflicts detected, will apply automatic cleanup:\n{report}")
    else:
        print(f"✅ Modified architecture passed integrity check.")

    # 3. Finalize
    if current_arch_json:
        # Resolve leftover conflicts (hard cleanup)
        validated_arch_json_str = solver.resolve(current_arch_json)
        
        try:
            if isinstance(validated_arch_json_str, str):
                arch_json = json.loads(validated_arch_json_str)
            else:
                arch_json = validated_arch_json_str
            
            # Extract module details
            module_details = arch_json.get("module_details", {})
            # Ensure description is the overall description from the plan
            final_desc = arch_json.get("overall_description", "Updated Architecture")

            # Attach basic reference info if missing (Frontend might expect it)
            if isinstance(module_details, list):
                for details in module_details:
                    # Strip internal diff marker before sending to frontend
                    details.pop("change_type", None)
                    if details.get("brief_description"):
                        details["function"] = details["brief_description"]
                    if "reference" not in details:
                        details["reference"] = None
            else:
                for mod_name, details in module_details.items():
                    details.pop("change_type", None)
                    if details.get("brief_description"):
                        details["function"] = details["brief_description"]
                    if "reference" not in details:
                        details["reference"] = None
            
            # Update Session Tree
            # We don't overwrite, we create a new node
            new_plan_id = plan_id # Fallback if we fail to create one
            derived_perspective = "User Modified"
            if plan_id:
                try:
                    # Include modification intent in the session tree node
                    new_plan_id = create_modified_plan(
                        session_id=session_id,
                        parent_node_id=plan_id,
                        user_instruction=f"{user_instruction} (Intent: {modification_intent})",
                        original_query=original_query,
                        modified_description=final_desc,
                        modified_arch_json=arch_json,
                        editor_memory=edit_result.get("editor_memory")
                    )
                    from server.session_manager import _load_session
                    sess = _load_session(session_id if session_id else "anonymous_session")
                    if new_plan_id in sess["nodes"]:
                        derived_perspective = sess["nodes"][new_plan_id].get("perspective", "User Modified")
                    print(f"✅ Modified plan saved to Session Tree as new node: {new_plan_id}")
                except Exception as e:
                    print(f"Failed to save modified plan in Session Tree: {e}")

            # Plan Data Structure mapping to new ID!
            plan_data = {
                "plan_id": new_plan_id,
                "option_id": "edited", 
                "perspective": derived_perspective,
                "modification_intent": modification_intent,
                "mermaid_code": "graph TD\n  A[Result rendered directly from JSON using ReactFlow] --> B[No Mermaid needed!]",
                "description": final_desc,
                "node_details": module_details,
                "used_references": {}, # References are not re-calculated here for now
                "explorable_references": []
            }
            
            yield json.dumps({"type": "plan", "plan_data": plan_data}) + "\n"

            # Plan is on screen now — run the deferred ~48s critique and push it
            # as a follow-up event so the diagram appeared without waiting for it.
            try:
                with tracing_context(metadata={"session_id": session_id, "plan_id": new_plan_id}):
                    critique = editor.critique_plan(edit_result, current_plan_json, user_instruction)
                yield json.dumps({
                    "type": "critique",
                    "plan_id": new_plan_id,
                    "llm_critique": critique,
                }) + "\n"
            except Exception as e:
                print(f"[edit] deferred critique failed (non-fatal): {e}")

        except Exception as e:
             yield json.dumps({"type": "error", "message": f"Failed to parse modified plan: {str(e)}"}) + "\n"
    else:
        yield json.dumps({"type": "error", "message": "Failed to modify architecture"}) + "\n"

    yield json.dumps({"type": "progress", "message": "Modification complete!", "percent": 100}) + "\n"

if __name__ == "__main__":
    query = "design a system for coding education towards children"
    # Example usage
    run_pipeline(query, base_anchors=[], user_relations=[], num_outputs=1)
