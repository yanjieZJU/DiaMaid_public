import sys
import os
import json

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from server.agents import (
    PlannerAgent, 
    RetrieverAgent, 
    IntegratorAgent,  
    ConflictSolver,
    EditorAgent,
    client
)
from server.retriever_neo4j import get_global_stores
from server.session_manager import create_initial_plan, create_modified_plan, get_session_tree, get_plan_lineage

def run_pipeline(user_query, base_anchors, user_relations, num_outputs=1, divergence=True, session_id=None):
    vector_store, graph_store = get_global_stores()
    planner = PlannerAgent(client, graph_store)
    retriever = RetrieverAgent(vector_store, graph_store, client)
    integrator = IntegratorAgent(client)
    solver = ConflictSolver() # Initialize Solver
    
    yield json.dumps({"type": "progress", "message": "Searching for structural paths...", "percent": 15}) + "\n"
    structural_hints = retriever.retrieve_structure_paths(user_relations)
    
    # Print structural hints once
    if structural_hints:
        print(integrator.format_hints(structural_hints))

    yield json.dumps({"type": "progress", "message": "Planning divergent perspectives...", "percent": 25}) + "\n"
    # === Pre-calculate Divergent Plans for ALL Options ===
    if divergence:
        divergent_plans = planner.divergent_planning(
            user_query=user_query,
            base_anchors=base_anchors,
            num_outputs=num_outputs
        )
    else:
        divergent_plans = None

    # === Step 1: Global Retrieval for Base Anchors (Higher Recall) ===
    print(f"\n🎯 Retrieving Base Anchors (Top-6 for richer context)...")
    yield json.dumps({"type": "progress", "message": "Extracting domain knowledge and base nodes...", "percent": 35}) + "\n"
    base_nodes_map = retriever.retrieve_for_module(base_anchors, top_k=6, include_topology=False)
    
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
                # Divergent anchors are new ideas, their topology is valuable context
                additional_map = retriever.retrieve_for_module(additional_anchors, top_k=3, include_topology=True)
                for anchor_name, retrieved_list in additional_map.items():
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
        
        # === SELF-CORRECTION LOOP ===
        yield json.dumps({"type": "progress", "message": f"Detecting and resolving conflicts for option {i+1}...", "percent": int(base_percent + 15)}) + "\n"
        max_retries = 2
        for attempt in range(max_retries + 1):
            if not current_arch_json: break
            
            is_valid, report = solver.detect_conflicts(current_arch_json)
            if is_valid:
                print(f"✅ Architecture passed integrity check (Attempt {attempt}).")
                break
            else:
                print(f"⚠️ Conflict detected (Attempt {attempt}):\n{report}")
                if attempt < max_retries:
                    # Request Fix
                    current_arch_json = integrator.refine_arch(full_query, current_arch_json, report)
                else:
                    print("❌ Max retries reached. Proceeding with best-effort cleanup.")

        
        if current_arch_json:
            # === FINAL SAFEGUARD ===
            # Resolve leftover conflicts (hard cleanup) before drawing
            validated_arch_json_str = solver.resolve(current_arch_json)
            
            # Skip mermaid generation to save extreme time since frontend natively renders the JSON now
            mermaid_code = "graph TD\n  A[Result rendered directly from JSON using ReactFlow] --> B[No Mermaid needed!]"
            
            # Add perspective info to description
            try:
                arch_json = json.loads(validated_arch_json_str)
                module_details = arch_json.get("module_details", {})
                raw_desc = arch_json.get("overall_description", "")
                
                # Collect References
                ref_data = integrator.collect_references(arch_json, current_ref_info)
                used_ref = ref_data.get("used", {})
                explorable_ref = ref_data.get("explorable", [])
                
                # Attach reference details directly to module_details
                if isinstance(module_details, list):
                    for details in module_details:
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
                        
            except:
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

def run_edit_pipeline(plan_id, current_plan_json, user_instruction, original_query, session_id=None):
    # Initialize agents
    integrator = IntegratorAgent(client)
    solver = ConflictSolver()
    editor = EditorAgent(client)
    
    # Initialize retriever for tool use
    vector_store, graph_store = get_global_stores()
    retriever = RetrieverAgent(vector_store, graph_store, client)

    if plan_id:
        print(f"User requested modification on parent node: {plan_id}")

    # Get Lineage History
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

    yield json.dumps({"type": "progress", "message": "Analyzing and modifying...", "percent": 20}) + "\n"
    
    # Call Editor Agent (One-shot with tools)
    edit_result = editor.edit_plan(current_plan_json, user_instruction, history_summary, retriever)
    
    # Handle cases where edit_result might not be a dictionary (e.g., if JSON decoding failed inside editor)
    if not isinstance(edit_result, dict) or not edit_result.get("revised_plan"):
        yield json.dumps({"type": "error", "message": "Failed to generate modified plan structure."}) + "\n"
        return

    modified_arch_json = edit_result.get("revised_plan")
    modification_intent = edit_result.get("modification_intent", "User modified plan")
    thought_process = edit_result.get("thought", "")
    
    print(f"✅ Editor Intent: {modification_intent}")
    
    # 2. Conflict Check & Refine Loop
    # Ensure we work with JSON string for consistency in prompt injection if needed, 
    # but solver handles dicts. Integrator expects string representation in prompt.
    current_arch_json = modified_arch_json
    
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        if not current_arch_json: break

        yield json.dumps({"type": "progress", "message": f"Verifying integrity (Attempt {attempt+1})...", "percent": 50 + (attempt * 10)}) + "\n"
        
        is_valid, report = solver.detect_conflicts(current_arch_json)
        if is_valid:
            print(f"✅ Modified architecture passed integrity check.")
            break
        else:
            print(f"⚠️ Conflict detected in modified arch:\n{report}")
            if attempt < max_retries:
                # Request Fix
                # Convert dict to json string for the prompt
                arch_str = json.dumps(current_arch_json, indent=2) if isinstance(current_arch_json, dict) else current_arch_json
                current_arch_json = integrator.refine_arch(original_query, arch_str, report)
                # refine_arch returns a string
            else:
                print("❌ Max retries reached for modification.")

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
                    if "reference" not in details:
                        details["reference"] = None
            else:
                for mod_name, details in module_details.items():
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
                        modified_arch_json=arch_json
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
             
        except Exception as e:
             yield json.dumps({"type": "error", "message": f"Failed to parse modified plan: {str(e)}"}) + "\n"
    else:
        yield json.dumps({"type": "error", "message": "Failed to modify architecture"}) + "\n"

    yield json.dumps({"type": "progress", "message": "Modification complete!", "percent": 100}) + "\n"

if __name__ == "__main__":
    query = "design a system for coding education towards children"
    # Example usage
    run_pipeline(query, base_anchors=[], user_relations=[], num_outputs=1)
