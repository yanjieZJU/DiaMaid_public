import os
import re
import httpx
import json
from openai import OpenAI
from retriever_neo4j import load_stores, retrieve_and_get_content, find_structural_path

http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(
    api_key="sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7",
    base_url="https://poloai.top/v1",
    http_client=http_client
)

STANDARD_IO_MAP = {
    "text",
    "image",
    "video",
    "audio",
    "structured_data",
    "user_interaction"
}

class PlannerAgent:
    def __init__(self, client):
        self.client = client

    def extract_struture_from_mermaid(self, mermaid_text):
        """
        Extract the explicit topological structure using regular expressions
        """
        relations = []
        patten = re.compile(r"(\w+)\s*(?:-->|-\.->|==>|--o)\s*\|?.*?\|?\s*(\w+)")
        matches = patten.findall(mermaid_text)
        for src, tgt in matches:
            relations.append({
                "source": src,
                "target": tgt,
                "type": "explicit"
            })

        return relations

    def plan_system_architecture(self, user_query):
        prompt = f"""
        You are a system architect and requirement analyst.
        ## Task
        Given a user query with a user-constructed workflow, output its normalized structure in JSON code.
        1. Identify mandatory components (anchors) and the conenction topology (including parallel connections, data flows, subgraphs, etc. if any) from user query. It might involve the functions that need to be implemented, the technical modules, fixed inputs/outputs, etc.
        2. Complete the design goal based on these anchors and relations which were not originally included in the original query.
        3. For each anchor: based on the user's requirements and the current context, infer the user's expectations for each anchor (their role within the entire workflow), and formulate a concise description.
        4. Extract ALL user defined relations (explicit) from the user query. For example, the arrow '-->' means the data flow from source to target, the use of subgraph means the inclusion relationship. DO NOT REPEAT.
        5. For each anchor, you must also infer its functional role based on the following taxonomy:
           - user_entity: target_user, stakeholder
           - sensing_input: bio_sensor, env_sensor, direct_input
           - processing_core: ai_inference, algorithm, pre_processor
           - interface_output: immersive_display, gui_display, haptic_feedback, audio_output
           - infrastructure: data_storage, communication
           (Output the Level 1 category, or 'Level 1: Level 2' if you are sure)
        Output Structure (in JSON format)
        {{
            "design_goal": "...",
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
                temperature=0.7,
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
        
    def generalize_domain(self, query):
        prompt = f"""
        You're going to analyze the user query and generalize the structure/domain of the query.
        e.g., history education system --> education system/interactive learning system
        User Query: {query}
    """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            ).choices[0].message.content
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        
        
class RetrieverAgent:
    def __init__(self, vector_store, graph_store, client):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.client = client

    def retrieve_for_module(self, anchors_list, top_k=3):
        anchor_mapping = {}
        for anchor in anchors_list:
            user_term = anchor["user_term"]
            description = anchor["description"]
            role = anchor.get("role", None) # Get role
            query_text = f"{user_term}: {description}"
        
            retrieved_nodes = retrieve_and_get_content(
                query_text, 
                self.vector_store, 
                self.graph_store, 
                top_k,
                node_type="MODULE",
                role_filter=role # Pass role filter
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

    def integrate_arch(self, plan, skeletons, retrieved_info, structural_hints=[]):
        skeletons_text = ""
        if skeletons:
            for i, s in enumerate(skeletons):
                skeletons_text += f"--- Skeleton {i} ---\n {s.node.text}\n\n"

        hints_text = ""
        if structural_hints:
            hints_text = "### Validated Connection Paths from Literature\n"
            for i, path in enumerate(structural_hints):
                nodes_chain = " -> ".join([f"[{n['name']}]" for n in path['path_nodes']])
                hints_text += f"   * For connection '{path['source_query']} -> {path['target_query']}':\n"
                hints_text += f"     Reference Path: {nodes_chain}\n"
                # 提取中间节点作为建议
                if len(path['path_nodes']) > 2:
                    middle_nodes = path['path_nodes'][1:-1]
                    hints_text += f"     💡 Suggestion: You might need intermediate modules like: {', '.join([m['name'] for m in middle_nodes])}\n"
            hints_text += "\n"
        prompt = f"""
        You are a professional designer and researcher in the field of HCI. Design a complete pipeline based on the user's requirements and the retrieved information.
        ## User Requirements
        The user requirements are the principle you should follow the most. The nodes in the query must be included in the pipeline.
        User Query: ```{json.dumps(plan, ensure_ascii=False, indent=2)}```
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
        Retrieved Modules: ```{json.dumps(retrieved_info, ensure_ascii=False, indent=2)}```\n\n
        Structural Paths: ```{hints_text}```\n\n
        Reference Skeletons: ```{skeletons_text}```\n\n
        
        ## TASK
        Select the appropriate nodes from the retrieved nodes, conceive the complete data link and functional architecture, and ultimately synthesize a cohesive flowchart/architecture diagram. 
        ### Critical Rules
        1. **Connect the Flow**: Use the topology info to decide logical connections. If Module A usually follows Module B in Paper X, try to preserve that flow if it fits.
        2. **Cite Sources**: Reference the original papers where the nodes are from.
        3. **Redundancy Check**: REMOVE redundant nodes and connections. 
        
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
                temperature=0.7,
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
    planner = PlannerAgent(client)
    retriever = RetrieverAgent(vector_store, graph_store, client)
    integrator = IntegratorAgent(client)
    drawer = DrawerAgent(client)

    arch_plan = planner.plan_system_architecture(user_query)
    # print(arch_plan)
    skeletons = retriever.retrieve_for_skeleton(arch_plan["design_goal"])
    filtered_skeletons = retriever.rerank_with_llm(user_query, skeletons)
    user_relations = arch_plan["relations"]

    # Limit to num_outputs to avoid unnecessary processing if we have more candidates
    target_skeletons = filtered_skeletons[:num_outputs]
    
    # If we don't have enough skeletons, we might just reuse valid ones or proceed with fewer.
    # But usually rerank_with_llm ensures up to 3.
    
    results = []
    
    # Retrieve modules once (shared for all variations)
    ref_nodes_map = retriever.retrieve_for_module(arch_plan["anchors"])
    ref_info = {}
    for anchor in arch_plan["anchors"]:
        anchor_name = anchor["user_term"]
        if anchor_name in ref_nodes_map:
            retrieved_list = ref_nodes_map[anchor_name]
            anchor_nodes = []
            for node_with_score in retrieved_list:
                node = node_with_score.node
                anchor_nodes.append({
                    'id': node.id_,
                    'text': node.text,
                    'code': node.metadata.get("code", ""),
                    'score': node_with_score.score,
                    'metadata': dict(node.metadata)
                })
            ref_info[anchor_name] = anchor_nodes
        else:
            print(f"No modules found for anchor: {anchor_name}")

    structural_hints = retriever.retrieve_structure_paths(user_relations)

    for i, s in enumerate(target_skeletons):
        print(f"\n=== Generating Option {i+1} using Skeleton {i} ===\n")
        print(f"Skeleton Preview: {s.node.text[:100]}...\n")
        
        # We pass ONLY the current skeleton as the reference to ensure diversity
        new_arch = integrator.integrate_arch(
            plan=arch_plan, 
            skeletons=[s], 
            retrieved_info=ref_info, 
            structural_hints=structural_hints
        )
        if new_arch:
            mermaid_code = drawer.draw_flowchart(new_arch)
            results.append({
                "option_id": i + 1,
                "based_on_skeleton_score": s.score,
                "mermaid_code": mermaid_code,
                "description": json.loads(new_arch).get("overall_description", "") if new_arch else ""
            })
            print(f"--- Mermaid Code (Option {i+1}) ---\n{mermaid_code}\n")
        else:
            print(f"Failed to generate architecture for option {i+1}")

    return results

if __name__ == "__main__":
    query = "design a physical education system based on LLM. Nodes: graph TD;\n Trainee[Trainee]-->|user data| Sensor[Sensor]\n LLM[LLM]-->|LLM output| Virtual Reality\n;"
    results = run_pipeline(query, num_outputs=3)
    for res in results:
        print(f"\nFinal Result {res['option_id']}:\n{res['mermaid_code']}")

    # vector_store, graph_store = load_stores()
    # planner = PlannerAgent(client)
    # retriever = RetrieverAgent(vector_store, graph_store, client)
    # query = "design a history education system based on LLM. Nodes: graph TD; LLM;\n Virtual Reality;"
    # arch_plan = planner.plan_system_architecture(query)
    # skeletons = retriever.retrieve_for_skeleton(arch_plan["design_goal"])
    # best_skeleton = retriever.rerank_with_llm(query, skeletons)
    # print(best_skeleton.node.text)
