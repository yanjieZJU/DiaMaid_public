import os
import httpx
import json
from openai import OpenAI
from retriever_neo4j import load_stores, retrieve_and_get_content

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

    def plan_system_architecture(self, user_query):
        prompt = f"""
        You are a system architect and requirement analyst.
        ## Task
        1. Identify mandatory components (anchors) from user query. It might involve the functions that need to be implemented, the technical modules, fixed inputs/outputs, etc.
        2. Complete the design goal based on these anchors which were not originally included in the original query.
        3. For each anchor:
            a. generate 3-5 technical synonyms commonly found in academic papers (HCI/CS domain).
            b. based on the user's requirements and the current context, infer the user's expectations for each anchor (their role within the entire workflow), and formulate a concise description.
            
        Output Structure (in JSON format)
        {{
            "design_goal": "...",
            "anchors": [
                {{
                    "user_term": "the user term for this anchor, e.g., AI tutor",
                    "aliases": "the alias of this anchor, use array. e.g., ["conversational agent", "chatbot"]",
                    "description": "the description of this anchor, e.g., a conversational agent that can answer questions about the user's topic."
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

        ref_nodes = []
        for anchor in anchors_list:
            user_term = anchor["user_term"]
            aliases = anchor["aliases"]
            description = anchor["description"]
            query_text = f"{user_term}: {description}; Aliases: {' or '.join(aliases)}"
        
            retrieved_nodes = retrieve_and_get_content(
                query_text, 
                self.vector_store, 
                self.graph_store, 
                top_k,
                node_type="MODULE",
                in_type=None
            )
            ref_nodes.extend(retrieved_nodes)

        for n in ref_nodes:
            print(n.node.text)
            # print(n.node.metadata.get("topology", "N/A"))
            print(n.score)
        # for node_with_score in retrieved_nodes:
        #     if node_with_score.score >= 0.7:
        #         valid_nodes.append(node_with_score.node)

        return ref_nodes
    
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
                for i in range(len(valid_skeletons)):
                    if i not in indices and len(valid_skeletons) < 3:
                        valid_skeletons.append(retrieved_skeletons[i])

            return valid_skeletons
            
        except Exception as e:
            print(f"Error: {e}")
            return None


class IntegratorAgent:
    def __init__(self, client):
        self.client = client

    def integrate_arch(self, query, skeletons, retrieved_info):
        skeletons_text = ""
        for i, s in enumerate(skeletons):
            skeletons_text += f"--- Skeleton {i} ---\n {s.node.text}\n\n"
        prompt = f"""
        You are a professional designer and researcher in the field of HCI. Try to plan a pipeline based on the user's requirements and the retrieved information, and generate a Mermaid graph/flowchart to visualize the pipeline. 
        User Query: {query}
        This is the user's demand, and it is the principle you should follow the most. The nodes in the query must be included in the pipeline.

        ## Retrieved Information
        For the entire system architecture, here is the most similar system found in the literature.
        You can use this as a reference or skeleton to guide your design:
        {skeletons_text}
        \n\n
        For each conceptual step, here are real validated modules found in papers:
        {json.dumps(retrieved_info, ensure_ascii=False, indent=2)}
        They incorporate the technical methods, module architectures and data processing approaches used in other studies. 
        Each module may have retrieved multiple nodes with similar functions. You need to conduct appropriate screening to be able to assemble a complete process.

        ## TASK
        Finally, you need to select the appropriate nodes from the retrieved nodes, conceive the complete data link and functional architecture, and ultimately synthesize a cohesive flowchart/architecture diagram. 
        ### Critical Rules
        1. **Connect the Flow**: Use the topology info to decide logical connections. If Module A usually follows Module B in Paper X, try to preserve that flow if it fits.
        2. **Cite Sources**: Reference the original papers where the nodes are from.
        3. **Resolve IO Conflicts**: If two modules have the different input/output type, you need to resolve the conflict. Insert a node to convert the input/output type (and mark it as [generated]).
        
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
                b. Use subgraphs to represent modules if there are more than one subnodes and data flow between them in a single module. 
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
            
def run_pipeline(user_query):
    vector_store, graph_store = load_stores()
    planner = PlannerAgent(client)
    retriever = RetrieverAgent(vector_store, graph_store, client)
    integrator = IntegratorAgent(client)
    drawer = DrawerAgent(client)
    arch_plan = planner.plan_system_architecture(user_query)
    print(arch_plan)
    skeletons = retriever.retrieve_for_skeleton(arch_plan["design_goal"])
    filtered_skeletons = retriever.rerank_with_llm(user_query, skeletons)
    
    for i, s in enumerate(filtered_skeletons):
        print(f"--- Skeleton {i} ---\n {s.node.text}\n\n")

    ref_nodes = retriever.retrieve_for_module(arch_plan["anchors"])
    ref_info = {}
    for anchor in arch_plan["anchors"]:
        anchor_name = anchor["user_term"]
        anchor_nodes = []
        for node_with_score in ref_nodes:
            node = node_with_score.node
            if anchor_name.lower() in node.text.lower() or any(alias.lower() in node.text.lower() for alias in anchor["aliases"]):
                anchor_nodes.append({
                    'id': node.id_,
                    'text': node.text,
                    'code': node.metadata.get("code", ""),
                    'score': node_with_score.score,
                    'metadata': dict(node.metadata)
                })
        if anchor_nodes:
            ref_info[anchor_name] = anchor_nodes
    new_arch = integrator.integrate_arch(query=user_query, skeletons=filtered_skeletons, retrieved_info=ref_info)
    mermaid_code = drawer.draw_flowchart(new_arch)
    print(mermaid_code)
    return mermaid_code

if __name__ == "__main__":
    query = "design a history education system based on LLM. Nodes: graph TD;\n Subgraph\n vr[Virtual Reality]--> EyeTracking[Eye Tracking]\n end\n;"
    run_pipeline(query)

    # vector_store, graph_store = load_stores()
    # planner = PlannerAgent(client)
    # retriever = RetrieverAgent(vector_store, graph_store, client)
    # query = "design a history education system based on LLM. Nodes: graph TD; LLM;\n Virtual Reality;"
    # arch_plan = planner.plan_system_architecture(query)
    # skeletons = retriever.retrieve_for_skeleton(arch_plan["design_goal"])
    # best_skeleton = retriever.rerank_with_llm(query, skeletons)
    # print(best_skeleton.node.text)
