import os
import httpx
import json
from openai import OpenAI
from retriever_neo4j import load_stores, retrieve_and_get_content
from process_mermaid import clean_mermaid_code

http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(
    api_key="sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7",
    base_url="https://poloai.top/v1",
    http_client=http_client
)

class PlannerAgent:
    def __init__(self, client):
        self.client = client

    def plan_system_architecture(self, user_query):
        prompt = f"""
        You are a professional designer and researcher in the field of HCI. Try to plan a pipeline based on the user's requirements. The user query may include their explicit inputs and outputs. If so, please strictly follow their needs.
        You should return several function (technical) modules that could create a complete pipeline that meets the requirements, and logically feasible.
        Output Structure (in JSON format)
        {{
            "overall_description": "How you plan to architect this system/pipeline? What technologies/methods are used?", 
            "details": [
                {{
                    "name": "the module name", 
                    "function": "the functions of this module", 
                    "in_type": "the needed input type",
                    "out_type": "the output type of this module"
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
        
    def feedback_and_plan(self, feedback):
        prompt = f"""
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        ).choices[0].message.content
        try:
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return feedback["last_plan"]
        
class RetrieverAgent:
    def __init__(self, vector_store, graph_store):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve_for_module(self, module_info, top_k=1):
        query_text = f"Module: {module_info['name']}\n Function: {module_info['function']}"
        in_type = module_info.get('in_type', '')
        out_type = module_info.get('out_type', '')
        retrieved_nodes = retrieve_and_get_content(
            query_text, 
            self.vector_store, 
            self.graph_store, 
            top_k,
            in_type=in_type,
            out_type=out_type
        )
        return retrieved_nodes


class IntegratorAgent:
    def __init__(self, client):
        self.client = client

    def integrate_and_generate_mermaid(self, query, plan, retrieved_info):
        prompt = f"""
        You are a professional designer and researcher in the field of HCI. Try to plan a pipeline based on the user's requirements and the retrieved information, and generate a Mermaid graph/flowchart to visualize the pipeline. 
        User Query: {query}
        This is the user's demand, and it is the principle you should follow the most.

        ## Plan
        Overall Description: {plan["overall_description"]}
        Details: {json.dumps(plan["details"], ensure_ascii=False, indent=2)}
        This is an idea, which encompasses the overall description of the entire system and possible functional modules. It meets the user's requirements, but lacks feasibility and specific details.

        ## Retrieved Information
        {json.dumps(retrieved_info, ensure_ascii=False, indent=2)}
        This is the node and module architecture retrieved from the relevant papers based on the functions of each conceptual module. It incorporates the technical methods, module architectures and data processing approaches used in other studies. Each module may have retrieved multiple nodes with similar functions. You need to conduct appropriate screening to be able to assemble a complete process.

        Finally, you need to select the appropriate nodes from the retrieved nodes, conceive the complete data link and functional architecture, and ultimately generate a complete Mermaid code. 
        Note:
        1. Adhere strictly to the grammar of Mermaid to ensure that it does not generate any errors when compiled. 
        2. You need to modify the names and descriptions of each node and relationship according to the user's requirements and context, so that the user can fully understand this flowchart.

        Final Output Structure (in JSON format)
        {{
            "overall_description": "How you plan to architect this system/pipeline? What technologies/methods are used?",
            "mermaid_code": "the generated mermaid code",
            "module_details": [
                "<module name>": {{
                    "name": "the module name",
                    "function": "the functions of this module",
                    "technical_methods": "the technical methods used in this module"
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
            mermaid_code = json.loads(response)["mermaid_code"]
            cleaned_code = clean_mermaid_code(mermaid_code)
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return None
        return cleaned_code
    
def interative_planning(
    user_query,
    planner,
    retriever,
    max_iter=3
):
    plan = planner.plan_system_architecture(user_query)
    retrieved_info = {}
    for iter_id in range(max_iter):
        for module in plan["details"]:
            nodes = retriever.retrieve_for_module(module)
            retrieved_info[module["name"]] = [{
                'id': node.node.id_,
                'text': node.node.text,
                'code': node.node.metadata.get("code", ""),
                'score': node.score
            } for node in nodes]
        feedback_input = {
            "query": user_query,
            "last_plan": plan,
            "retrieved_info": retrieved_info
        }
        plan = planner.feedback_and_plan(feedback_input)
    
    return plan, retrieved_info
            
def main():
    vector_store, graph_store = load_stores()
    planner = PlannerAgent(client)
    retriever = RetrieverAgent(vector_store, graph_store)
    integrator = IntegratorAgent(client)

    user_query = "design a history education system based on LLM"

    # 1. Plan System Architecture
    arch_plan = planner.plan_system_architecture(user_query)
    if arch_plan:
        print("System Architecture Plan:")
        print(json.dumps(arch_plan, indent=2))
    else:
        print("Failed to generate system architecture plan.")
        return
    
    retrieved_info = {}
    for i, module in enumerate(arch_plan["details"]):
        nodes = retriever.retrieve_for_module(module)
        retrieved_info[module["name"]] = [{
            'id': node.node.id_,
            'text': node.node.text,
            'code': node.node.metadata.get("code", ""),
            'score': node.score
        } for node in nodes]

        print(f"Retrieved {len(nodes)} relevant nodes for module {module['name']}")

    print("Retrieval finished. Retrieved Info:")
    print(json.dumps(retrieved_info, indent=2))

    mermaid_code = integrator.integrate_and_generate_mermaid(query=user_query, plan=arch_plan, retrieved_info=retrieved_info)

    print(mermaid_code)

if __name__ == "__main__":
    main()