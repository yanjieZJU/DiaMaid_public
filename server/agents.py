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
        Your task is to decompose the user query into a series of functional requirements and technical constraints.
        DO NOT name specific technologies.
        DO NOT create a final architecture.
        Output Structure (in JSON format)
        {{
            "overall_description": "How you plan to architect this system/pipeline?",
            "details": [
                {{
                    "name": "the name of this step",
                    "goal": "the goal of this step",
                    "required_inputs": "the required input type for this step. Only choose from {STANDARD_IO_MAP}",
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
        
        
class RetrieverAgent:
    def __init__(self, vector_store, graph_store):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve_for_module(self, module_info, top_k=1):
        query_text = f"{module_info['goal']}"
        in_type = module_info.get('required_inputs', '')
        retrieved_nodes = retrieve_and_get_content(
            query_text, 
            self.vector_store, 
            self.graph_store, 
            top_k,
            in_type=in_type
        )
        return retrieved_nodes


class IntegratorAgent:
    def __init__(self, client):
        self.client = client

    def integrate_arch(self, query, plan, retrieved_info):
        prompt = f"""
        You are a professional designer and researcher in the field of HCI. Try to plan a pipeline based on the user's requirements and the retrieved information, and generate a Mermaid graph/flowchart to visualize the pipeline. 
        User Query: {query}
        This is the user's demand, and it is the principle you should follow the most.

        ## Plan
        Overall Description: {plan["overall_description"]}
        Details: {json.dumps(plan["details"], ensure_ascii=False, indent=2)}
        This is an idea, which encompasses the overall description of the entire system and possible functional modules. It meets the user's requirements, but lacks feasibility and specific details.

        ## Retrieved Information
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
                    "references": "the references of this module, e.g. [Paper X]",
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
        Make a flowchart for the following system architecture plan. Use Mermaid syntax.
        {arch_plan}
        The flowchart should be in Mermaid.js format. 
        ## CRITICAL RULES:
            1. **Adhere strictly to the grammar of Mermaid** to ensure that it does not generate any errors when compiled. 
            2. You need to modify the names and descriptions of each node and relationship according to the user's requirements and context, so that the user can fully understand this flowchart.
        
        Final Output Structure (in JSON format)
        {{
            "mermaid_code": "...",
        }}
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
        # 替换转义的换行符
        code = code.replace('\\n', '\n')
        # 替换转义的双引号
        code = code.replace('\\"', '"')
        # 移除可能存在的首尾引号（如果有的话）
        if code.startswith('"') and code.endswith('"'):
            code = code[1:-1]
        # 移除多余的空白字符
        code = code.strip()
        
        return code
            
def main():
    vector_store, graph_store = load_stores()
    planner = PlannerAgent(client)
    retriever = RetrieverAgent(vector_store, graph_store)
    integrator = IntegratorAgent(client)
    drawer = DrawerAgent(client)

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

    arch = integrator.integrate_arch(query=user_query, plan=arch_plan, retrieved_info=retrieved_info)
    mermaid_code = drawer.draw_flowchart(arch)
    print(mermaid_code)

if __name__ == "__main__":
    main()
    # planner = PlannerAgent(client)
    # user_query = "design a history education system based on LLM"
    # arch_plan = planner.plan_system_architecture(user_query)
    # print(arch_plan)