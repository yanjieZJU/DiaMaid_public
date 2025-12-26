import json
import os
from llama_index.core import PropertyGraphIndex
from llama_index.core.schema import TextNode
from llama_index.core.graph_stores.types import Relation
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.graph_stores import SimplePropertyGraphStore

# --- 配置 ---
DATA_DIR = "./json_files"
STORAGE_DIR = "./graph_storage"  # 数据将保存在这里
GRAPH_STORE_PATH = os.path.join(STORAGE_DIR, "property_graph_store.json")
os.environ["OPENAI_API_KEY"] = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
os.environ["OPENAI_API_BASE"] = "https://poloai.top/v1"

def initialize_index():
    """
    初始化图索引 (使用 SimpleStore，无需 Neo4j)
    """
    # 确保存储目录存在
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    if os.path.exists(GRAPH_STORE_PATH):
        graph_store = SimplePropertyGraphStore.from_persist_path(GRAPH_STORE_PATH)
    else:
        print("No existing graph store found.")
        graph_store = SimplePropertyGraphStore()

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
        llm=OpenAI(model="gpt-4o"),
        kg_extractors=[], # 禁用自动抽取，完全依赖 JSON 结构
    )
    return index

def find_json_files(folder_path=DATA_DIR):
    json_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                json_files.append(os.path.join(folder_path, file))
    return json_files

def parse_json_to_graph_elements(json_path):
    """解析 JSON 提取节点和边"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 尝试多种路径获取 modules
    modules = data.get('modules')
    if not modules:
        content_comp = data.get('content_comprehension', {})
        if content_comp:
            modules = content_comp.get('details', {})
    
    if not modules:
        return [], []

    nodes = []
    relations = []
    
    # 1. 构建节点
    for module_name, module_data in modules.items():
        node_id = module_data.get('node_id', module_name)
        
        content_text = (
            f"Module: {module_name}\n"
            f"Function: {module_data.get('function_description', module_data.get('function', ''))}\n"
            f"Details: {module_data.get('technical_details', '')}\n"
        )

        properties = {
            "name": module_name,
            "role": module_data.get('role', 'unknown'),
            "file_source": os.path.basename(json_path)
        }

        node = TextNode(
            text=content_text,
            id_=node_id,
            metadata=properties
        )
        nodes.append(node)

        # 2. 构建 Mermaid 关系
        mermaid_spec = module_data.get('mermaid_spec', {})
        ports = mermaid_spec.get('ports', {})
        outgoing = ports.get('outgoing', [])

        for conn in outgoing:
            target_id = conn.get('target')
            edge_label = conn.get('edge_label', 'CONNECTS_TO')
            flow_meaning = conn.get('data_flow_meaning', '')

            if target_id and target_id != "NONE":
                rel = Relation(
                    source_id=node_id,
                    target_id=target_id,
                    label=edge_label.upper().replace(" ", "_"),
                    properties={"meaning": flow_meaning}
                )
                relations.append(rel)

    return nodes, relations

def build_graph_index(json_files=None):
    if json_files is None:
        json_files = find_json_files(DATA_DIR)
    
    index = initialize_index()
    
    all_nodes = []
    all_relations = []

    print("Parsing JSON files...")
    for json_file in json_files:
        try:
            f_nodes, f_relations = parse_json_to_graph_elements(json_file)
            all_nodes.extend(f_nodes)
            all_relations.extend(f_relations)
            print(f"Parsed {json_file}: {len(f_nodes)} nodes, {len(f_relations)} relations")
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    # 批量插入节点
    if all_nodes:
        print(f"Inserting {len(all_nodes)} nodes...")
        index.insert_nodes(all_nodes)
    
    # 批量插入关系
    if all_relations:
        print(f"Inserting {len(all_relations)} relations...")
        index.property_graph_store.upsert_relations(all_relations)

    # 关键：保存到磁盘！
    print("Persisting to disk...")
    index.property_graph_store.persist(GRAPH_STORE_PATH)
    # 注意：vector_store 通常也需要 persist，但在 SimplePGStore 中它是绑定的
    # 如果使用默认 VectorStore，也需要保存
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    print("Graph construction complete!")
    return index

if __name__ == "__main__":
    build_graph_index()