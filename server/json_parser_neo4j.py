import json
import os
import shutil
import uuid
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.graph_stores.types import Relation
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores import SimpleVectorStore

# --- 配置 ---
DATA_DIR = "./case_llm_education/json_files"
STORAGE_DIR = "./server/graph_storage" 

os.environ["OPENAI_API_KEY"] = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
os.environ["OPENAI_API_BASE"] = "https://poloai.top/v1"

NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "BQLjYc4gOpIxTi6o5c-es481vKejEHHBGuxJeEvX9hg"
NEO4J_URL = "neo4j+s://8891a7fc.databases.neo4j.io"

STANDARD_IO_MAP = {
    "text":["text", "query", "string", "prompt", "document"],
    "image":["image", "frame", "picture", "visual"],
    "video":["video", "stream", "film", "footage", "recording"],
    "audio":["audio", "sound", "voice", "music", "speech"],
    "structured_data":["json", "csv", "vector", "database", "embedding", "graph"],
    "user_interaction":["click", "interaction", "gesture", "feedback"]
}

def normalize_io_type(raw_text):
    if not raw_text or raw_text == "NONE":
        return "unknown"
    for std_type, keywords in STANDARD_IO_MAP.items():
        if any(k in raw_text.lower() for k in keywords):
            return std_type
    return "unknown"

def initialize_storage():
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
    
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
    )
    vector_store = SimpleVectorStore()
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        property_graph_store=graph_store
    )
    return storage_context

def clear_neo4j_database(graph_store):
    """
    清空数据库，防止旧数据干扰
    """
    print("🧹 Clearing Neo4j database...")
    with graph_store._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("✅ Neo4j database cleared.")

def parse_json_to_graph_elements(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = []
    relations = []
    
    article_info = data.get('basic_information', {}) or data.get('article_information', {}) 
    workflows = data.get('workflows', [])
    modules = data.get('modules', [])
    full_code = data.get('full_code', '')

    file_name = article_info.get('file_name', os.path.basename(json_path))

    if article_info.get("arch_exits") == "False":
        return None

    # --- 1. 创建 Article 根节点 ---
    article_id = f"ART_{uuid.uuid4().hex[:8]}"
    
    article_text = (
        f"Article: {file_name}\n"
        f"Function: {article_info.get('function','')}\n"
        f"Application Scenarios: {article_info.get('application_scenarios', '')}\n"
        f"Diagram Caption: {article_info.get('caption', '')}"
    )

    article_metadata = {
        "name": file_name,
        "source": file_name,
        "type": "ARTICLE",
        "full_code": full_code
    }

    article_node = TextNode(
        text=article_text,
        id_=article_id,
        metadata=article_metadata,
        excluded_embed_metadata_keys=["full_code"],
        excluded_llm_metadata_keys=[]
    )
    nodes.append(article_node)

    # 创建映射字典 - 使用 mod_id 和 flow_id 作为键
    module_id_to_node_id = {}
    flow_id_to_node_id = {}

    # --- 2. 创建 Module 节点 ---
    for module_data in modules:
        mod_id = module_data.get('mod_id')
        if not mod_id:
            continue  # 如果没有mod_id，跳过这个模块
            
        node_uid = f"MOD_{mod_id}_{uuid.uuid4().hex[:6]}"
        module_id_to_node_id[mod_id] = node_uid
        
        mermaid_code = module_data.get("mermaid_def", "")
        io_spec = module_data.get('io_spec', {})

        mod_text = (
            f"Module: {mod_id}\n"
            f"Role: {module_data.get('role', 'unknown')}\n"
            f"Function: {module_data.get('function_description', '')}\n"
            f"Technical Details: {module_data.get('technical_details', '')}\n"
            f"Input: {io_spec.get('in', '')}\n"
            f"Output: {io_spec.get('out', '')}"
        )
        
        mod_metadata = {
            "name": mod_id,
            "type": "MODULE",
            "role": module_data.get('role', ''),
            "code": mermaid_code,
            "source": file_name,
        }

        mod_node = TextNode(
            text=mod_text, 
            id_=node_uid, 
            metadata=mod_metadata,
            excluded_embed_metadata_keys=["code"], 
            excluded_llm_metadata_keys=[]   
        )
        nodes.append(mod_node)

        relations.append(Relation(
            source_id=article_id,
            target_id=node_uid,
            label="DEFINES_MODULE"
        ))

    # --- 3. 创建 Workflow 节点 ---
    for wf_data in workflows:
        flow_id = wf_data.get("flow_id")
        if not flow_id:
            continue  # 如果没有flow_id，跳过这个工作流
            
        wf_uid = f"WF_{flow_id}_{uuid.uuid4().hex[:6]}"
        flow_id_to_node_id[flow_id] = wf_uid

        wf_text = (
            f"Workflow: {flow_id}\n"
            f"Description: {wf_data.get('description', '')}\n"
            f"Input: {wf_data.get('input', '')}\n"
            f"Output: {wf_data.get('output', '')}"
        )

        wf_metadata = {
            "name": flow_id,
            "type": "WORKFLOW",
            "source": file_name,
            "flow_type": wf_data.get('flow_type', '')
        }
        
        wf_node = TextNode(
            text=wf_text, 
            id_=wf_uid, 
            metadata=wf_metadata 
        )
        nodes.append(wf_node)

        relations.append(Relation(
            source_id=article_id,
            target_id=wf_uid,
            label="HAS_WORKFLOW"
        ))

    for wf_data in workflows:
        flow_id = wf_data.get("flow_id")

        wf_uid = flow_id_to_node_id.get(flow_id)
        if not wf_uid:
            continue  
        
        # workflow --> workflow (dependencies)
        dependencies = wf_data.get("dependencies", [])
        for dep_flow_id in dependencies:
            target_wf_node_id = flow_id_to_node_id.get(dep_flow_id)
            if target_wf_node_id:
                relations.append(Relation(
                    source_id=wf_uid,
                    target_id=target_wf_node_id,
                    label="DEPENDS_ON"
                ))
            else:
                print(f"⚠️  Workflow {flow_id} references missing workflow with flow_id: {dep_flow_id}")
    

        # workflow --> module (module sequence)
        seq = wf_data.get("module_seq", [])
        for index, mod_id_ref in enumerate(seq):
            target_mod_node_id = module_id_to_node_id.get(mod_id_ref)
            if target_mod_node_id:
                relations.append(Relation(
                    source_id=wf_uid,
                    target_id=target_mod_node_id,
                    label="EXECUTES_STEP",
                    properties={"step_index": index, "step_name_in_flow": mod_id_ref}
                ))
            else:
                print(f"⚠️  Workflow {flow_id} references missing module: {mod_id_ref}")
        
    # --- 4. 创建 Module 之间的 FLOWS_TO 关系 ---
    for mod_data in modules:
        mod_id = mod_data.get('mod_id')
        if not mod_id:
            continue
            
        source_node_id = module_id_to_node_id.get(mod_id)
        if not source_node_id:
            continue  # 如果源模块不存在，跳过
            
        ports = mod_data.get("ports", {})
        outgoing = ports.get("outgoing", [])

        for conn in outgoing:
            target_mod_id = conn.get("target_mod")
            target_mod_node_id = module_id_to_node_id.get(target_mod_id)
            if source_node_id and target_mod_node_id:
                relations.append(Relation(
                    source_id=source_node_id,
                    target_id=target_mod_node_id,
                    label="FLOWS_TO",
                    properties={"edge_label": conn.get("edge_label", ""), 
                               "internal_source": conn.get("originates_from_internal_node", ""), 
                               "target_subnode": conn.get("target_mod_internal_node", "")}
                ))

    return nodes, relations



def force_update_neo4j_content(nodes, graph_store):
    """
    [CRITICAL FIX]: 使用 MERGE 而不是 MATCH。
    这确保如果节点不存在，它会被创建；如果存在，属性会被强制覆盖。
    """
    print(f"\n[NEO4J] Force updating {len(nodes)} nodes with visible text content...")
    driver = graph_store._driver
    
    query = """
    MERGE (n:LlamaNode {id: $node_id})
    SET n += $props
    SET n.text = $text_content
    SET n.name = $name
    SET n.node_type = $node_type
    SET n.code = $code
    """
    
    success_count = 0
    with driver.session() as session:
        for node in nodes:
            try:
                props = node.metadata.copy()
                node_name = props.get("name", '')
                node_type = props.get("type", '')
                node_code = props.get("code") or props.get("full_code") or ""
                for k, v in props.items():
                    if isinstance(v, (dict, list)):
                        if isinstance(v, dict) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], (dict, list))):
                            props[k] = json.dumps(v, ensure_ascii=False)
                if 'id' in props:
                    props['ref_id'] = props.pop('id')
                
                # 执行写入
                session.run(query, 
                            node_id=node.node_id, 
                            text_content=node.text, 
                            name=node_name,
                            node_type=node_type,
                            code=node_code,
                            props=props)
                success_count += 1
            except Exception as e:
                print(f"  Failed to update node {node.node_id}: {e}")
                
    print(f"✅ [NEO4J] Update complete. Processed {success_count}/{len(nodes)} nodes.\n")

def force_update_neo4j_relations(relations, graph_store):
    print(f"\n[NEO4J] Force updating {len(relations)} relations...")
    driver = graph_store._driver
    success_count = 0

    with driver.session() as session:
        for rel in relations:
            try:
                raw_label = rel.label
                label = raw_label.replace(" ", "_").replace("<br/>", "_").replace("<br>", "_")
                query = f"""
                MATCH (s: LlamaNode {{id: $source_id}})
                MATCH (t: LlamaNode {{id: $target_id}})
                MERGE (s)-[r:`{label}`]->(t)
                SET r += $props
                """
                session.run(query, 
                            source_id=rel.source_id, 
                            target_id=rel.target_id, 
                            props=rel.properties)
                success_count += 1
            except Exception as e:
                print(f"  Failed to update relation {rel.source_id} -> {rel.target_id}: {e}")
                
    print(f"✅ [NEO4J] Update complete. Processed {success_count}/{len(relations)} relations.\n")

def build_graph_index():
    if os.path.exists(STORAGE_DIR):
        try:
            shutil.rmtree(STORAGE_DIR)
        except:
            pass

    print("Initializing storage context...")
    storage_context = initialize_storage()
    graph_store = storage_context.property_graph_store
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

    clear_neo4j_database(graph_store)
    
    json_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    all_nodes = []
    all_relations = []

    print(f"Found {len(json_files)} JSON files. Parsing...")
    for json_file in json_files:
        try:
            f_nodes, f_relations = parse_json_to_graph_elements(json_file)
            all_nodes.extend(f_nodes)
            all_relations.extend(f_relations)
            print(f"Parsed {json_file}: {len(f_nodes)} nodes")
        except Exception as e:
            print(f"❌ Error parsing {json_file}: {e}")

    if not all_nodes:
        print("❌ CRITICAL ERROR: No nodes parsed.")
        return

    print(f"Total nodes: {len(all_nodes)}")
    
    print("Generating embeddings manually...")
    texts = [n.text for n in all_nodes]
    embeddings = embed_model.get_text_embedding_batch(texts)
    for node, emb in zip(all_nodes, embeddings):
        node.embedding = emb
    
    # 1. save to Vector Store
    print("1. Adding nodes to Local Vector Store...")
    storage_context.vector_store.add(all_nodes)

    print("2. Force writing nodes to Neo4j...")
    force_update_neo4j_content(all_nodes, graph_store)
    
    if all_relations:
        print(f"3. Upserting {len(all_relations)} relations...")
        force_update_neo4j_relations(all_relations, graph_store)

    print("Persisting Vector Store to disk...")
    storage_context.persist(persist_dir=STORAGE_DIR)
    
    print("="*40)
    print(f"✅ BUILD SUCCESS. Nodes in VectorStore: {len(storage_context.vector_store._data.embedding_dict)}")
    print("="*40)

if __name__ == "__main__":
    build_graph_index()