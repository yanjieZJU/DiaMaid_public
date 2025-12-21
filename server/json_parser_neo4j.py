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
    
    # 1. 注册表：记录所有已创建节点的 ID
    created_node_ids = set()
    
    # 2. 查找表：Mermaid ID -> 全名 ID (用于快速定位子节点)
    component_lookup = {}
    # Component --> Parent
    component_to_parent_map = {}
    existing_module_flows = set()

    article_info = data.get('article_information', {}) 
    if not article_info:
        article_info = data.get('article_info', {}) 
        
    diagram = data.get('diagram', {})
    overview = diagram.get('overview', {})
    modules = diagram.get('modules', {})

    file_name = article_info.get('file_name', os.path.basename(json_path))

    # --- 1. 创建 System 根节点 ---
    system_id = f"SYS_{file_name}_{uuid.uuid4().hex[:8]}".replace(" ", "_")
    
    system_text = (
        f"System Function: {article_info.get('function','')}\n"
        f"Application Scenarios: {article_info.get('application_scenarios', '')}\n"
        f"Diagram Caption: {overview.get('caption', '')}"
    )

    system_metadata = {
        "name": system_id,
        "file_name": file_name,
        "type": "SYSTEM",
        "full_code": overview.get("full_code", "")
    }

    system_node  = TextNode(
        text=system_text,
        id_=system_id,
        metadata=system_metadata,
        excluded_embed_metadata_keys=["full_code"],
        excluded_llm_metadata_keys=[]
    )
    nodes.append(system_node)
    created_node_ids.add(system_id)

    # --- 2. 创建 Module 和 Component 节点 ---
    for module_name, module_data in modules.items():
        mod_id = module_data.get('mod_id')
        if not mod_id: mod_id = module_name
        mod_id = str(mod_id).strip()
        
        mermaid = module_data.get('mermaid_spec', {})
        io_sig = module_data.get('io_signature', {})
        raw_in = io_sig.get('input_type', '')
        raw_out = io_sig.get('output_type', '')

        std_in = normalize_io_type(raw_in)
        std_out = normalize_io_type(raw_out)
        
        mod_text = (
            f"Module: {module_name}\n"
            f"Function: {module_data.get('function_description', '')}\n"
            f"Technical Details: {module_data.get('technical_details', '')}\n"
            f"Input: {io_sig.get('input_type', '')}\n"
            f"Output: {io_sig.get('output_type', '')}"
        )
        
        mod_metadata = {
            "name": module_name,
            "type": "MODULE",
            "role": module_data.get('role', 'processor'),
            "code": mermaid.get('definition_code', ''),
            "parent_system_id": system_id,
            "source": file_name,
            "std_in": std_in,
            "std_out": std_out,
            "raw_in": raw_in,
            "raw_out": raw_out
        }

        mod_node = TextNode(
            text=mod_text, 
            id_=mod_id, 
            metadata=mod_metadata,
            excluded_embed_metadata_keys=["code"], 
            excluded_llm_metadata_keys=[]   
        )
        nodes.append(mod_node)
        created_node_ids.add(mod_id)

        relations.append(Relation(
            source_id=system_id,
            target_id=mod_id,
            label="HAS_MODULE"
        ))
        
        # --- 创建 Subnodes (Components) ---
        sub_nodes = mermaid.get("sub_nodes",[])
        for sub in sub_nodes:
            sub_raw_id = sub.get("node_id")
            if not sub_raw_id: continue
            
            sub_full_id = f"{mod_id}_{sub_raw_id}"
            
            # 存入查找表
            component_lookup[sub_raw_id] = sub_full_id
            component_to_parent_map[sub_full_id] = mod_id
            
            sub_text = (
                f"Component: {sub_raw_id}\n"
                f"Function: {sub.get('node_function','')}"
            )

            sub_metadata = {
                "name": sub_raw_id,
                "id": sub_raw_id,
                "type": "COMPONENT",
                "role": sub.get('node_role', ''),
                "parent_module": mod_id
            }

            sub_node = TextNode(
                text=sub_text, id_=sub_full_id, metadata=sub_metadata
            )
            nodes.append(sub_node)
            created_node_ids.add(sub_full_id)

            relations.append(Relation(
                source_id=mod_id, target_id=sub_full_id, label="HAS_PART"
            ))

    # --- 3. 创建 Flow 关系 (增强版逻辑) ---
    for module_name, module_data in modules.items():
        mod_id = module_data.get('mod_id')
        if not mod_id: mod_id = module_name
        mod_id = str(mod_id).strip()

        component_to_parent_map[mod_id] = mod_id
        mermaid = module_data.get('mermaid_spec', {})
        ports = mermaid.get('ports', {})
        outgoing = ports.get('outgoing', [])

        for conn in outgoing:
            target_raw_name = conn.get('target')
            if not target_raw_name or target_raw_name == "NONE":
                continue

            # --- A. 处理源节点 (支持逗号分割) ---
            raw_src_internal = conn.get('originates_from_internal_node')
            source_candidates = []

            if not raw_src_internal or raw_src_internal == "NONE":
                # 默认源头是模块
                source_candidates.append(mod_id)
            else:
                # 拆分逗号 (例如 "Microphone, Camera")
                internal_parts = [p.strip() for p in raw_src_internal.split(',')]
                
                for p in internal_parts:
                    # 尝试构建子节点 ID
                    potential_sub_id = f"{mod_id}_{p}"
                    
                    if potential_sub_id in created_node_ids:
                        source_candidates.append(potential_sub_id)
                    elif p == mod_id and mod_id in created_node_ids:
                        # 修正: 如果子节点名等于模块名 (如 Coachee), 回退到模块 ID
                        source_candidates.append(mod_id)
                    else:
                        print(f"⚠️ Warning: Internal source '{p}' not found in module '{mod_id}'. Skipping this part.")

            # --- B. 处理目标节点 ---
            # 同样尝试查找 component，如果找不到则找 module
            target_full_id = None
            
            if target_raw_name in component_lookup:
                target_full_id = component_lookup[target_raw_name]
            elif target_raw_name in created_node_ids:
                target_full_id = target_raw_name
            else:
                # 如果目标真的不存在 (比如 'End'), 这里会依然跳过
                print(f"⚠️ Warning: Target '{target_raw_name}' not found (Missing module definition?). Skipping relation.")
                continue

            # --- C. 创建所有连线 ---
            for src_id in source_candidates:
                if src_id in created_node_ids and target_full_id in created_node_ids: # 双重检查
                    relations.append(Relation(
                        source_id=src_id,
                        target_id=target_full_id,
                        label="FLOWS_TO",
                        properties={"meaning": conn.get('data_flow_meaning', 'Data Flow'), "description": conn.get('edge_label', '')}
                    ))

                    parent_src = component_to_parent_map.get(src_id, src_id)
                    parent_tgt = component_to_parent_map.get(target_full_id, target_full_id)

                    if parent_src != parent_tgt:
                        flow_key = f"{parent_src}->{parent_tgt}"
                        relations.append(Relation(
                            source_id=parent_src,
                            target_id=parent_tgt,
                            label="FLOWS_TO",
                            properties={"type": "high_level", "generated_from": f"{src_id}->{target_full_id}"}
                        ))
                        existing_module_flows.add(flow_key)

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