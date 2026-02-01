import json
import os
import shutil
import uuid
import re
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.graph_stores.types import Relation
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores import SimpleVectorStore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# --- 配置 ---
DATA_DIR = "./case_llm_artifact/json_files"
STORAGE_DIR = "./server/graph_storage" 

VALID_ROLES_MAP = {
    "sensing_input": ["bio_sensor", "behavior_sensor", "env_sensor", "direct_input", "system_context", "data_import"],
    "processing_core": ["signal_optimization", "feature_extraction", "state_assessment", "decision_logic", "content_generation"],
    "interface_output": ["immersive_display", "gui_display", "haptic_feedback", "audio_output"],
    "infrastructure": ["data_storage", "communication", "model_lifecycle", "personalization_store"]
}

ROLE_TO_CATEGORY = {}
for cat, roles in VALID_ROLES_MAP.items():
    for r in roles:
        ROLE_TO_CATEGORY[r] = cat

STANDARD_IO_MAP = {
    "text":["text", "query", "string", "prompt", "document"],
    "image":["image", "frame", "picture", "visual"],
    "video":["video", "stream", "film", "footage", "recording"],
    "audio":["audio", "sound", "voice", "music", "speech"],
    "structured_data":["json", "csv", "vector", "database", "embedding", "graph"],
    "user_interaction":["click", "interaction", "gesture", "feedback"]
}

def analyze_similarity_distribution(nodes):
    # 1. 提取所有 Module 节点的 Embedding
    module_nodes = [n for n in nodes if n.metadata.get("type") == "MODULE" and n.embedding]
    
    if len(module_nodes) < 2:
        print("节点数量不足，无法分析。")
        return

    print(f"正在分析 {len(module_nodes)} 个模块节点的相似度分布...")
    
    embeddings = np.array([n.embedding for n in module_nodes])
    sources = [n.metadata.get("source") for n in module_nodes]
    texts = [n.text for n in module_nodes]
    
    # 2. 计算矩阵
    sim_matrix = cosine_similarity(embeddings)
    
    # 3. 收集所有跨文章的相似度分数
    cross_doc_scores = []
    pairs = []
    
    for i in range(len(module_nodes)):
        for j in range(i + 1, len(module_nodes)):
            # 只看跨文章的
            if sources[i] != sources[j]:
                score = sim_matrix[i][j]
                cross_doc_scores.append(score)
                pairs.append({
                    "score": score,
                    "node_a": texts[i].split('\n')[0], # 只打印第一行方便看
                    "node_b": texts[j].split('\n')[0]
                })

    if not cross_doc_scores:
        print("没有跨文章的节点对。")
        return

    # 4. 打印统计数据
    print("\n=== 相似度统计 (跨文章) ===")
    print(f"最大值: {max(cross_doc_scores):.4f}")
    print(f"平均值: {np.mean(cross_doc_scores):.4f}")
    print(f"中位数: {np.median(cross_doc_scores):.4f}")
    print(f"90分位: {np.percentile(cross_doc_scores, 90):.4f} (意味着前10%的对分数高于此值)")
    
    # 5. 打印 Top 5 最相似的对，让你看看是否合理
    print("\n=== Top 5 最相似的模块对 ===")
    sorted_pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)[:5]
    for p in sorted_pairs:
        print(f"Score: {p['score']:.4f} | {p['node_a']} <--> {p['node_b']}")


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
    modules = data.get('modules', [])
    full_code = data.get('full_code', '')

    file_name = article_info.get('file_name', os.path.basename(json_path))

    if article_info.get("arch_exits") == "False":
        return None

    # --- 1. 创建 Article 根节点 ---
    article_id = f"ART_{uuid.uuid4().hex[:8]}"
    
    article_text = (
        # f"Article: {file_name}\n"
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

    
    # 跟踪本文件内创建的角色节点，避免重复
    created_role_nodes = {}

    # --- 2. 创建 Module 节点 ---
    for module_data in modules:
        mod_id = module_data.get('mod_id')
        if not mod_id:
            continue  # 如果没有mod_id，跳过这个模块
            
        node_uid = f"MOD_{mod_id}_{uuid.uuid4().hex[:6]}"
        module_id_to_node_id[mod_id] = node_uid
        
        mermaid_code = module_data.get("mermaid_def", "")
        io_spec = module_data.get('io_spec', {})

        mod_full_text = (
            f"Module: {mod_id}\n"
            f"Role: {module_data.get('role', 'unknown')}\n"
            f"Function: {module_data.get('function_description', '')}\n"
            f"Technical Details: {module_data.get('technical_details', '')}\n"
            f"Input: {io_spec.get('in', '')}\n"
            f"Output: {io_spec.get('out', '')}"
        )

        semantic_text_for_embedding = (
            f"Name: {mod_id}\n"
            f"Role: {module_data.get('role', 'unknown')}\n"
            f"Function: {module_data.get('function_description', '')}\n"
            f"Input: {io_spec.get('in', '')}\n"
            f"Output: {io_spec.get('out', '')}"
        )
        
        mod_metadata = {
            "name": mod_id,
            "type": "MODULE",
            "role": module_data.get('role', ''),
            "function": module_data.get('function_description', ''),
            "code": mermaid_code,
            "source": file_name,
            "subnodes": module_data.get('subnodes', [])
        }
            
        mod_node = TextNode(
            text=mod_full_text, 
            id_=node_uid, 
            metadata=mod_metadata,
            excluded_embed_metadata_keys=["code"], 
            excluded_llm_metadata_keys=[]   
        )
        mod_node.metadata["_embed_content"] = semantic_text_for_embedding
        nodes.append(mod_node)

        relations.append(Relation(
            source_id=article_id,
            target_id=node_uid,
            label="DEFINES_MODULE"
        ))

        # --- NEW: 处理 Role 节点 (with Validation) ---
        role_str = module_data.get('role', '')
        # 格式预期: "category: subclass" (例如: "user_entity: target_user")
        if role_str and ':' in role_str:
            parts = role_str.split(':', 1)
            raw_cat = parts[0].strip()
            raw_role = parts[1].strip()
            
            final_cat = None
            final_role = None
            
            # Check validity
            is_valid_cat = raw_cat in VALID_ROLES_MAP
            is_valid_role = is_valid_cat and (raw_role in VALID_ROLES_MAP[raw_cat])
            
            if is_valid_role:
                final_cat = raw_cat
                final_role = raw_role
            else:
                # Correction Logic
                # Case 1: The "Category" part is actually a valid Role (e.g. content_generation: text_generation)
                if raw_cat in ROLE_TO_CATEGORY:
                    final_cat = ROLE_TO_CATEGORY[raw_cat]
                    final_role = raw_cat # The category text was actually the role
                    print(f"⚠️ [Role Fix] Mapped invalid '{role_str}' to '{final_cat}: {final_role}'")
                
                # Case 2: Broad Category is valid, but Role is invalid (e.g. processing_core: data_collection)
                elif is_valid_cat:
                    final_cat = raw_cat
                    final_role = None # No valid role to assign
                    print(f"⚠️ [Role Fix] Invalid role in '{role_str}'. Fallback to Category '{final_cat}'.")
                
                else:
                    print(f"⚠️ [Role Fix] Totally invalid role '{role_str}'. Skipping.")
                    continue

            # --- Create Nodes & Relations ---
            
            # 1. Category Node (Must exist if final_cat is set)
            if final_cat:
                cat_node_id = f"ROLE_CAT_{final_cat.replace(' ', '_')}"
                if cat_node_id not in created_role_nodes:
                    cat_text = f"Role Category: {final_cat}"
                    cat_node = TextNode(
                        text=cat_text,
                        id_=cat_node_id,
                        metadata={
                            "name": final_cat,
                            "type": "ROLE_CATEGORY",
                            "source": "SYSTEM_ONTOLOGY" 
                        }
                    )
                    created_role_nodes[cat_node_id] = cat_node
                    nodes.append(cat_node)
                
                # 2. Role Node (Only if final_role is set)
                if final_role:
                    sub_node_id = f"ROLE_SUB_{final_role.replace(' ', '_')}"
                    if sub_node_id not in created_role_nodes:
                        sub_text = f"Role: {final_role}\nCategory: {final_cat}"
                        sub_node = TextNode(
                            text=sub_text,
                            id_=sub_node_id,
                            metadata={
                                "name": final_role,
                                "type": "ROLE",
                                "category": final_cat,
                                "source": "SYSTEM_ONTOLOGY"
                            }
                        )
                        created_role_nodes[sub_node_id] = sub_node
                        nodes.append(sub_node)
                        
                        # Sub -> Cat Relation
                        relations.append(Relation(
                            source_id=sub_node_id,
                            target_id=cat_node_id,
                            label="IS_CATEGORY"
                        ))
                    
                    # Module -> Role Relation
                    relations.append(Relation(
                        source_id=node_uid,
                        target_id=sub_node_id,
                        label="HAS_ROLE"
                    ))
                
                else:
                    # Fallback: Link Module -> Category Node directly
                    relations.append(Relation(
                        source_id=node_uid,
                        target_id=cat_node_id,
                        label="HAS_ROLE"
                    ))


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
    print(f"\n[NEO4J] Force updating {len(nodes)} nodes with visible text content (Batch Mode)...")
    driver = graph_store._driver
    
    query = """
    UNWIND $batch AS row
    MERGE (n:LlamaNode {id: row.node_id})
    SET n += row.props
    SET n.text = row.text_content
    SET n.name = row.name
    """
    
    batch_data = []
    for node in nodes:
        props = node.metadata.copy()
        node_name = props.get("name", '')
        if 'full_code' in props:
            props['code'] = props['full_code']
            del props['full_code']
        
        if 'id' in props:
            props['ref_id'] = props.pop('id')
            
        batch_data.append({
            "node_id": node.node_id,
            "text_content": node.text,
            "name": node_name,
            "props": props
        })
        
    # Batch execution
    batch_size = 500
    total_processed = 0
    with driver.session() as session:
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i+batch_size]
            try:
                session.run(query, batch=batch)
                total_processed += len(batch)
                print(f"  Processed {total_processed}/{len(nodes)} nodes...")
            except Exception as e:
                print(f"  Failed to update batch starting at {i}: {e}")

    print(f"✅ [NEO4J] Update complete.\n")

def force_update_neo4j_relations(relations, graph_store):
    print(f"\n[NEO4J] Force updating {len(relations)} relations (Batch Mode)...")
    driver = graph_store._driver
    
    # Group by label to optimize batching
    grouped_rels = {}
    for rel in relations:
        raw_label = rel.label
        label = raw_label.replace(" ", "_").replace("<br/>", "_").replace("<br>", "_")
        if label not in grouped_rels:
            grouped_rels[label] = []
        
        grouped_rels[label].append({
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "props": rel.properties
        })
    
    with driver.session() as session:
        for label, batch in grouped_rels.items():
            print(f"  Updating {len(batch)} relations of type '{label}'...")
            query = f"""
            UNWIND $batch AS row
            MATCH (s: LlamaNode {{id: row.source_id}})
            MATCH (t: LlamaNode {{id: row.target_id}})
            MERGE (s)-[r:`{label}`]->(t)
            SET r += row.props
            """
            
            # Sub-batch execution
            batch_size = 1000
            for i in range(0, len(batch), batch_size):
                sub_batch = batch[i:i+batch_size]
                try:
                    session.run(query, batch=sub_batch)
                except Exception as e:
                     print(f"  Failed to update batch of {label}: {e}")

    print(f"✅ [NEO4J] Relations update complete.\n")

def create_semantic_bridges(nodes, graph_store, threshold=0.76):
    """
    
    """
    # 1. 筛选出所有的 MODULE 节点且具备 Embedding
    module_nodes = [n for n in nodes if n.metadata.get("type") == "MODULE" and n.embedding]
    
    if len(module_nodes) < 2:
        print("Not enough module nodes to bridge.")
        return

    print(f"   Calculating similarity matrix for {len(module_nodes)} modules...")
    
    # 2. 提取 Embeddings 构建矩阵
    # shape: (N, 1536)
    embeddings = np.array([n.embedding for n in module_nodes])
    node_ids = [n.node_id for n in module_nodes]
    sources = [n.metadata.get("source") for n in module_nodes]
    
    # 3. 计算余弦相似度矩阵 (O(N^2)) - 对于 <10000 节点非常快
    sim_matrix = cosine_similarity(embeddings)
    
    # 4. 遍历矩阵寻找高相似度对
    new_relations = []
    count = 0
    
    # 只遍历上三角矩阵，避免重复 (A-B 和 B-A) 和自环 (A-A)
    # range(len) 是 0 到 N-1
    for i in range(len(module_nodes)):
        for j in range(i + 1, len(module_nodes)):
            score = sim_matrix[i][j]
            
            # 逻辑判断：
            # 1. 分数超过阈值
            # 2. 来源不同 (可选：通常我们希望连接不同文章的模块，同一文章内已有拓扑连接)
            if score >= threshold and sources[i] != sources[j]:
                rel = Relation(
                    source_id=node_ids[i],
                    target_id=node_ids[j],
                    label="IS_SIMILAR",
                    properties={"score": float(score)}
                )
                new_relations.append(rel)
                count += 1

    print(f"   Found {count} semantic links. Writing to Neo4j...")
    
    # 5. 批量写入 Neo4j
    force_update_neo4j_relations(new_relations, graph_store)
    print("✅ Semantic Bridges created successfully.\n")

def build_graph_index():
    if os.path.exists(STORAGE_DIR):
        try:
            shutil.rmtree(STORAGE_DIR)
        except:
            pass

    print("Initializing storage context...")
    storage_context = initialize_storage()
    graph_store = storage_context.property_graph_store
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

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

    # 去重: 因为角色节点可能在多个文件中被重复生成，这里按 id_ 唯一化
    unique_nodes_dict = {n.node_id: n for n in all_nodes}
    all_nodes = list(unique_nodes_dict.values())

    print(f"Total nodes (after deduplication): {len(all_nodes)}")
    
    print("Generating embeddings manually...")
    text_to_embed = [n.metadata.get("_embed_content", n.text) for n in all_nodes]
    embeddings = embed_model.get_text_embedding_batch(text_to_embed)
    for node, emb in zip(all_nodes, embeddings):
        node.embedding = emb
    
    # save to Vector Store
    print("1. Adding nodes to Local Vector Store...")
    storage_context.vector_store.add(all_nodes)

    print("2. Force writing nodes to Neo4j...")
    force_update_neo4j_content(all_nodes, graph_store)
    
    if all_relations:
        print(f"3. Upserting {len(all_relations)} relations...")
        force_update_neo4j_relations(all_relations, graph_store)

    # # NEW: create semantic bridges
    # analyze_similarity_distribution(all_nodes)
    # create_semantic_bridges(all_nodes, graph_store)

    print("Persisting Vector Store to disk...")
    storage_context.persist(persist_dir=STORAGE_DIR)
    
    print("="*40)
    print(f"✅ BUILD SUCCESS. Nodes in VectorStore: {len(storage_context.vector_store._data.embedding_dict)}")
    print("="*40)

if __name__ == "__main__":
    build_graph_index()