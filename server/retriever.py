import os
import json
from llama_index.core import PropertyGraphIndex, StorageContext, PromptTemplate
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.graph_stores import SimplePropertyGraphStore

# --- 配置 ---
os.environ["OPENAI_API_KEY"] = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
os.environ["OPENAI_API_BASE"] = "https://poloai.top/v1"

llm = OpenAI(model="gpt-4o")

STORAGE_DIR = "./graph_storage"
GRAPH_STORE_PATH = os.path.join(STORAGE_DIR, "property_graph_store.json")

def load_or_create_index():
    """加载索引（保持不变）"""
    if os.path.exists(GRAPH_STORE_PATH):
        print(f"Loading existing graph store from {GRAPH_STORE_PATH}...")
        graph_store = SimplePropertyGraphStore.from_persist_path(GRAPH_STORE_PATH)
        
        if os.path.exists(os.path.join(STORAGE_DIR, "docstore.json")):
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            storage_context.property_graph_store = graph_store
        else:
            storage_context = StorageContext.from_defaults()
            
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            storage_context=storage_context,
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
            llm=OpenAI(model="gpt-4o"),
        )
        return index
    else:
        print("No index found, running builder...")
        from json_parser import build_graph_index
        return build_graph_index()

def create_retriever(index):
    """创建混合检索器"""
    sub_retrievers = [
        VectorContextRetriever(
            index.property_graph_store, 
            vector_store=index.vector_store,
            similarity_top_k=10# 可以调整这个数字来获取更多上下文
        ),
        LLMSynonymRetriever(
            index.property_graph_store, 
            llm=OpenAI(model="gpt-4o"),
        ),
    ]
    retriever = PGRetriever(
        sub_retrievers=sub_retrievers,
        use_vector_index=True, 
    )
    return retriever

# --- 新增：核心可视化函数 ---
def display_retrieval_path(nodes, index):
    """
    打印检索到的节点内容，并查找这些节点在图中的关系（路径）
    针对 SimplePropertyGraphStore 的直接内存访问修复版
    """
    print("\n" + "="*50)
    print(f"🔍 RETRIEVAL DEBUG INFO: Found {len(nodes)} relevant nodes")
    print("="*50)

    # 【核心修复】：直接访问 SimpleStore 的内部字典 _relations
    # 这是一个字典: {relation_id: RelationObject}
    try:
        # 尝试直接访问内部存储（最稳妥的方式用于调试）
        all_relations_dict = index.property_graph_store._relations
    except AttributeError:
        print("⚠️ Warning: Could not access internal relation store. Skipping path display.")
        all_relations_dict = {}

    for i, node in enumerate(nodes):
        print(f"\n[Node {i+1}] ID: {node.node_id}")
        print(f"Score: {node.score:.4f}") 
        print("-" * 20)
        # 打印内容片段
        print(f"📄 Content: {node.text}")
        
        print("-" * 20)
        print("🔗 Graph Connections (The Path):")
        
        # 筛选：在所有关系中，找出 source_id 等于当前节点 ID 的关系
        # values() 返回的是 Relation 对象列表
        found_relations = [
            rel for rel in all_relations_dict.values() 
            if rel.source_id == node.node_id
        ]
        
        if not found_relations:
            print("   (No direct outgoing edges found in graph store for this node)")
        
        for rel in found_relations:
            # 打印格式: [当前节点] --(关系)--> [目标节点]
            print(f"   🚀 {rel.source_id} --[{rel.label}]--> {rel.target_id}")
            
            # 打印属性（如果存在且非空）
            if hasattr(rel, 'properties') and rel.properties:
                # 过滤掉空值
                props = {k: v for k, v in rel.properties.items() if v}
                if props:
                    print(f"      (Props: {props})")

    print("\n" + "="*50 + "\n")


def extract_subgraph_context(retrieved_nodes, index):
    """
    健壮版：尝试从内存读取，失败则自动解析 JSON 文件
    """
    subgraph_triplets = []
    retrieved_ids = {n.node_id for n in retrieved_nodes}
    
    # 容器用于存放所有的关系对象（或字典）
    relations_source = []

    # --- 尝试 1: 内存访问 (可能失败) ---
    try:
        # 不同的版本可能使用 _relations, relations, 或者 _data
        if hasattr(index.property_graph_store, "_relations"):
            relations_source = index.property_graph_store._relations.values()
        elif hasattr(index.property_graph_store, "relations"):
            relations_source = index.property_graph_store.relations.values()
        else:
            raise AttributeError("In-memory relations not accessible")
        print("✅ Success: Accessed relations from memory.")
        
    except AttributeError:
        # --- 尝试 2: 文件回退 (通用且稳定) ---
        print("⚠️ Notice: Could not access in-memory store. Switching to JSON file parsing...")
        
        if not os.path.exists(GRAPH_STORE_PATH):
            print(f"❌ Error: Graph store file not found at {GRAPH_STORE_PATH}")
            return []
            
        try:
            with open(GRAPH_STORE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 调试：打印 JSON 的顶级键，帮助定位数据位置
            # print(f"DEBUG: JSON Keys found: {list(data.keys())}")
            
            # LlamaIndex 的存储结构通常在 'relations' 键下，或者是直接的字典
            raw_relations = {}
            if "relations" in data:
                raw_relations = data["relations"]
            elif "_relations" in data:
                raw_relations = data["_relations"]
            else:
                # 假设整个文件就是一个关系字典（不太常见，但也可能）
                # 或者它可能在更深层，这里做一个简单的启发式搜索
                for key, val in data.items():
                    if isinstance(val, dict) and len(val) > 0:
                        first_val = next(iter(val.values()))
                        if isinstance(first_val, dict) and "source_id" in first_val:
                            raw_relations = val
                            break
            
            # 将 JSON 字典转换为列表供后续处理
            # 注意：从 JSON 读出的都是字典，不是对象，所以下面访问属性要用 dict['key']
            relations_source = raw_relations.values()
            print(f"✅ Success: Loaded {len(relations_source)} relations from JSON file.")
            
        except Exception as e:
            print(f"❌ Error parsing JSON file: {e}")
            return []

    # --- 统一处理逻辑 (兼容对象属性访问和字典键值访问) ---
    print(f"🔍 Filtering {len(relations_source)} total relations for {len(retrieved_ids)} retrieved nodes...")
    
    for rel in relations_source:
        # 兼容性处理：rel 可能是对象(内存)也可能是字典(JSON)
        def get_attr(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        src_id = get_attr(rel, "source_id")
        tgt_id = get_attr(rel, "target_id")
        label = get_attr(rel, "label")
        properties = get_attr(rel, "properties") or {}
        
        # 核心筛选：双向匹配
        if (src_id in retrieved_ids) or (tgt_id in retrieved_ids):
            meaning = properties.get('meaning', label)
            # 构造三元组字符串
            triplet_str = f"Node({src_id}) --[{meaning}]--> Node({tgt_id})"
            subgraph_triplets.append(triplet_str)

    # 去重并返回
    unique_triplets = list(set(subgraph_triplets))
    print(f"🎯 Final Result: Found {len(unique_triplets)} relevant triplets.")
    
    return unique_triplets
            

def generate_diagram(query, subgraph_context, llm):
    context_str = "\n".join(subgraph_context)
    MerMaid_prompt = f"""
    You are an expert technical documentation assistant capable of generating Mermaid.js code.

    Your task is to generate a **Mermaid Flowchart** based ONLY on the provided graph connections.
    The user has asked a query, and I have retrieved a sub-graph of relevant connections.

    **User Query:** {query}

    **Retrieved Graph Connections:**
    {context_str}

    **Instructions:**
    1. Analyze the connections to understand the data flow.
    2. Construct a valid Mermaid.js `graph TD` code block.
    3. Use the Node IDs from the connections as the Mermaid node IDs.
    4. Put the Relation/Edge description in the connection label (e.g., `A -->|Description| B`).
    5. **CRITICAL:** Return ONLY the mermaid code block wrapped in markdown tags. Do not explain the code.
    6. If the graph is disconnected, show all retrieved parts.
    7. Style the nodes simply.

    **Output Format:**
    ```mermaid
    graph TD
        A[Node Name] -->|Relation| B[Node Name]
    """
    response = llm.complete(MerMaid_prompt)
    return response


if __name__ == "__main__":
    # 1. 获取索引
    try:
        index = load_or_create_index()
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Error loading index: {e}")
        exit(1)

    # 2. 创建检索器
    retriever = create_retriever(index)

    # 3. 定义查询
    query = "make a flowchart that using LLM for pysical exercises education in virtual reality"
    print(f"\nUser Query: {query}")

    # 显式执行检索 (Retrieve)
    # 返回 NodeWithScore 对象列表，包含找到的节点
    retrieved_nodes = retriever.retrieve(query)

    print(f"Raw Retrieved Nodes {len(retrieved_nodes)}")
    for i, node in enumerate(retrieved_nodes):
        print(f"Node {i+1}: {node.node_id} | Score: {node.score}")
        print(f"Source: {node.metadata.get('file_source','Unknown')}") 
        print(f"Text: {node.text}")   
        
    subgraph_context = extract_subgraph_context(retrieved_nodes, index)
    print(subgraph_context)

    if subgraph_context:
        mermaid_code = generate_diagram(query, subgraph_context, llm)
        print(mermaid_code)
    else:
        print("No subgraph context found.")