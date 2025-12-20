import os
import httpx
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# --- 配置 ---
os.environ["OPENAI_API_KEY"] = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
os.environ["OPENAI_API_BASE"] = "https://poloai.top/v1"
http_client = httpx.Client(proxy=None, trust_env=False)

llm = OpenAI(model="gpt-4o", http_client=http_client)

STORAGE_DIR = "./server/graph_storage"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "BQLjYc4gOpIxTi6o5c-es481vKejEHHBGuxJeEvX9hg"
NEO4J_URL = "neo4j+s://8891a7fc.databases.neo4j.io"

def load_stores():
    print("Connecting to Stores...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
    )
    try:
        vector_store = SimpleVectorStore.from_persist_dir(persist_dir=STORAGE_DIR)
        print(f"✅ Loaded VectorStore with {len(vector_store._data.embedding_dict)} nodes.")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        exit(1)
    return vector_store, graph_store

def custom_retrieve(query_text, vector_store, graph_store, top_k=5, in_type=None, out_type=None):
    print(f"\n🔍 Executing Custom Retrieval for: '{query_text}'")
    
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    query_embedding = embed_model.get_query_embedding(query_text)

    filter_list = [ExactMatchFilter(key="type", value="MODULE")]
    # if in_type:
    #     filter_list.append(ExactMatchFilter(key="input_type", value=in_type))
    # if out_type: 
    #     filter_list.append(ExactMatchFilter(key="output_type", value=out_type))

    filters = MetadataFilters(filters=filter_list)
    
    query_obj = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=top_k,
        filters=filters
    )
    
    vec_result = vector_store.query(query_obj)
    found_ids = vec_result.ids
    found_scores = vec_result.similarities
    
    print(f"   -> Vector Search found {len(found_ids)} candidate IDs.")
    
    retrieved_nodes = []
    with graph_store._driver.session() as session:
        for node_id, score in zip(found_ids, found_scores):
            # 修正查询: 匹配任何标签，只要ID对上
            cypher = """
            MATCH (n:LlamaNode) WHERE n.id = $id 
            RETURN n.text as text, n.name as name, n.id as id, n.code as code, n.node_type as node_type, properties(n) as props
            """
            result = session.run(cypher, id=node_id).single()
            
            if result:
                text_content = result["text"]
                name = result["name"]
                code = result["code"]
                node_type = result["node_type"]
                
                # [DEBUG] 如果内容还是空，打印所有属性看看
                if not text_content:
                    print(f"   ⚠️ Found Node {node_id}, but 'text_content' is missing/empty.")
                    print(f"      Available properties: {result['props'].keys()}")
                    text_content = str(result['props'])
                
                # Reconstruct node
                node = TextNode(text=text_content, id_=result["id"])
                node.metadata["name"] = name
                node.metadata["type"] = node_type
                node.metadata["code"] = code
                retrieved_nodes.append(NodeWithScore(node=node, score=score))
            else:
                print(f"   ❌ ID {node_id} NOT FOUND in Neo4j. (This should not happen with MERGE)")

    return retrieved_nodes

def retrieve_and_get_content(query, vector_store=None, graph_store=None, top_k=5, in_type=None, out_type=None):
    if vector_store is None or graph_store is None:
        vector_store, graph_store = load_stores()

    retrieved_nodes = custom_retrieve(query, vector_store, graph_store, top_k, in_type, out_type)
    return retrieved_nodes


if __name__ == "__main__":
    vector_store, graph_store = load_stores()
    query = "Module: Content Generation Module\n Function: Creates accurate, comprehensive, and contextually rich history content tailored to the user's level, interests, and learning goals"
    
    retrieved_nodes = custom_retrieve(query, vector_store, graph_store)

    print("\n" + "="*50)
    print(f"✅ Found {len(retrieved_nodes)} nodes with content.")
    for n in retrieved_nodes:
        code_snippet = n.node.metadata.get("code", "")
        if code_snippet:
            code_snippet = code_snippet[:50].replace("\n", " ")
        else:
            code_snippet = "N/A"
        print(f"ID: {n.node.node_id} | Code: {code_snippet}...")
    print("="*50 + "\n")
