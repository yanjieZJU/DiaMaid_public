import os
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 配置 ---
os.environ["OPENAI_API_KEY"] = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
os.environ["OPENAI_API_BASE"] = "https://poloai.top/v1"
STORAGE_DIR = "./graph_storage"

def run_diagnostic():
    print("--- DIAGNOSTIC RETRIEVAL TEST ---")
    
    # 1. 加载向量库
    try:
        vector_store = SimpleVectorStore.from_persist_dir(persist_dir=STORAGE_DIR)
        print(f"✅ Vector Store loaded from disk.")
        # 修正：打印嵌入字典的大小
        count = len(vector_store._data.embedding_dict)
        print(f"📊 Node count in store: {count}")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return

    # 2. 准备查询
    query_text = "history education"
    print(f"🔎 Querying for: '{query_text}'")
    
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    query_embedding = embed_model.get_query_embedding(query_text)
    
    # 3. 执行查询
    query_obj = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5
    )
    
    result = vector_store.query(query_obj)
    
    # 4. 打印结果
    print(f"\n✅ Found {len(result.ids)} results (IDs validated):")
    for i, (node_id, score) in enumerate(zip(result.ids, result.similarities)):
        print(f"[{i+1}] Score: {score:.4f} | ID: {node_id}")
        print(f"    (Text content resides in Neo4j)")

if __name__ == "__main__":
    run_diagnostic()