import os
import httpx
import json
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

def custom_retrieve(query_text, vector_store, graph_store, top_k=10, node_type=None, role_filter=None):
    print(f"\n🔍 Executing Custom Retrieval for: '{query_text}'")
    
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    query_embedding = embed_model.get_query_embedding(query_text)

    filter_list = []
    if node_type:
        filter_list.append(ExactMatchFilter(key="type", value=node_type))

    filters = MetadataFilters(filters=filter_list) if filter_list else None
    
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
            # Stop if we have enough
            if len(retrieved_nodes) >= top_k:
                break

            cypher = """
            MATCH (center:LlamaNode) WHERE center.id = $id 
            // Role Filtering
            AND ($role_filter IS NULL OR center.role CONTAINS $role_filter)
            WITH center
            OPTIONAL MATCH (center)-[r:FLOWS_TO]->(next:LlamaNode)
            WITH center, collect(distinct {name: next.name, id: next.id, function: next.text, code: next.code, edge_label:r.edge_label, properties: properties(r)}) as downstreams
            OPTIONAL MATCH (prev:LlamaNode)-[r:FLOWS_TO]->(center)
            WITH center, downstreams, collect(distinct {name: prev.name, id: prev.id, function: prev.text, code: prev.code, edge_label:r.edge_label, properties: properties(r)}) as upstreams   
            // 如果是WORKFLOW节点，获取依赖的工作流
            OPTIONAL MATCH (center)-[:DEPENDS_ON]->(dep_wf:LlamaNode)
            WITH center, upstreams, downstreams, collect(distinct {name: dep_wf.name, id: dep_wf.id, function: dep_wf.text, code: dep_wf.code}) as dependencies
            // 获取源文件信息
            RETURN 
                center, 
                center.source as source_file,
                upstreams,
                downstreams,
                dependencies
            """
            result = session.run(cypher, id=node_id, role_filter=role_filter).single()
            
            if result:
                center_node = result["center"]
                source_file = result.get("source_file")
                upstreams = result["upstreams"]
                downstreams = result["downstreams"]
                dependencies = result["dependencies"]
                
                node_type = center_node.get("type")
                node = TextNode(text=center_node.get("text", ''), id_=center_node["id"])
                node.metadata = dict(center_node)
                node.metadata["topology"] = {
                    "upstream": upstreams,
                    "downstream": downstreams,
                    "dependencies": dependencies,
                    "source_file": source_file,
                }

                retrieved_nodes.append(NodeWithScore(node=node, score=score))

            else:
                if role_filter:
                    print(f"   ⚠️  ID {node_id} filtered out (Role mismatch). Required: {role_filter}")
                else: 
                    print(f"   ❌ ID {node_id} NOT FOUND in Neo4j. (This should not happen with MERGE)")

    return retrieved_nodes

def retrieve_and_get_content(query, vector_store=None, graph_store=None, top_k=5, node_type=None, role_filter=None):
    if vector_store is None or graph_store is None:
        vector_store, graph_store = load_stores()

    retrieved_nodes = custom_retrieve(query, vector_store, graph_store, top_k, node_type, role_filter)
    processed_nodes = []
    for node_with_score in retrieved_nodes:
        node = node_with_score.node
        score = node_with_score.score
        node_type = node.metadata.get("type")

        if node_type == "WORKFLOW":
            topology = node.metadata.get("topology")
            enriched_text = (
                f"Workflow: {node.metadata.get('name')}\n"
                f"Description: {node.metadata.get('text')}\n"
                f"Source: {node.metadata.get('source')}\n"
                f"--- Dependencies ---\n"
                f"Depends on {[d['name'] for d in topology.get("dependencies", []) if d['name']]}"
            )
        elif node_type == "MODULE":
            topology = node.metadata.get("topology")
            upstreams = topology.get("upstream", [])
            downstreams = topology.get("downstream", [])
            enriched_text = (
                f"Module: {node.metadata.get('name')}\n"
                f"Source: {node.metadata.get('source')}\n"
                f"Text content: {node.metadata.get('text')}\n"
                f"--- Context ---\n"
                f"Proceded By: {[u['name'] for u in upstreams]}\n"
                f"Followed By: {[d['name'] for d in downstreams]}\n"
            )
        else:
            enriched_text = (
                f"Article: {node.metadata.get('name')}\n"
                f"Text Conent: {node.metadata.get('text')}\n"
            )
        # Reconstruct node
        new_node = TextNode(text=enriched_text, id_=node.node_id)
        new_node.metadata = node.metadata
        processed_nodes.append(NodeWithScore(node=new_node, score=score))

    return processed_nodes

def find_structural_path(source_text, target_text, vector_store, graph_store, rel_type="DATA_FLOW", top_k=5):
    """
    Find the structural path between two nodes in the graph.
    """
    src_nodes = custom_retrieve(source_text, vector_store, graph_store, top_k, node_type="MODULE")
    src_ids = [node.node.id_ for node in src_nodes]
    tgt_nodes = custom_retrieve(target_text, vector_store, graph_store, top_k, node_type="MODULE")
    tgt_ids = [node.node.id_ for node in tgt_nodes]
    if not src_ids or not tgt_ids:
        return []
    found_paths = []

    if rel_type == "CONTAINS":        
        pass


    else:
        cypher = """
        MATCH (s:LlamaNode), (t:LlamaNode)
        WHERE s.id IN $src_ids AND t.id IN $tgt_ids
        AND s.id <> t.id
        MATCH p = shortestPath((s)-[*1..5]-(t))
        RETURN
            [n in nodes(p) | {id: n.id, name: n.name, text: n.text}] as path_nodes,
            [r in relationships(p) | type(r)] as relations,
            length(p) as len
        ORDER BY len ASC
        LIMIT 3
        """
        with graph_store.client.session() as session:
            result = session.run(cypher, src_ids=src_ids, tgt_ids=tgt_ids)
            for record in result:
                path_data = {
                    "source_query": source_text,
                    "target_query": target_text,
                    "path_nodes": record["path_nodes"],
                    "relations": record["relations"]
                }
                found_paths.append(path_data)

                node_names = "-> ".join([node["name"] for node in record["path_nodes"]])
                print(f"   ✅ Path found: {node_names}")

    

    return found_paths

if __name__ == "__main__":
    vector_store, graph_store = load_stores()
    query = "The query is routed to the appropriate category manager, which then provides an answer using its knowledge base. If the query doesn't fit a category, a fine-tuned model generates the answer."
    
    # retrieved_nodes = retrieve_and_get_content(query, vector_store, graph_store, top_k=5, node_type="WORKFLOW", in_type=None)

    # print("\n" + "="*50)
    # print(f"✅ Found {len(retrieved_nodes)} nodes with content.")
    # for n in retrieved_nodes:
    #     print(n.node.text)
    #     print(n.node.metadata.get("topology", "N/A"))
    #     print(n.score)
    # print("="*50 + "\n")

    paths = find_structural_path("detector", "facial recognition", vector_store, graph_store, rel_type="CONTAINS", top_k=5)
    print(paths)