import os
import httpx
import json
import sys
import os
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# --- 配置 ---
http_client = httpx.Client(proxy=None, trust_env=False)

llm = OpenAI(model="gpt-4o", http_client=http_client)

STORAGE_DIR = "./server/graph_storage"


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

def custom_retrieve(query_text, vector_store, graph_store, top_k, node_type=None, role_filter=None, community_id=None):
    print(f"\n🔍 Executing Custom Retrieval for: '{query_text}'")
    embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_BASE_URL")
    )
    query_embedding = embed_model.get_query_embedding(query_text)
    
    found_ids = []
    found_scores = []
    
    # CASE A: Structure-Based Retrieval (Community ID provided)
    if community_id:
        print(f"   🎯 Targeted Retrieval in Community: {community_id}")
        
        def _fetch_community_candidates(tx, comm_id, role_filter):
            cypher = """
            MATCH (n:LlamaNode {community_L0: $comm_id})
            WHERE n.type = 'MODULE'
            AND ($role_filter IS NULL OR replace(toLower(n.role), ' ', '') CONTAINS replace(toLower($role_filter), ' ', ''))
            RETURN n.id as id
            """
            result = tx.run(cypher, comm_id=comm_id, role_filter=role_filter)
            return [r["id"] for r in result]

        with graph_store._driver.session() as session:
            # Fetch all candidate IDs in this community
            candidate_ids = session.execute_read(_fetch_community_candidates, community_id, role_filter)
            
        print(f"      Found {len(candidate_ids)} candidates in graph.")
        
        if candidate_ids:
            # We use the VectorStore query but with a "Pre-Filter" via scoring logic manually
            # Since we can't easily pass IDs to a filter in SimpleVectorStore without custom logic
            # We will Query Global Top-100 and intersect with our Community IDs.
            
            query_obj = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=100, # Broad search
                filters=MetadataFilters(filters=[ExactMatchFilter(key="type", value=node_type)]) if node_type else None
            )
            vec_result = vector_store.query(query_obj)
            
            # Intersect
            id_score_map = {id_: score for id_, score in zip(vec_result.ids, vec_result.similarities)}
            
            scored_candidates = []
            for cid in candidate_ids:
                if cid in id_score_map:
                    scored_candidates.append((cid, id_score_map[cid] * 1.5)) # Boost scores heavily
                else:
                    # If it's in the community but not in global vector top-100, it might still be relevant contextually
                    # We give it a base score to allow it to appear if we have few results
                    scored_candidates.append((cid, 0.4)) 
            
            # Sort
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            found_ids = [x[0] for x in scored_candidates]
            found_scores = [x[1] for x in scored_candidates]
            
        else:
            print("   ⚠️ Community empty or role mismatch. Fallback to global search.")
            community_id = None # Trigger fallback

    # CASE B: Global Vector Search (Fallback or Default)
    if not community_id:
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

    def _fetch_node_details(tx, node_id, role_filter, skip_role):
        cypher = """
        MATCH (center:LlamaNode) WHERE center.id = $id 
        // Role Filtering (ignore case and spaces) - ONLY apply if not already filtered by community logic
        AND ($skip_role_check = true OR $role_filter IS NULL OR replace(toLower(center.role), ' ', '') CONTAINS replace(toLower($role_filter), ' ', ''))
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
            center.community_L0 as comm_id,
            center.summary_L0 as comm_summary,
            upstreams,
            downstreams,
            dependencies
        """
        return tx.run(cypher, id=node_id, role_filter=role_filter, skip_role_check=skip_role).single()

    with graph_store._driver.session() as session:
        for node_id, score in zip(found_ids, found_scores):
            # Stop if we have enough
            if len(retrieved_nodes) >= top_k:
                break
            
            # specific role check skip if we already filtered via community query
            skip_role = True if community_id else False
            
            result = session.execute_read(_fetch_node_details, node_id, role_filter, skip_role)
            
            if result:
                center_node = result["center"]
                source_file = result.get("source_file")
                upstreams = result["upstreams"]
                downstreams = result["downstreams"]
                dependencies = result["dependencies"]
                comm_summary = result.get("comm_summary")
                
                node_type = center_node.get("type")
                node = TextNode(text=center_node.get("text", ''), id_=center_node["id"])
                node.metadata = dict(center_node)
                node.metadata["topology"] = {
                    "upstream": upstreams,
                    "downstream": downstreams,
                    "dependencies": dependencies,
                    "source_file": source_file,
                }
                if comm_summary:
                    node.metadata["community_summary"] = comm_summary

                retrieved_nodes.append(NodeWithScore(node=node, score=score))

            else:
                 pass # Already handled or filtered

    return retrieved_nodes

def get_all_community_summaries(vector_store, graph_store):
    """
    Retrieve all available high-level community summaries to inform the Planner.
    """
    print("🌍 Fetching Community Capabilities...")
    # Since we can't easily 'scan' vector store, we use a simple Cypher query 
    # assuming we wrote summaries back to Module nodes.
    cypher = """
    MATCH (n:LlamaNode) 
    WHERE n.community_L0 IS NOT NULL
    RETURN DISTINCT n.community_L0 as id, n.summary_L0 as summary
    LIMIT 50
    """
    summaries = []
    
    def _fetch_summaries(tx):
        result = tx.run(cypher)
        return [f"[Community ID: {record['id']}]: {record['summary']}" for record in result]

    with graph_store._driver.session() as session:
        summaries = session.execute_read(_fetch_summaries)
    return "\n".join(summaries)

def retrieve_and_get_content(query, vector_store, graph_store, top_k=5, node_type=None, role_filter=None, community_id=None, include_topology=True):
    if vector_store is None or graph_store is None:
        vector_store, graph_store = load_stores()

    retrieved_nodes = custom_retrieve(query, vector_store, graph_store, top_k, node_type, role_filter, community_id)

    processed_nodes = []
    for node_with_score in retrieved_nodes:
        node = node_with_score.node
        score = node_with_score.score
        n_type = node.metadata.get("type") # Use n_type to avoid confusion with argument node_type

        enriched_text = ""
        topology_text = ""
        
        if include_topology:
            topology = node.metadata.get("topology", {})
            upstreams = topology.get("upstream", [])
            downstreams = topology.get("downstream", [])
            # Only include if not empty
            if upstreams or downstreams:
                topology_text = (
                    f"--- Context ---\n"
                    f"Proceded By: {[u['name'] for u in upstreams]}\n"
                    f"Followed By: {[d['name'] for d in downstreams]}\n"
                )

        if n_type == "WORKFLOW":
            topology = node.metadata.get("topology", {})
            dependencies = topology.get("dependencies", [])
            enriched_text = (
                f"Workflow: {node.metadata.get('name')}\n"
                f"Description: {node.metadata.get('text')}\n"
                f"Source: {node.metadata.get('source')}\n"
                f"--- Dependencies ---\n"
                f"Depends on {[d['name'] for d in dependencies if 'name' in d]}"
            )
        elif n_type == "MODULE":
            # Incorporate Community Context if available
            comm_summary = node.metadata.get("community_summary", "")
            context_str = f"Community Context: {comm_summary}\n" if comm_summary else ""
            
            enriched_text = (
                f"Module: {node.metadata.get('name')}\n"
                f"Source: {node.metadata.get('source')}\n"
                f"{context_str}"
                f"Text content: {node.metadata.get('text')}\n"
                f"{topology_text}"
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
        WHERE any(r in relationships(p) WHERE type(r) = 'FLOWS_TO')
        RETURN
            [n in nodes(p) | {id: n.id, name: n.name, text: n.text, function: n.function}] as path_nodes,
            [r in relationships(p) | type(r)] as relations,
            [r in relationships(p) | properties(r)] as rel_props,
            length(p) as len
        ORDER BY len ASC
        LIMIT 10
        """
        def _find_paths(tx, src_ids, tgt_ids):
            result = tx.run(cypher, src_ids=src_ids, tgt_ids=tgt_ids)
            return list(result)

        with graph_store._driver.session() as session:
            records = session.execute_read(_find_paths, src_ids, tgt_ids)
            unique_signatures = set()
            
            for record in records:
                path_nodes = record["path_nodes"]
                relations = record["relations"]
                
                # Generate a signature based on the actual data flow structure
                # This dedups paths that only differ by their entry point (via HAS_ROLE)
                flow_signature_parts = []
                for i, rel in enumerate(relations):
                    if rel == "FLOWS_TO":
                         flow_signature_parts.append(f"{path_nodes[i]['name']}->{path_nodes[i+1]['name']}")
                
                signature = "|".join(flow_signature_parts)
                # If no flows_to (shouldn't happen due to WHERE), skip this path
                if not signature:
                    continue
                if signature in unique_signatures:
                    continue
                unique_signatures.add(signature)
                
                if len(found_paths) >= 3:
                    break

                path_data = {
                    "source_query": source_text,
                    "target_query": target_text,
                    "path_nodes": record["path_nodes"],
                    "relations": record["relations"],
                    "rel_props": record["rel_props"]
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