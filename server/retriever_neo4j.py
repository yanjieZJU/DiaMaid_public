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

# Role Collaboration Graph: defines typical data flow patterns between role categories
# Used for far-hop retrieval to suggest complementary modules
ROLE_COLLABORATION = {
    "user_entity": ["sensing_input", "interface_output"],  # users interact with sensors and outputs
    "sensing_input": ["processing_core"],  # sensors feed data to processors
    "processing_core": ["interface_output", "infrastructure", "processing_core"],  # processors output results or feed other processors
    "interface_output": ["user_entity"],  # outputs reach users
    "infrastructure": ["processing_core", "sensing_input", "interface_output"]  # infrastructure supports all
}

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

def extract_role_category(role_str):
    """
    Extract category from role string (format: "category: subcategory")
    Returns: category string or None
    """
    if not role_str or ':' not in role_str:
        return None
    return role_str.split(':', 1)[0].strip()

def analyze_skeleton_roles(skeleton_node, graph_store):
    """
    Analyze the role distribution in a skeleton (WORKFLOW node).
    Returns a dict with role category counts and suggested complementary roles.
    
    Args:
        skeleton_node: NodeWithScore object containing a WORKFLOW node
        graph_store: Neo4j graph store
    
    Returns:
        {
            'role_counts': {'processing_core': 3, 'sensing_input': 1, ...},
            'suggested_roles': ['interface_output', ...],  # roles to complement
            'dominant_category': 'processing_core'
        }
    """
    print(f"\n📊 Analyzing skeleton role distribution for: {skeleton_node.node.metadata.get('name', 'Unknown')}")
    
    skeleton_id = skeleton_node.node.id_
    role_counts = {}
    module_roles = []
    
    # Query to get all modules in this workflow's module_seq
    cypher = """
    MATCH (wf:LlamaNode {id: $wf_id})-[:EXECUTES_STEP]->(mod:LlamaNode)
    WHERE mod.type = 'MODULE'
    RETURN mod.role as role, mod.name as name
    ORDER BY mod.name
    """
    
    with graph_store._driver.session() as session:
        result = session.run(cypher, wf_id=skeleton_id)
        for record in result:
            role = record.get("role", "")
            name = record.get("name", "")
            module_roles.append(role)
            
            category = extract_role_category(role)
            if category:
                role_counts[category] = role_counts.get(category, 0) + 1
                print(f"   - {name}: {role} → category: {category}")
    
    # Find dominant category
    dominant_category = max(role_counts.items(), key=lambda x: x[1])[0] if role_counts else None
    
    # Suggest complementary roles based on ROLE_COLLABORATION graph
    suggested_roles = []
    for category in role_counts.keys():
        if category in ROLE_COLLABORATION:
            for complementary in ROLE_COLLABORATION[category]:
                if complementary not in role_counts or role_counts[complementary] == 0:
                    if complementary not in suggested_roles:
                        suggested_roles.append(complementary)
    
    print(f"   ✓ Role distribution: {role_counts}")
    print(f"   ✓ Dominant category: {dominant_category}")
    print(f"   ✓ Suggested complementary roles: {suggested_roles}")
    
    return {
        'role_counts': role_counts,
        'suggested_roles': suggested_roles,
        'dominant_category': dominant_category,
        'all_module_roles': module_roles
    }

def retrieve_with_far_hop(anchors_list, vector_store, graph_store, skeleton_analysis=None, 
                          strategy="complementary", top_k=3, context=""):
    """
    Far-hop retrieval: supplement user's anchors with modules from related but different role categories.
    
    Args:
        anchors_list: list of anchor dicts with 'user_term', 'description', 'role'
        vector_store: vector store
        graph_store: graph store
        skeleton_analysis: optional dict from analyze_skeleton_roles()
        strategy: 'complementary' (based on skeleton) or 'diverse' (explore different categories)
        top_k: number of results per anchor
        context: (str) Domain context or original user query to make retrieval specific
    
    Returns:
        dict mapping anchor names to retrieved nodes, PLUS additional complementary modules
    """
    print(f"\n🔭 Far-hop Retrieval (strategy: {strategy})")
    
    anchor_mapping = {}
    user_role_categories = set()
    
    # First, do normal retrieval for user's anchors (近跳部分)
    for anchor in anchors_list:
        user_term = anchor["user_term"]
        description = anchor["description"]
        role = anchor.get("role", None)
        query_text = f"{user_term}: {description}"
        
        # Extract category from user's specified role
        if role:
            category = extract_role_category(role)
            if category:
                user_role_categories.add(category)
        
        retrieved_nodes = retrieve_and_get_content(
            query_text, 
            vector_store, 
            graph_store, 
            top_k,
            node_type="MODULE",
            role_filter=role
        )
        
        if retrieved_nodes:
            anchor_mapping[user_term] = retrieved_nodes
            print(f"   ✓ Retrieved {len(retrieved_nodes)} modules for anchor '{user_term}'")
    
    # Now, the FAR-HOP part: retrieve complementary modules
    complementary_roles = []
    
    if strategy == "complementary" and skeleton_analysis:
        # Use skeleton's suggested roles
        complementary_roles = skeleton_analysis.get('suggested_roles', [])
        print(f"   → Complementary strategy: using skeleton-suggested roles {complementary_roles}")
        
    elif strategy == "diverse":
        # Explore categories NOT in user's anchors
        all_categories = set(ROLE_COLLABORATION.keys())
        unexplored = all_categories - user_role_categories
        complementary_roles = list(unexplored)
        print(f"   → Diverse strategy: exploring unexplored categories {complementary_roles}")
    
    # Retrieve modules from complementary roles
    for comp_role_category in complementary_roles[:2]:  # Limit to 2 complementary categories
        # Create a specific query combining Domain Context + Target Role
        if context:
            query_text = f"{context} {comp_role_category.replace('_', ' ')}"
        else:
            query_text = f"module for {comp_role_category.replace('_', ' ')}"
            
        print(f"   🔍 Far-hop Query: '{query_text}'")

        retrieved_nodes = retrieve_and_get_content(
            query_text,
            vector_store,
            graph_store,
            top_k=2,  # Fewer results for补充 modules
            node_type="MODULE",
            role_filter=comp_role_category  # Filter by category (partial match in Cypher)
        )
        
        if retrieved_nodes:
            # Add as a "补充" anchor
            comp_anchor_name = f"[Complementary] {comp_role_category}"
            anchor_mapping[comp_anchor_name] = retrieved_nodes
            print(f"   ✓ Added {len(retrieved_nodes)} complementary modules for '{comp_role_category}'")
    
    return anchor_mapping


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