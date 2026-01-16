"""
Simple verification script to check GraphRAG clustering results and save to JSON
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.retriever_neo4j import load_stores

def verify_clustering():
    results = {}
    
    vector_store, graph_store = load_stores()
    
    with graph_store._driver.session() as session:
        # 1. Check SIMILAR_TO edges
        result = session.run("""
            MATCH ()-[r:SIMILAR_TO]->()
            RETURN count(r) as edge_count,
                   min(r.weight) as min_weight,
                   max(r.weight) as max_weight,
                   avg(r.weight) as avg_weight
        """).single()
        
        results['edges'] = {
            'total': result['edge_count'],
            'min_weight': float(result['min_weight']),
            'max_weight': float(result['max_weight']),
            'avg_weight': float(result['avg_weight'])
        }
        
        # 2. Check community assignments
        result = session.run("""
            MATCH (n:LlamaNode)
            WHERE n.community_id IS NOT NULL
            RETURN count(n) as nodes_with_community
        """).single()
        
        results['nodes_with_community'] = result['nodes_with_community']
        
        # 3. Community distribution
        result = session.run("""
            MATCH (n:LlamaNode)
            WHERE n.community_id IS NOT NULL
            RETURN n.community_id as comm_id, count(n) as size
            ORDER BY size DESC
        """)
        
        communities = []
        for record in result:
            communities.append({
                'id': record['comm_id'],
                'size': record['size']
            })
        
        results['communities'] = {
            'total_count': len(communities),
            'distribution': communities
        }
        
        # 4. Get community summaries
        result = session.run("""
            MATCH (n:LlamaNode)
            WHERE n.community_id IS NOT NULL AND n.community_summary IS NOT NULL
            RETURN DISTINCT n.community_id as comm_id, n.community_summary as summary
        """)
        
        summaries = []
        for record in result:
            summaries.append({
                'community_id': record['comm_id'],
                'summary': record['summary']
            })
        
        results['summaries'] = summaries
        
        # 5. Sample nodes from top 3 communities
        top_communities = communities[:3]
        community_samples = []
        
        for comm in top_communities:
            result = session.run("""
                MATCH (n:LlamaNode {community_id: $comm_id})
                RETURN n.name as name, n.role as role
                LIMIT 5
            """, comm_id=comm['id'])
            
            nodes = []
            for record in result:
                nodes.append({
                    'name': record['name'],
                    'role': record['role']
                })
            
            community_samples.append({
                'community_id': comm['id'],
                'size': comm['size'],
                'sample_nodes': nodes
            })
        
        results['sample_communities'] = community_samples
    
    # Save to JSON
    output_file = 'clustering_verification_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("GraphRAG Clustering Verification Summary")
    print("="*60)
    print(f"Total SIMILAR_TO edges: {results['edges']['total']}")
    print(f"Nodes with community assignment: {results['nodes_with_community']}")
    print(f"Total communities: {results['communities']['total_count']}")
    print(f"Summaries generated: {len(results['summaries'])}")
    print("="*60)


if __name__ == "__main__":
    verify_clustering()
