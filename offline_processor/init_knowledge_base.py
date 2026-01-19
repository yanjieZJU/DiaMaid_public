import subprocess
import os
import sys
import time

def run_step(script_name, description):
    print(f"\n{'='*60}")
    print(f"🚀 [Step: {description}]")
    print(f"   Running: offline_processor/{script_name}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join("offline_processor", script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: Script {script_path} not found. Make sure you are in the project root.")
        return False
        
    try:
        # Use the same python interpreter
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], check=True)
        duration = time.time() - start_time
        
        print(f"\n✅ Finished {script_name} in {duration:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed executing {script_name}. Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    # Ensure correct CWD
    if not os.path.exists("offline_processor"):
        print("❌ Error: 'offline_processor' directory not found.")
        print("   Please run this script from the PROJECT ROOT directory.")
        print("   Example: python offline_processor/init_knowledge_base.py")
        sys.exit(1)

    print("🤖 Starting Knowledge Base Initialization Pipeline...")
    total_start = time.time()

    # Step 1: PDF Processing (Extract Structure)
    # Extracts text/JSON from new PDFs in ./case_llm_artifact
    if not run_step("pdfReader.py", "PDF Parsing & Extraction"):
        sys.exit(1)

    # Step 2: Graph Construction (JSON -> Neo4j)
    # Clears Neo4j and rebuilds the graph + Basic Vector Store
    if not run_step("json_parser_neo4j.py", "Graph Database Construction"):
        sys.exit(1)

    # Step 3: Graph Clustering (RAG)
    # Runs Leiden clustering and generates community summaries
    if not run_step("graphrag_clustering.py", "GraphRAG Clustering"):
        sys.exit(1)

    # Step 4: Workflow Analysis (Topology)
    # Analyzes paths, common patterns, and scenario clusters
    if not run_step("workflow_analyzer.py", "System Topology Analysis"):
        sys.exit(1)

    total_duration = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ All Systems Operational! (Total Time: {total_duration:.2f}s)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
