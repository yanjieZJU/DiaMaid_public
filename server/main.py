import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
# from agents import run_pipeline # Removed
from pipeline_runner import run_pipeline # Updated Import
import sys
# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

app = FastAPI()

# --- CORS Setting ---
origins = [
    "http://localhost:5173",
    "[http://127.0.0.1:5173](http://127.0.0.1:5173)",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM Client Setting ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

# --- 3. 定义请求数据格式 ---
class GenRequest(BaseModel):
    prompt: str          # 用户的总体描述
    userNodes: list[str] # 用户定义的节点列表 (现在可能包含更复杂的描述字符串)
    allowRandomNodes: bool = False # 是否允许生成随机节点 (深度生成模式)

# --- 4. 核心接口 ---
@app.post("/generate")
async def generate_mermaid(req: GenRequest):
    print(f"收到请求: 描述={req.prompt}, 允许随机节点={req.allowRandomNodes}")
    # 打印前几个节点看看格式
    if req.userNodes:
        print(f"节点示例: {req.userNodes[:3]}...")
    
    # 构造 Prompt
    # 技巧：强制要求 LLM 返回纯文本，不要 markdown 格式
    system_prompt = """
    你是一个 Mermaid 流程图专家。
    任务：根据用户的描述和提供的节点列表，生成 Mermaid 代码。
    
    输入说明：
    用户提供的节点列表 (Nodes should be included) 可能包含以下几种格式：
    1. 纯节点名称： "NodeA"
    2. 连线关系： "NodeA --> NodeB"
    3. 包含关系： "ParentNode contains ChildNode"
    
    要求：
    1. 必须包含用户提供的所有节点文本。
    2. 必须严格遵守用户定义的连线关系（如 A --> B）。
    3. 必须严格遵守用户定义的包含关系（使用 subgraph）。
       例如：如果输入 "GroupA contains NodeB"，则生成的 Mermaid 代码应类似：
       subgraph GroupA
         NodeB
       end
    4. [CRITICAL] 节点生成约束：
       - 如果 allowRandomNodes 为 False (默认)：生成的流程图中只能包含 'userNodes' 中的节点，严禁添加任何额外的、用户未指定的节点。
       - 如果 allowRandomNodes 为 True (深度生成)：你可以在保留所有 'userNodes' 的基础上，根据 'user query' 的语境，智能添加必要的中间节点或辅助模块，以完善系统逻辑。
    5. 根据逻辑合理连接节点。
    6. [CRITICAL] 连线文字规范：
       - 严禁将连线说明（如“数据流”、“控制信号”）作为节点名称的一部分。
       - 连线说明必须放在 Mermaid 的连线语法中，例如 `A -->|说明文字| B`。
       - 节点名称必须保持简洁，只包含实体名（如 "User", "Server"），不要包含动作描述。
       - 如果用户在 prompt 中明确要求修改或隐藏连线文字，请遵照执行。
    7. 只返回 Mermaid 代码字符串，不要包含 ```mermaid 或 ``` 这样的 Markdown 标记。
    8. 默认使用 graph TD (从上到下) 布局。
    """

    user_message = f"""
    user query: {req.prompt}
    Nodes and Relations strictly required: 
    {json.dumps(req.userNodes, indent=2, ensure_ascii=False)}
    Allow Random/Extra Nodes: {req.allowRandomNodes}
    """
    
    # 注意：这里的 system_prompt 其实是在 main.py 里定义的，但实际上我们调用的是 run_pipeline
    # run_pipeline 内部有自己的 prompt 逻辑 (在 agents.py 中)
    # 我们需要把 allowRandomNodes 参数传递给 run_pipeline
    
    print(f"Calling run_pipeline with: {user_message}")
    try:
        results = run_pipeline(user_message)
        if results and len(results) > 0:
            # 暂时只取第一个生成的方案
            generated_code = results[0]["mermaid_code"]
            # 我们还需要返回每个节点的详细描述信息
            # results[0]["description"] 可能是整个架构的描述，我们需要解析出节点的描述
            # 实际上 integrator.integrate_arch 返回的是一个 JSON 结构的 architecture
            # 我们应该让 run_pipeline 返回这个原始 JSON 数据
            
            node_details = results[0].get("node_details", {})
            
            return {
                "code": generated_code,
                "node_details": node_details
            }
        else:
            raise HTTPException(status_code=500, detail="生成失败，未返回有效结果")

    except Exception as e:
        print(f"LLM 调用出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)