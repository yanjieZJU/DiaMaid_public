# server/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
from agents import run_pipeline

API_KEY = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
API_BASE = "https://poloai.top/v1"

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
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# --- 3. 定义请求数据格式 ---
class GenRequest(BaseModel):
    prompt: str          # 用户的总体描述
    userNodes: list[str] # 用户定义的节点列表

# --- 4. 核心接口 ---
@app.post("/generate")
async def generate_mermaid(req: GenRequest):
    print(f"收到请求: 描述={req.prompt}, 节点={req.userNodes}")
    
    # 构造 Prompt
    # 技巧：强制要求 LLM 返回纯文本，不要 markdown 格式
    system_prompt = """
    你是一个 Mermaid 流程图专家。
    任务：根据用户的描述和提供的节点列表，生成 Mermaid 代码。
    要求：
    1. 必须包含用户提供的所有节点文本。
    2. 根据逻辑合理连接这些节点。
    3. 只返回 Mermaid 代码字符串，不要包含 ```mermaid 或 ``` 这样的 Markdown 标记。
    4. 默认使用 graph TD (从上到下) 布局。
    """

    user_message = f"""
    user query: {req.prompt}
    Nodes should be included: {", ".join(req.userNodes)}
    """

    try:
        generated_code = run_pipeline(user_message)
        return {"code": generated_code}

    except Exception as e:
        print(f"LLM 调用出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)