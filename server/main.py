import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
from pipeline_runner import run_pipeline, run_edit_pipeline # Updated Import
import sys
# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from exporter import export_to_docx

app = FastAPI()

# --- CORS Setting ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
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

from typing import Optional, Union # Add Optional and Union

# --- 3. 定义请求数据格式 ---
class AnchorDef(BaseModel):
    user_term: str
    description: str = ""
    role: str = ""

class RelationDef(BaseModel):
    source: str
    target: str
    type: str = "FLOWS_TO"

class GenRequest(BaseModel):
    session_id: Optional[str] = None # Added session_id
    prompt: str          # 用户的总体描述
    anchors: list[AnchorDef] = []
    relations: list[RelationDef] = []
    userNodes: list[str] = [] # 用户定义的纯文本节点列表 (作为后续预留或旧版本兼容)
    allowRandomNodes: bool = False # 是否允许生成随机节点

class EditRequest(BaseModel):
    session_id: Optional[str] = None # Added session_id
    plan_id: Optional[str] = None # Actually represents parent_node_id now
    current_plan: dict
    instruction: str
    original_query: str

class ExportRequest(BaseModel):
    image: str # Base64 encoded image string
    plan: Union[dict, list] # Module details json
    overall_description: Optional[str] = None

# --- 4. 核心接口 ---
@app.post("/generate")
async def generate_mermaid(req: GenRequest):
    print(f"收到请求: 描述={req.prompt}, 允许随机节点={req.allowRandomNodes}")
    # 打印前几个节点看看格式
    if req.userNodes:
        print(f"节点示例: {req.userNodes[:3]}...")
    
    print(f"Calling streaming run_pipeline with:")
    print(f"    user query: {req.prompt}")
    print(f"    Anchors: {len(req.anchors)}")
    print(f"    Relations: {len(req.relations)}")
    
    anchors_list = [a.model_dump() for a in req.anchors]
    relations_list = [r.model_dump() for r in req.relations]
    
    # 使用 StreamingResponse, media_type设为 ndjson 方便前端按行解析
    def event_generator():
        try:
            # 修改 num_outputs 增加生成方案数量，以流式逐步返回
            for chunk in run_pipeline(
                user_query=req.prompt,
                base_anchors=anchors_list,
                user_relations=relations_list,
                num_outputs=1, 
                divergence=True,
                session_id=req.session_id # Pass session_id
            ):
                yield chunk
        except Exception as e:
            print(f"Pipeline error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/edit")
async def edit_architecture(req: EditRequest):
    print(f"收到修改请求: 指令={req.instruction}, PlanID={req.plan_id}")
    
    def event_generator():
        try:
            # 使用 run_edit_pipeline 处理修改
            for chunk in run_edit_pipeline(
                plan_id=req.plan_id,
                current_plan_json=req.current_plan,
                user_instruction=req.instruction,
                original_query=req.original_query,
                session_id=req.session_id # Pass session_id
            ):
                yield chunk
        except Exception as e:
            print(f"Edit Pipeline error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/export")
async def export_document(req: ExportRequest):
    print("Received request to export docx.")
    try:
        doc_stream = export_to_docx(image_b64=req.image, node_details=req.plan)
        headers = {
            'Content-Disposition': 'attachment; filename="architecture_plan.docx"'
        }
        return StreamingResponse(
            doc_stream, 
            headers=headers,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        print(f"Error exporting Docx: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)