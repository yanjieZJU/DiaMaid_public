import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE any other imports — many server modules (agents.py,
# editor_memory.py, retriever_neo4j.py, pipeline_runner.py) read env vars
# at module-import time, so the file must be loaded first.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
import hashlib
import httpx
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)  # 添加 server 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 添加项目根目录

# from config import *

from pipeline_runner import run_pipeline, run_edit_pipeline # Updated Import
from session_manager import SESSIONS_DIR, create_modified_plan, get_session_tree
from exporter import export_to_docx
# from retriever_neo4j import get_global_stores
from retriever_neo4j import get_graph_store, get_vector_store
from retriever_rag import get_rag_store, rag_retrieve
from chat_session_manager import (
    create_chat_session, save_chat_turn, get_chat_session,
    list_chat_sessions, delete_chat_session, rename_chat_session
)

import threading

KG_READY = False
RAG_READY = False
KG_LOADING = False
RAG_LOADING = False
KG_INIT_ERROR = None
RAG_INIT_ERROR = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...", flush=True)
    print("Startup complete.", flush=True)
    yield

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Server starting — pre-loading knowledge stores...")
#     try:
#         # get_global_stores()
#         get_graph_store()
#         print("Knowledge graph stores ready.")
#     except Exception as e:
#         print(f"Failed to pre-load knowledge graph stores: {e}")

#     # RAG index build can take 30-60 min on first run — run in background thread
#     def _init_rag():
#         try:
#             get_rag_store()
#             print("RAG vector store ready.")
#         except Exception as e:
#             print(f"Failed to initialize RAG store: {e}")

#     threading.Thread(target=_init_rag, daemon=True, name="rag-init").start()
#     print("RAG index initializing in background — server ready.")
#     yield


app = FastAPI(lifespan=lifespan)

# --- CORS Setting ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM Client Setting ---
http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(
    http_client=http_client,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
    )

from typing import Optional, Union # Add Optional and Union


# --- 3. 定义请求数据格式 ---
class AnchorDef(BaseModel):
    user_term: str
    description: str = ""

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
    session_id: Optional[str] = None
    plan_id: Optional[str] = None
    current_plan: dict
    instruction: str
    original_query: str
    referenced_modules: Optional[list] = None  # [{plan_name, module_name, module_data}]
    focused_modules: Optional[list] = None  # module name(s) the user selected on the canvas

class ExportRequest(BaseModel):
    image: str # Base64 encoded image string
    plan: Union[dict, list] # Module details json
    overall_description: Optional[str] = None
    plan_id: Optional[str] = None

class SaveRequest(BaseModel):
    session_id: Optional[str] = None
    plan_id: Optional[str] = None
    current_plan: dict
    original_query: Optional[str] = None

@app.get("/ready")
async def ready():
    return {
        "kg_ready": KG_READY,
        "rag_ready": RAG_READY,
        "kg_loading": KG_LOADING,
        "rag_loading": RAG_LOADING,
        "kg_error": KG_INIT_ERROR,
        "rag_error": RAG_INIT_ERROR,
        "all_ready": KG_READY and RAG_READY,
    }

@app.get("/ready/kg")
async def ready_kg():
    return {
        "ready": KG_READY,
        "loading": KG_LOADING,
        "error": KG_INIT_ERROR,
    }


@app.get("/ready/rag")
async def ready_rag():
    return {
        "ready": RAG_READY,
        "loading": RAG_LOADING,
        "error": RAG_INIT_ERROR,
    }

@app.post("/warmup/kg")
async def warmup_kg():
    global KG_READY, KG_LOADING, KG_INIT_ERROR

    if KG_READY:
        return {"status": "ok", "message": "KG already ready."}
    if KG_LOADING:
        return {"status": "ok", "message": "KG warmup already running."}

    def _run():
        global KG_READY, KG_LOADING, KG_INIT_ERROR
        KG_LOADING = True
        KG_INIT_ERROR = None
        try:
            print("[KG] Warmup started...", flush=True)
            get_graph_store()
            get_vector_store()
            KG_READY = True
            print("[KG] Warmup finished.", flush=True)
        except Exception as e:
            KG_INIT_ERROR = str(e)
            print(f"[KG] Warmup failed: {e}", flush=True)
        finally:
            KG_LOADING = False

    threading.Thread(target=_run, daemon=True, name="kg-warmup").start()
    return {"status": "ok", "message": "KG warmup started."}


@app.post("/warmup/rag")
async def warmup_rag():
    global RAG_READY, RAG_LOADING, RAG_INIT_ERROR

    if RAG_READY:
        return {"status": "ok", "message": "RAG already ready."}
    if RAG_LOADING:
        return {"status": "ok", "message": "RAG warmup already running."}

    def _run():
        global RAG_READY, RAG_LOADING, RAG_INIT_ERROR
        RAG_LOADING = True
        RAG_INIT_ERROR = None
        try:
            print("[RAG] Warmup started...", flush=True)
            get_rag_store()
            RAG_READY = True
            print("[RAG] Warmup finished.", flush=True)
        except Exception as e:
            RAG_INIT_ERROR = str(e)
            print(f"[RAG] Warmup failed: {e}", flush=True)
        finally:
            RAG_LOADING = False

    threading.Thread(target=_run, daemon=True, name="rag-warmup").start()
    return {"status": "ok", "message": "RAG warmup started."}

# --- 4. 核心接口 ---
@app.get("/ping")
async def ping():
    return {"ok": True}

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
                num_outputs=3, 
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
    try:
        md = req.current_plan.get("module_details", {})
        md_str = json.dumps(md, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        sha = hashlib.sha256(md_str).hexdigest()
        count = len(md) if isinstance(md, dict) else (len(md) if isinstance(md, list) else 0)
        print(f"接收模块数={count}, SHA={sha}")
    except Exception as e:
        print(f"接收架构摘要失败: {e}")
        sha = None
    # 将“前端上传的当前架构”作为一次版本节点写入会话树，并把后续 Editor 结果挂在其下
    effective_parent_id = req.plan_id
    try:
        pre_desc = req.current_plan.get("overall_description", "前端上传的当前架构")
        new_pre_id = create_modified_plan(
            session_id=req.session_id or "anonymous_session",
            parent_node_id=req.plan_id,
            user_instruction="(用户上传) 前端当前架构",
            original_query=req.original_query or "",
            modified_description=pre_desc,
            modified_arch_json=req.current_plan,
            perspective_prefix="(手动快照)"
        )
        print(f"已在会话树中保存编辑前快照节点: {new_pre_id}")
        effective_parent_id = new_pre_id
    except Exception as e:
        print(f"保存编辑前快照到会话树失败: {e}")
    
    def event_generator():
        try:
            try:
                first_msg = {"type": "progress", "message": f"已接收架构: {sha[:8] if sha else 'unknown'}", "percent": 10}
                yield json.dumps(first_msg) + "\n"
            except Exception:
                pass
            # 使用 run_edit_pipeline 处理修改
            for chunk in run_edit_pipeline(
                plan_id=effective_parent_id,
                current_plan_json=req.current_plan,
                user_instruction=req.instruction,
                original_query=req.original_query,
                session_id=req.session_id,
                referenced_modules=req.referenced_modules,
                focused_modules=req.focused_modules,
            ):
                yield chunk
        except Exception as e:
            print(f"Edit Pipeline error: {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/sessions/save")
async def save_snapshot(req: SaveRequest):
    try:
        desc = req.current_plan.get("overall_description", "Manual Save")
        new_id = create_modified_plan(
            session_id=req.session_id or "anonymous_session",
            parent_node_id=req.plan_id,
            user_instruction="(手动保存) 用户保存当前架构",
            original_query=req.original_query or "",
            modified_description=desc,
            modified_arch_json=req.current_plan,
            perspective_prefix="(手动保存)"
        )
        return JSONResponse({"status": "ok", "new_plan_id": new_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/revert")
async def revert_to_parent(session_id: Optional[str] = Query(None), plan_id: Optional[str] = Query(None)):
    if not plan_id:
        raise HTTPException(status_code=400, detail="Missing session_id or plan_id")
    sid = session_id or "anonymous_session"
    tree = get_session_tree(sid)
    nodes = tree.get("nodes", {})
    if plan_id not in nodes:
        raise HTTPException(status_code=404, detail="Plan not found")
    current = nodes[plan_id]
    parent_id = current.get("parent_id")
    if not parent_id:
        raise HTTPException(status_code=404, detail="Already at root; no parent to revert")
    if parent_id not in nodes:
        raise HTTPException(status_code=404, detail="Parent plan not found")
    parent = nodes[parent_id]
    arch = parent.get("arch_json", {})
    return JSONResponse({
        "status": "ok",
        "plan": {
            "plan_id": parent_id,
            "arch_json": arch,
            "description": parent.get("description", arch.get("overall_description", "")),
            "perspective": parent.get("perspective", "")
        }
    })

@app.post("/export")
async def export_document(req: ExportRequest):
    print("Received request to export docx.")
    try:
        doc_stream = export_to_docx(image_b64=req.image, node_details=req.plan, overall_description=req.overall_description, plan_id=req.plan_id)
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

# ─────────────────────────────────────────────
# Chat endpoints (pure LLM conversation page)
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: Union[str, list]  # str for text, list for vision (image_url blocks)

class ChatRequest(BaseModel):
    messages: list[ChatMessage]   # full history
    session_id: Optional[str] = None
    save_history: bool = True  # whether to save to server

class ImageRequest(BaseModel):
    prompt: str
    history: list[ChatMessage] = []  # prior text turns for context
    session_id: Optional[str] = None

# RAG tool definition for function calling
_RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Search the academic paper knowledge base for relevant information. "
            "Use this tool before answering questions about: system design, "
            "HCI methods, task scheduling, anomaly detection, robot systems, "
            "interaction techniques, or any technical/research topic. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the paper knowledge base. Use specific technical terms.",
                }
            },
            "required": ["query"],
        },
    },
}

def _format_rag_results(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable tool result string."""
    if not chunks:
        return "No relevant papers found."
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        section = c.get("section", "")
        text = c.get("text", "")
        header = f"[{i}] {source}"
        if section:
            header += f" — {section}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


@app.post("/chat")
async def chat_text(req: ChatRequest):
    """
    Streaming text chat with RAG tool calling (baseline).
    The LLM can invoke search_papers to retrieve relevant paper chunks via
    hybrid BM25 + dense retrieval with RRF fusion and query expansion.
    """
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    # Extract user message for saving
    user_msg_content = ""
    if msgs and msgs[-1]["role"] == "user":
        content = msgs[-1]["content"]
        if isinstance(content, str):
            user_msg_content = content
        elif isinstance(content, list):
            # Extract text from multimodal content
            user_msg_content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            )

    def stream():
        assistant_content = ""
        rag_sources = None  # will hold source info if RAG was used
        try:
            # Signal that we're thinking (before first LLM call)
            yield json.dumps({"type": "thinking"}) + "\n"

            # ── Step 1: first call with tools available ──────────────────────
            # Retry up to 2 times on timeout — the relay API has ~10% timeout rate.
            first_resp = None
            last_exc = None
            for _attempt in range(3):
                try:
                    first_resp = client.chat.completions.create(
                        model=os.environ.get("CHAT_MODEL"),
                        messages=msgs,
                        tools=[_RAG_TOOL],
                        tool_choice="auto",
                        stream=False,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    print(f"[chat] Step1 attempt {_attempt+1} failed: {exc}")
            if first_resp is None:
                raise last_exc
            choice = first_resp.choices[0]

            # ── Step 2: if LLM wants to call the tool, execute it ────────────
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                all_queries = []
                all_chunks = []

                for tool_call in choice.message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query", "")
                    except Exception:
                        query = ""
                    all_queries.append((tool_call, query))

                # Report first query to frontend
                first_query = all_queries[0][1] if all_queries else ""
                yield json.dumps({"type": "tool_call", "query": first_query}) + "\n"

                assistant_msg = {
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ],
                }

                tool_result_msgs = []
                for tool_call, query in all_queries:
                    if query:
                        if not RAG_READY:
                            tool_result = "RAG knowledge base is not ready yet. Please warm it up first."
                        else:
                            chunks = rag_retrieve(query)
                            all_chunks.extend(chunks)
                            tool_result = _format_rag_results(chunks)
                    else:
                        tool_result = "Empty query."
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })

                if all_chunks:
                    rag_sources = {
                        "query": ", ".join(q for _, q in all_queries if q),
                        "count": len(all_chunks),
                        "ready": True,
                        "titles": [c.get("source", "unknown") for c in all_chunks[:5]],
                    }
                elif all_queries and not RAG_READY:
                    rag_sources = {"query": first_query, "count": 0, "ready": False}

                msgs_with_tool = msgs + [assistant_msg] + tool_result_msgs

                # ── Step 3: stream final answer with retrieved context ───────
                final_resp = client.chat.completions.create(
                    model=os.environ.get("CHAT_MODEL"),
                    messages=msgs_with_tool,
                    stream=True,
                )
                for chunk in final_resp:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        assistant_content += delta.content
                        yield json.dumps({"type": "delta", "content": delta.content}) + "\n"

            else:
                # ── No tool call — stream the direct response ────────────────
                content = choice.message.content or ""
                assistant_content = content
                if content:
                    yield json.dumps({"type": "delta", "content": content}) + "\n"

            yield json.dumps({"type": "done", "rag_sources": rag_sources}) + "\n"

            # Save to server if requested
            if req.save_history and req.session_id:
                try:
                    save_chat_turn(req.session_id, {
                        "role": "user",
                        "type": "text",
                        "content": user_msg_content,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_chat_turn(req.session_id, {
                        "role": "assistant",
                        "type": "text",
                        "content": assistant_content,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Failed to save chat history: {e}")

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/chat/image")
async def chat_image(req: ImageRequest):
    """Generate an image using Qwen image generation model and return base64 PNG."""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY not set")
    try:
        import dashscope
        from dashscope import MultiModalConversation
        import httpx as _httpx
        import base64 as _base64
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

        # Build full conversation history + current request, same as /chat.
        # GPT reads the context and compresses it into a self-contained image prompt.
        history_msgs = [
            {"role": m.role, "content": m.content}
            for m in req.history
            if isinstance(m.content, str) and m.content.strip()
        ]
        messages_for_gpt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. The user wants to generate an image based on the current conversation. "
                    "Read the full conversation history to understand what has been discussed "
                    "(e.g. system design, module breakdown, data flow, visual style). "
                    "Compress the relevant context into a clear, self-contained image generation prompt in English. "
                    "The prompt should describe: what to show, how it is arranged, and what visual style to use. "
                    "Output ONLY the prompt text — no explanation, no preamble."
                )
            },
            *history_msgs,
            {"role": "user", "content": req.prompt},
        ]
        gpt_resp = client.chat.completions.create(
            model=os.environ.get("CHAT_MODEL"),
            messages=messages_for_gpt,
            temperature=0.4,
        )
        image_prompt = gpt_resp.choices[0].message.content.strip()
        print(f"🖼️  Image prompt: {image_prompt[:120]}...")

        # Call Qwen image generation with the context-aware prompt
        response = MultiModalConversation.call(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            model="qwen-image-2.0-pro",
            messages=[{"role": "user", "content": [{"text": image_prompt}]}],
            result_format='message',
            stream=False,
            watermark=False,
            prompt_extend=True,
            size='1536*1152',
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Qwen API error: {response}")

        content = response.output['choices'][0]['message']['content']
        image_url = next(item['image'] for item in content if isinstance(item, dict) and 'image' in item)
        img_bytes = _httpx.get(image_url, timeout=30).content
        b64 = _base64.b64encode(img_bytes).decode('utf-8')

        # Save image turn to session if requested
        if req.session_id:
            try:
                save_chat_turn(req.session_id, {
                    "role": "user",
                    "type": "text",
                    "content": req.prompt,
                    "timestamp": datetime.now().isoformat()
                })
                save_chat_turn(req.session_id, {
                    "role": "assistant",
                    "type": "image",
                    "content": b64,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Failed to save image chat history: {e}")

        return JSONResponse({"type": "image", "b64": b64})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Chat session management endpoints
# ─────────────────────────────────────────────

@app.get("/chat/sessions")
async def get_chat_sessions():
    """List all chat sessions."""
    try:
        sessions = list_chat_sessions()
        return JSONResponse({"sessions": sessions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/{session_id}")
async def get_chat_session_by_id(session_id: str):
    """Get a specific chat session by ID."""
    try:
        session = get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return JSONResponse(session)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RenameChatSessionRequest(BaseModel):
    title: str

@app.put("/chat/sessions/{session_id}/rename")
async def rename_chat_session_endpoint(session_id: str, req: RenameChatSessionRequest):
    """Rename a chat session."""
    try:
        success = rename_chat_session(session_id, req.title)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return JSONResponse({"status": "ok"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session_endpoint(session_id: str):
    """Delete a chat session."""
    try:
        success = delete_chat_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return JSONResponse({"status": "ok"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
