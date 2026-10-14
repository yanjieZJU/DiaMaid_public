# DiaMaid

面向交互系统概念设计的 AI 协作工具。DiaMaid 将自然语言需求转化为可查看、可编辑的系统架构方案，支持围绕同一份结构化表示持续探索和修改设计。

项目简介

系统概念设计需要确定模块、模块关系和数据流。DiaMaid 将这些内容组织为架构图和模块详情，并提供对话式方案生成与编辑流程。它是一个研究原型，用于探索 AI 如何参与架构构思与迭代。

主要功能

- 多视角架构生成：结合需求、基础模块、关系约束和检索结果，生成多种设计方案。
- 知识图谱检索：从 Neo4j 和本地向量索引中检索模块信息与结构路径。
- 架构图编辑：查看模块、端口和连接；通过自然语言修改方案，并保存为新的版本。
- 方案历史与编辑记忆：按父子关系记录方案迭代，支持查看分支历史和恢复上一个版本。
- 文献问答：独立聊天页面可结合关键词和向量检索查询论文内容。
- Word 导出：导出方案信息与架构图。

系统结构

- `client/`：React、Vite 前端；主设计界面使用 AntV X6，另有 React Flow 和 Mermaid 视图组件及聊天页面。
- `server/`：FastAPI 服务，提供生成、编辑、会话、问答和文档导出接口。
- `offline_processor/`：论文解析、知识图谱构建及离线处理脚本。
- `server/agents.py`、`server/editor_memory.py`：Agent 工作流与分支编辑记忆。

运行前准备

仓库不包含完整论文语料、Neo4j 实例或本地向量矩阵。架构生成需要预先准备知识图谱，以及对应的 `server/graph_storage/embeddings.npy` 和 `server/graph_storage/embedding_ids.json`。文献问答需要把论文 PDF 放入 `case_llm_artifact/original_articles/`，并构建检索索引。

需要一个兼容 OpenAI API 的聊天模型服务、已导入 DiaMaid 数据的 Neo4j 数据库，以及图谱使用的嵌入服务。默认嵌入模型为 DashScope `text-embedding-v4`，维度为 1024。图像问答和文档插图功能还需 DashScope API 密钥。环境变量模板见 `.env.example`；复制为根目录 `.env` 并填入自己的配置。不要提交 `.env` 或在代码中写入密钥。

安装与启动

建议 Python 3.10+、Node.js 18+。在项目根目录运行 `pip install -r requirements.txt`。进入 `server/` 后运行 `python main.py`，默认监听 `http://localhost:8000`。另开终端进入 `client/`，运行 `npm install` 和 `npm run dev`，再访问 Vite 显示的地址。

启动前请确认 `.env`、Neo4j 和本地向量数据均已准备好。仅克隆仓库不足以直接运行架构生成。前端开发配置默认将 API 请求发送到 `http://localhost:8000`。

接口入口

- 方案生成：`POST /generate`
- 方案编辑：`POST /edit`
- 文献问答：`POST /chat`
- 会话管理：`/chat/sessions`
- Word 导出：`POST /export`
- 服务状态：`GET /ready`、`GET /ready/kg`、`GET /ready/rag`
