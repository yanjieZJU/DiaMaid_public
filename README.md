# DiaMaid (Diagram Maid)

**DiaMaid** 是一款智能驱动的系统设计与架构图助手。它能够将自然语言的系统描述和简单的节点约束，转化为专业、详尽的系统架构方案和架构图。通过结合大语言模型 (LLMs) 多智能体协同，DiaMaid 致力于帮助研究人员、开发者和架构师轻松推演复杂系统的可视化与方案设计。

## 🌟 核心特性 (Key Features)

### 1. 智能生成多视角方案 
- **发散性规划 (Divergent Planning)**：只要输入一段大体的需求描述，后台多智能体会自动发散，提供多个不同技术侧重点的系统架构方案（以 Excel 工作表 Tab 形式呈现），全方位启发您的设计灵感。
- **丰富的节点详情**：您设计的不仅仅是连线的图表。每个模块节点内都包含了：
  - **核心功能 (Function)**
  - **技术手段与实现细节 (Technical Methods)**
  - **模块说明 (Brief Description)** 等重要结构化设计参考。
  
### 2. 交互式对话编辑与演进 (Agentic Editing)
- **对话修改架构**：对当前生成的方案细节不满意？在左侧聊天栏直接输入修改指令（如“添加一个 Redis 缓存中心”或“将数据存储改为 MySQL”），AI 会基于当前所选方案进行局部逻辑更新，并生成一条新的方案分支。
- **无缝回溯**：所有用户的修改指令和系统的更新结果均会产生完整的对话历史演进记录，方便您回顾过程和版本对比。

### 3. 直观的可视化工作区
- **React Flow & Mermaid.js**：提供了直观的白板结构和详尽的自动排版流程图。右侧面板分为渲染图区和信息详情区。单击流程图内的任何节点即可在右下角查看该模块的详细参数。
- **手动节点辅助**：支持在生成前，在左侧向画布中预置强行要求的核心基础节点，AI将确保将这些预设元素融合到最终生成的架构中。
- **快速渲染测试**：提供独立的 JSON 免后端测试通道，粘贴合法的架构方案 JSON 即可直接渲染图表，方便研发调试。

### 4. 专业报告导出 (Export)
- **一键导出 Word 文档**：在完成架构推演设计后，点击方案选项卡右上方的 **“导出为 Word”** 按钮。后台将自动为您排版生成一份专业`.docx`设计报告（内含当前的高清架构图图片、总体介绍以及各个模块属性细节表）。

## 🛠️ 技术栈 (Tech Stack)

- **前端 (Client)**：React, Vite, React Flow, Mermaid.js
- **后端 (Server)**：Python, FastAPI, Pydantic, python-docx
- **AI 与核心引擎**: 基于 OpenAI API 的底层多智能体协同管线 (Planner, Integrator, Editor, Reviewer 等)。

## 🚀 快速上手 (Getting Started)

### 前置要求 (Prerequisites)
- [Node.js](https://nodejs.org/) (建议 v18 及以上)
- [Python](https://www.python.org/) (建议 v3.10 及以上)

### 安装与运行 (Installation)

1. **获取代码**
   ```bash
   git clone https://github.com/mistymoonight/DiaMaid.git
   cd DiaMaid
   ```

2. **启动后端服务 (Server)**
   ```bash
   cd server
   # 推荐创建并激活您的虚拟环境，例如：python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   
   # 运行前请确保配置相应的环境变量 API 密钥 (可在代码端配置或使用 .env)
   # export OPENAI_API_KEY="YOUR_API_KEY"
   # export OPENAI_BASE_URL="YOUR_BASE_URL"
   
   # 启动后端程序 (默认运行在 http://localhost:8000)
   python main.py
   ```

3. **启动前端客户端 (Client)**
   ```bash
   cd client
   # 安装 npm 依赖
   npm install
   
   # 运行 Vite 开发服务器
   npm run dev
   ```
   启动成功后，即可在浏览器中访问 `http://localhost:5173`。

## 📝 基本使用流程 (Usage Guide)

1. **输入基础信息**：在页面左侧，您可以填入“被试 ID”用于本地追踪数据日志。
2. **初始化生成**：
   - 在底部的大文本框中，输入您的核心系统需求（例如：“设计一个能够跨域监控的大型物联网数据处理系统架构”）。
   - 点击 **“开始生成流程图”**，等待等待进度更新。
3. **查看多方案**：生成完成之后，屏幕会切入全屏审阅模式。顶部的 Tab 可以切换不同技术解决思路的方案（例如方案A、方案B），右侧画布能直观审视对应的拓扑图。
4. **对话式修正**：在当前最佳的方案下，可在左侧输入框继续与系统对话（例如提出“请把消息队列换为Kafka”），AI 会根据需求演化出下一代修改版图形。
5. **归档与导出**：一切就绪后，点击 **“导出为 Word”** 即可直接获取设计文档。

## 🤝 贡献与许可 (Contributing & License)

欢迎提交 issue 和 pull request 共同完善本项目。
本项目采用 [MIT License](LICENSE) 开源协议。
