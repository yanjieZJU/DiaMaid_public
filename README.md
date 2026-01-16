# DiaMaid (Diagram Maid)

**DiaMaid** is an intelligent, AI-powered system design assistant that transforms natural language descriptions and sparse node inputs into professional, comprehensive system architecture diagrams. By combining the power of Large Language Models (LLMs) with Graph RAG (Retrieval-Augmented Generation), DiaMaid helps developers and architects visualize complex systems effortlessly.

## 🌟 Key Features

### 1. Interactive & Synchronized Canvas
- **React Flow Integration**: A fully interactive canvas where you can drag, drop, and connect nodes manually.
- **Bi-directional Sync**: 
  - Changes in the canvas (adding nodes, connecting lines) are understood by the backend.
  - AI-generated Mermaid diagrams are automatically parsed and synchronized back to the interactive canvas.
- **Smart Grouping**: Drag a node over another to create parent-child relationships automatically.

### 2. Deep Generation Mode (深度生成)
Unlock the full potential of AI with the **Deep Generation** toggle.
- **Intelligent Completion**: Unlike standard tools that strictly follow inputs, DiaMaid can infer missing components (e.g., adding a Redis cache between API and DB, or a Message Queue for async tasks).
- **Visual Distinction**: Auto-generated nodes appear in **Blue** to distinguish them from your manual inputs.
- **User Control**: 
  - **Retain**: One-click to accept AI suggestions and make them permanent.
  - **Remove**: One-click to discard AI additions and revert to your original design.

### 3. Rich Node Details
Don't just look at boxes—understand them.
- **Detail Inspector**: Click on any node (in the canvas or the preview image) to open the details panel.
- **Deep Insights**: View the backend-generated metadata for each node, including:
  - **Function**: What does this module actually do?
  - **Technical Methods**: How is it implemented?
  - **I/O Ports**: Explicit data flow definitions (what comes in, what goes out).

### 4. Optimized UI/UX
- **Dynamic Progress**: Visual feedback during the generation process.
- **Clean Controls**: Unified styling for input fields and action buttons.
- **One-Click Clear**: Easily reset your workspace.

## 🛠️ Tech Stack

- **Frontend**: React, React Flow, Mermaid.js, Vite
- **Backend**: Python, FastAPI
- **AI/Data**: LLM Agents (Planner, Retriever, Integrator, Drawer), Neo4j (Graph Storage)

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- Neo4j Database (for RAG features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mistymoonight/DiaMaid.git
   cd DiaMaid
   ```

2. **Start the Backend Server**
   ```bash
   cd server
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables (Create a .env file if needed for API Keys)
   # export OPENAI_API_KEY=...
   # export NEO4J_URI=...
   
   # Run the server
   python main.py
   ```
   Server will run at `http://localhost:8000`.

3. **Start the Frontend Client**
   ```bash
   cd client
   # Install dependencies
   npm install
   
   # Run the development server
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

## 📝 Usage Guide

1. **Add Nodes**: Type a node name (e.g., "User", "Server") and press Enter or click "Add".
2. **Describe Intent**: In the bottom text area, describe what you want to build (e.g., "A scalable e-commerce system").
3. **Toggle Deep Generation**: Check "允许添加随机节点 (深度生成)" if you want the AI to brainstorm and add helper nodes.
4. **Generate**: Click the "Generate" button.
5. **Refine**: 
   - Drag nodes to adjust layout.
   - Click "Retain" (保留) to keep AI-generated nodes.
   - Click nodes to view details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT](LICENSE)
