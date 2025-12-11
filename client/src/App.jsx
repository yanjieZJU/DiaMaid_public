// client/src/App.jsx
import React, { useState } from 'react';
import MermaidDiagram from './MermaidDiagram';
import './App.css';
import axios from 'axios';

function App() {
  // --- 状态管理 ---
  
  // 1. 用户输入的总体描述
  const [description, setDescription] = useState('');
  
  // 2. 用户手动创建的节点列表 (模拟可视化界面的节点数据)
  const [nodes, setNodes] = useState([{ id: 1, text: '开始' }]);

  // 3. 后端返回的 Mermaid 代码 (这里先给一个默认值用于测试)
  const [mermaidCode, setMermaidCode] = useState(`
    graph TD
    A[开始] --> B{是否也是简单?}
    B -- 是 --> C[完成前端]
    B -- 否 --> D[继续优化]
  `);

  const [loading, setLoading] = useState(false);

  // --- 事件处理 ---

  // 添加一个新节点输入框
  const addNode = () => {
    const newNode = { id: Date.now(), text: '' };
    setNodes([...nodes, newNode]);
  };

  // 更新节点文本
  const updateNodeText = (id, newText) => {
    const updatedNodes = nodes.map(node => 
      node.id === id ? { ...node, text: newText } : node
    );
    setNodes(updatedNodes);
  };

  // 删除节点
  const removeNode = (id) => {
    setNodes(nodes.filter(node => node.id !== id));
  };

  // 发送给后端
  const handleGenerate = async () => {
    if (!description && nodes.length === 0) {
      alert("请输入描述或至少添加一个节点");
      return;
    }

    setLoading(true);
    setMermaidCode(''); // 清空旧图，提示用户正在生成

    try {
      // 1. 发送请求给 FastAPI 后端
      const response = await axios.post('http://localhost:8000/generate', {
        prompt: description,
        userNodes: nodes.map(n => n.text) // 提取节点文本数组
      });

      // 2. 获取后端返回的代码
      const generatedCode = response.data.code;
      console.log("后端返回代码:", generatedCode);
      
      // 3. 更新状态，MermaidDiagram 组件会自动重绘
      setMermaidCode(generatedCode);

    } catch (error) {
      console.error("请求失败:", error);
      alert("生成失败，请检查后端服务是否启动。");
      setMermaidCode(`graph TD\nError[生成出错: ${error.message}]`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>AI 流程图生成器</h1>
      </header>

      <main className="main-content">
        {/* 左侧：输入控制区 */}
        <div className="input-panel">
          <div className="section">
            <h3>1. 总体描述</h3>
            <textarea 
              placeholder="例如：生成一个用户注册的流程图..." 
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>

          <div className="section">
            <h3>2. 定义关键节点</h3>
            <div className="node-list">
              {nodes.map((node, index) => (
                <div key={node.id} className="node-item">
                  <span className="node-label">节点 {index + 1}:</span>
                  <input 
                    type="text" 
                    value={node.text}
                    onChange={(e) => updateNodeText(node.id, e.target.value)}
                    placeholder="输入节点内容"
                  />
                  <button onClick={() => removeNode(node.id)} className="btn-delete">×</button>
                </div>
              ))}
            </div>
            <button onClick={addNode} className="btn-secondary">+ 添加节点</button>
          </div>

          <button 
            onClick={handleGenerate} 
            disabled={loading} 
            className="btn-primary"
          >
            {loading ? '生成中...' : '生成流程图'}
          </button>
        </div>

        {/* 右侧：显示区 */}
        <div className="display-panel">
          <h3>渲染结果</h3>
          <MermaidDiagram code={mermaidCode} />
          
          <div className="code-preview">
            <h4>原始 Mermaid 代码:</h4>
            <pre>{mermaidCode}</pre>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;