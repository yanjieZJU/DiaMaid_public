// client/src/App.jsx
import React, { useState, useRef } from 'react';
import MermaidDiagram from './MermaidDiagram';
import './App.css';
import axios from 'axios';

function App() {
  const [description, setDescription] = useState('');
  const [nodes, setNodes] = useState([]);
  const [newNodeName, setNewNodeName] = useState('');
  const [mermaidCode, setMermaidCode] = useState('');
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  // 添加节点
  const addNode = () => {
    if (!newNodeName.trim()) return;
    
    // 计算画布中心位置
    const width = canvasRef.current ? canvasRef.current.offsetWidth : 800;
    const height = canvasRef.current ? canvasRef.current.offsetHeight : 600;

    const newNode = {
      id: Date.now(),
      name: newNodeName.trim(),
      // 简单的随机散布在中心附近
      x: width / 2 + (Math.random() * 200 - 100),
      y: height / 2 + (Math.random() * 200 - 100)
    };
    
    setNodes([...nodes, newNode]);
    setNewNodeName('');
  };

  const removeNode = (id) => {
    setNodes(nodes.filter(node => node.id !== id));
  };

  const handleGenerate = async () => {
    if (!description && nodes.length === 0) return;
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/generate', {
        prompt: description,
        userNodes: nodes.map(n => n.name)
      });
      setMermaidCode(response.data.code);
    } catch (error) {
      console.error(error);
      setMermaidCode(`graph TD\nError[生成失败: ${error.message}]`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>DiaMaid - 智能流程图设计台</h1>
      </header>

      <main className="main-content">
        <div className="generator-layout">
          
          {/* --- 左侧边栏 (输入 + 工具 + 代码) --- */}
          <div className="sidebar-left">
            
            {/* 1. 需求描述 */}
            <div className="left-card" style={{ flex: '0 0 auto' }}>
              <div className="card-title">需求描述</div>
              <textarea 
                className="styled-input"
                rows={4}
                placeholder="例如：设计一个包含RAG和LLM的历史教育系统..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>

            {/* 2. 功能面板 (手动节点) */}
            <div className="left-card" style={{ flex: '0 0 auto' }}>
              <div className="card-title">功能面板 (预设节点)</div>
              <div className="action-bar" style={{marginTop:0}}>
                <input 
                  className="styled-input"
                  style={{marginBottom:0}}
                  placeholder="节点名称"
                  value={newNodeName}
                  onChange={(e) => setNewNodeName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addNode()}
                />
                <button className="btn-add" onClick={addNode}>添加</button>
              </div>
              
              {/* 显示已添加的标签 */}
              <div className="manual-nodes-container">
                {nodes.length === 0 && <span style={{color:'#ccc', fontSize:'12px'}}>暂无节点</span>}
                {nodes.map(node => (
                  <span key={node.id} className="tag-node">
                    {node.name}
                    <span className="tag-remove" onClick={() => removeNode(node.id)}>×</span>
                  </span>
                ))}
              </div>

              <button 
                className="btn-primary" 
                style={{marginTop: '15px'}}
                onClick={handleGenerate}
                disabled={loading}
              >
                {loading ? '生成中...' : '开始生成流程图'}
              </button>
            </div>

            {/* 3. 生成的代码 (对应草图左下角) */}
            <div className="left-card" style={{ flex: '1 1 auto' }}>
              <div className="card-title">生成的 Mermaid 代码</div>
              <textarea 
                className="styled-input code-editor-area"
                value={mermaidCode}
                onChange={(e) => setMermaidCode(e.target.value)}
                placeholder="Mermaid 代码将在这里生成，支持手动修改..."
                spellCheck="false"
              />
            </div>
          </div>

          {/* --- 中间画布 (图表 + 节点) --- */}
          <div className="panel canvas-panel" ref={canvasRef}>
            {/* 这里的 div 用于展示手动节点 */}
            <div className="canvas-overlay">
               {nodes.map(node => (
                 <div 
                   key={node.id}
                   className="manual-node-visual"
                   style={{ left: node.x, top: node.y }}
                 >
                   {node.name}
                 </div>
               ))}
            </div>

            {/* 这里的 MermaidDiagram 是主要的渲染结果 */}
            <div className="mermaid-wrapper">
              {mermaidCode ? (
                <MermaidDiagram code={mermaidCode} />
              ) : (
                <div style={{color: '#999', textAlign:'center'}}>
                  <h3>画布空闲</h3>
                  <p>请在左侧输入需求并生成，<br/>或者在功能面板添加手动节点。</p>
                </div>
              )}
            </div>
          </div>

          {/* --- 右侧边栏 (属性/详情) --- */}
          <div className="panel sidebar-right">
            <div className="card-title">节点属性 / 详情</div>
            <div className="info-placeholder">
              生成图表后，<br/>
              点击画布上的节点<br/>
              可在此处查看详细信息<br/>
              (该功能待开发)
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;