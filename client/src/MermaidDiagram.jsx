// client/src/MermaidDiagram.jsx
import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

const MermaidDiagram = ({ code }) => {
  const elementRef = useRef(null);

  useEffect(() => {
    // 1. 初始化 mermaid 配置
    mermaid.initialize({
      startOnLoad: false, // 禁止自动加载，我们需要手动控制
      theme: 'default',
      securityLevel: 'loose',
    });

    const renderDiagram = async () => {
      if (code && elementRef.current) {
        try {
          // 清空旧内容
          elementRef.current.innerHTML = '';
          
          // 2. 生成唯一的 ID，防止冲突
          const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
          
          // 3. 调用 mermaid 的 API 进行渲染
          // .render(id, code) 会返回 { svg } 对象
          const { svg } = await mermaid.render(id, code);
          
          // 4. 将生成的 SVG 插入到 DOM 中
          elementRef.current.innerHTML = svg;
        } catch (error) {
          console.error('Mermaid 渲染失败:', error);
          elementRef.current.innerHTML = '<p style="color:red">流程图语法错误，无法渲染</p>';
        }
      }
    };

    renderDiagram();
  }, [code]); // 当 code 变化时重新渲染

  return (
    <div className="diagram-container" style={{ border: '1px solid #ddd', padding: '20px', borderRadius: '8px', minHeight: '300px', backgroundColor: '#f9f9f9' }}>
      <div ref={elementRef} style={{ width: '100%', overflowX: 'auto' }} />
    </div>
  );
};

export default MermaidDiagram;