// client/src/MermaidDiagram.jsx
import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

const MermaidDiagram = ({ code, onLayoutChange, onNodeClick }) => {
  const elementRef = useRef(null);

  useEffect(() => {
    // 1. 初始化 mermaid 配置
    mermaid.initialize({
      startOnLoad: false, // 禁止自动加载，我们需要手动控制
      theme: 'default',
      securityLevel: 'loose',
    });

    const renderDiagram = async () => {
      if (code && typeof code === 'string' && elementRef.current) {
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

          // 5. 解析节点位置并通知父组件
          if (onLayoutChange) {
            setTimeout(() => {
              const positions = {};
              const edges = [];
              const clusters = [];

              // 1. 解析节点位置
              const nodes = elementRef.current.querySelectorAll('.node');
              nodes.forEach(node => {
                const transform = node.getAttribute('transform');
                const match = /translate\(([^,]+),\s*([^)]+)\)/.exec(transform);
                
                if (match) {
                  const x = parseFloat(match[1]);
                  const y = parseFloat(match[2]);
                  
                  const textElement = node.querySelector('.nodeLabel');
                  const name = textElement ? textElement.innerText.trim() : node.textContent.trim();
                  
                  // 尝试解析节点ID (mermaid 自动生成的 ID)
                  // Mermaid 节点 ID 通常在 id 属性中
                  const nodeId = node.id;

                  if (name) {
                    positions[name] = { x, y, id: nodeId };
                    
                    // 绑定点击事件
                    // 注意：我们需要移除之前的监听器，或者确保不重复绑定
                    // 这里简单处理：直接覆盖 onclick (虽然不推荐，但在 React useEffect 中可能需要更小心的处理)
                    // 更好的方式是 addEventListener，并在 cleanup 中移除
                    // 但由于 Mermaid 每次重新渲染都会重置 DOM，所以在这里绑定是安全的
                    node.style.cursor = 'pointer';
                    node.onclick = (e) => {
                        e.stopPropagation();
                        if (onNodeClick) {
                            onNodeClick(name);
                        }
                    };
                  }
                }
              });

              // 2. 解析连线 (Edges)
              // Mermaid 的连线通常是 path 元素，带有 .edgePath 类
              // 这里的解析比较困难，因为 path 的 d 属性是一系列坐标，很难直接反推出连接了哪两个节点
              // 替代方案：我们可以尝试解析 Mermaid 代码本身，或者在 App.jsx 中解析
              // 但为了视觉同步，我们尽量从 DOM 获取（虽然很难获取关系，只能获取路径）
              // 更好的方法：我们在 App.jsx 中解析 mermaidCode 字符串来获取关系结构。
              // 这里主要负责返回 DOM 中的布局坐标。
              
              // 3. 解析子图 (Clusters/Subgraphs)
              const clusterElements = elementRef.current.querySelectorAll('.cluster');
              clusterElements.forEach(cluster => {
                 // 尝试获取子图的范围和标签
                 // Mermaid cluster 结构比较复杂，通常包含一个 rect 和一个 label
                 const labelEl = cluster.querySelector('.cluster-label');
                 const label = labelEl ? labelEl.innerText.trim() : '';
                 
                 // 获取 rect 尺寸
                 const rect = cluster.querySelector('rect');
                 if (rect && label) {
                     // 注意：cluster 的位置通常也是通过 transform 定义的，或者 rect 的 x,y
                     const transform = cluster.getAttribute('transform');
                     let x = 0, y = 0;
                     if (transform) {
                        const match = /translate\(([^,]+),\s*([^)]+)\)/.exec(transform);
                        if (match) {
                            x = parseFloat(match[1]);
                            y = parseFloat(match[2]);
                        }
                     }
                     
                     // rect 的 x,y 可能是相对于 cluster 组的偏移
                     x += parseFloat(rect.getAttribute('x') || 0);
                     y += parseFloat(rect.getAttribute('y') || 0);
                     const width = parseFloat(rect.getAttribute('width'));
                     const height = parseFloat(rect.getAttribute('height'));
                     
                     clusters.push({
                         label,
                         x, y, width, height
                     });
                 }
              });
              
              if (Object.keys(positions).length > 0) {
                console.log('Detected layout positions:', positions);
                // 传递更多信息回去
                onLayoutChange({ positions, clusters });
              }
            }, 100); // 稍微延迟确保 DOM 渲染完成
          }

        } catch (error) {
          console.error('Mermaid 渲染失败:', error);
          elementRef.current.innerHTML = '<p style="color:red">流程图语法错误，无法渲染</p>';
        }
      } else if (code && typeof code !== 'string') {
        console.error('MermaidDiagram received non-string code:', code);
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