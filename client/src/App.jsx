// client/src/App.jsx
import React, { useState, useRef, useCallback } from 'react';
import MermaidDiagram from './MermaidDiagram';
import './App.css';
import axios from 'axios';
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

// 初始节点
const initialNodes = [];
const initialEdges = [];

function App() {
  const [description, setDescription] = useState('');
  
  // 使用 React Flow 的 hook 管理状态
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const [newNodeName, setNewNodeName] = useState('');
  const [mermaidCode, setMermaidCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0); // 进度条状态
  const [nodeDetails, setNodeDetails] = useState({}); // 存储后端返回的节点详细信息
  const [retainRandomNodes, setRetainRandomNodes] = useState(false); // 是否保留随机节点

  // React Flow 实例
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const reactFlowWrapper = useRef(null);

  const [useRandomNodes, setUseRandomNodes] = useState(false); // 是否使用随机节点（深度生成模式）
  const [selectedNodeInfo, setSelectedNodeInfo] = useState(null); // 当前选中节点的信息
  const [editingNodeId, setEditingNodeId] = useState(null); // 当前正在编辑名称的节点ID
  const [editingNodeLabel, setEditingNodeLabel] = useState(''); // 当前编辑的名称

  // --- React Flow 事件处理 ---
  
  // 节点点击事件
  const onNodeClick = useCallback((event, node) => {
    event.stopPropagation(); // 防止触发 onPaneClick
    // 查找该节点的详细信息
    // node.data.label 是节点显示的名称
    const label = node.data.label;
    if (label && nodeDetails[label]) {
        setSelectedNodeInfo(nodeDetails[label]);
    } else {
        setSelectedNodeInfo({ name: label, description: '暂无详细描述' });
    }
  }, [nodeDetails]);

  // 连线事件
  const onConnect = useCallback((params) => {
    console.log('Connect:', params);
    setEdges((eds) => addEdge({ 
      ...params, 
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
      // 允许交互（选中、删除）
      interactionWidth: 20, 
    }, eds));
  }, [setEdges]);

  // 双击连线删除
  const onEdgeDoubleClick = useCallback((event, edge) => {
    setEdges((eds) => eds.filter((e) => e.id !== edge.id));
  }, [setEdges]);

  // 拖拽结束事件：处理“包含关系”
  const onNodeDragStop = useCallback((event, node) => {
    // 获取拖拽节点的位置和尺寸（大致估算，或者从 node.width/height 获取如果已测量）
    // React Flow 默认不直接提供 width/height 除非已渲染并测量，但我们可以尝试获取
    // 这里简化逻辑：检查中心点是否在另一个节点内
    
    // 获取所有节点（除了自己）
    const otherNodes = nodes.filter(n => n.id !== node.id);
    
    // 计算当前节点的中心点
    // 注意：node.position 是相对于父级的。如果已经是子节点，需要累加父级坐标才能得到绝对坐标用于检测
    // 但为了简化，我们假设第一层检测是基于绝对位置的，或者我们只支持单层嵌套（Level 1 -> Level 2）
    
    // 获取当前节点的绝对位置（如果它已经在某个父节点下，这个逻辑会复杂一点，
    // 这里简化为：只处理从顶层拖入某个节点，或者从某个节点拖回顶层）
    
    // 简单的碰撞检测逻辑
    // 假设节点大小大概是 150x40 (根据样式)
    const nodeWidth = node.width || 150;
    const nodeHeight = node.height || 40;
    const centerX = node.position.x + nodeWidth / 2;
    const centerY = node.position.y + nodeHeight / 2;

    let targetParent = null;

    // 遍历其他节点，寻找潜在的父节点
    // 注意：我们只允许成为“非子节点”的节点的子节点，防止无限嵌套或复杂逻辑
    for (const other of otherNodes) {
      // 忽略已经是子节点的节点作为父节点（简化）
      if (other.parentNode) continue;

      const otherWidth = other.width || 150;
      const otherHeight = other.height || 40;
      
      // 检查是否重叠（包含）
      // 严格一点：要求中心点在父节点范围内
      // 或者：如果父节点现在是“扩大”状态，它的实际渲染尺寸可能比原始尺寸大。
      // React Flow 的 node.width/height 通常是上次渲染的尺寸。
      // 如果我们手动修改了 style.width/height，React Flow 的内部测量可能还没更新。
      // 所以我们优先使用 style 中的尺寸（如果存在）
      const styleWidth = other.style?.width ? parseInt(other.style.width) : otherWidth;
      const styleHeight = other.style?.height ? parseInt(other.style.height) : otherHeight;

      if (
        centerX > other.position.x &&
        centerX < other.position.x + styleWidth &&
        centerY > other.position.y &&
        centerY < other.position.y + styleHeight
      ) {
        targetParent = other;
        break;
      }
    }

    if (targetParent) {
      console.log(`Node ${node.id} dropped into ${targetParent.id}`);
      
      // 如果已经是这个父节点的子节点，不需要做父子关系的变更，但可能需要更新相对位置
      // 但 React Flow 的 onNodeDragStop 默认是在拖拽结束后触发，
      // 如果节点已经是子节点，拖拽时它的 position 已经是相对于父节点的。
      // 如果我们在这里重新计算，可能会导致跳动。
      // 所以：如果已经是该父节点的子节点，并且仍在父节点范围内，我们只需要确认它还在里面即可。
      
      if (node.parentNode === targetParent.id) {
          // 已经在里面了，不需要做任何结构性变更
          // 但我们需要检查是否移出了（上面的 targetParent 逻辑已经涵盖了“在里面”）
          // 所以如果能走到这里，说明还在范围内。
          return;
      }

      // 更新节点：设置为子节点
      setNodes((nds) => {
        // 先计算，如果这个父节点已经有其他子节点，或者即将有这个新子节点
        // 这里逻辑稍微复杂一点：我们需要知道父节点是否有 *其他* 子节点。
        // 但其实更简单的逻辑是：只要有子节点加入，父节点就变大；
        // 只要子节点离开，我们再检查父节点是否还有剩余子节点。
        
        return nds.map((n) => {
          // 处理被拖入的子节点：缩小并建立父子关系
          if (n.id === node.id) {
            // 计算相对坐标
            const relativeX = node.position.x - targetParent.position.x;
            const relativeY = node.position.y - targetParent.position.y;
            
            return {
              ...n,
              parentNode: targetParent.id,
              position: { x: relativeX, y: relativeY },
              extent: 'parent', // 限制在父节点范围内 (可选)
              // 修改样式：缩小
              style: { 
                  ...n.style, 
                  zIndex: 10,
                  width: 100,
                  height: 40,
                  fontSize: '12px'
              } 
            };
          }
          
          // 处理父节点：扩大并改变外观以表示包含
          if (n.id === targetParent.id) {
              const currentWidth = parseInt(n.style?.width || 150);
              const currentHeight = parseInt(n.style?.height || 40);
              
              const minParentWidth = 250; 
              const minParentHeight = 150;
              
              return {
                  ...n,
                  style: {
                      ...n.style,
                      width: Math.max(currentWidth, minParentWidth),
                      height: Math.max(currentHeight, minParentHeight),
                      backgroundColor: 'rgba(240, 240, 240, 0.5)', 
                      borderStyle: 'dashed',
                  }
              };
          }
          
          return n;
        });
      });
    } else {
      // 如果没有落在任何节点内
      // 检查当前节点是否有父节点
      if (node.parentNode) {
         // 获取父节点对象（为了坐标转换）
         const parent = nodes.find(p => p.id === node.parentNode);
         
         // 关键逻辑：
         // 如果节点有父节点，但 targetParent 为空，说明节点被拖出了父节点范围。
         // 此时应该执行“脱离父节点”的操作。
         
         console.log(`Node ${node.id} moved out of parent ${node.parentNode}`);
         const oldParentId = node.parentNode;

         // 恢复为顶层节点
         setNodes((nds) => {
           // 注意：在 setNodes 回调中获取最新的 parent 可能更安全
           // 但 node.parentNode 已经是已知的
           const currentParent = nds.find(p => p.id === oldParentId);
           if (!currentParent) return nds; 
           
           // 检查父节点是否还有其他子节点
           const hasOtherChildren = nds.some(n => n.parentNode === oldParentId && n.id !== node.id);

           return nds.map(n => {
             // 1. 恢复子节点
             if (n.id === node.id) {
               // 坐标转换：相对 -> 绝对
               // 子节点的 position 是相对的
               // 但 React Flow 的 onNodeDragStop 传递的 node.position 是拖拽后的位置
               // 如果 extent='parent'，它会被限制在父节点内；如果没限制，它可以是负数或超出范围
               // 我们的逻辑是：一旦拖出（targetParent 为 null），就转换坐标。
               
               // 新的绝对坐标 = 父节点绝对坐标 + 子节点相对坐标
               const newAbsX = currentParent.position.x + node.position.x;
               const newAbsY = currentParent.position.y + node.position.y;

               return {
                 ...n,
                 parentNode: undefined,
                 position: {
                   x: newAbsX,
                   y: newAbsY
                 },
                 extent: undefined,
                 // 恢复默认样式
                 style: { 
                     ...n.style, 
                     width: 150, 
                     height: 40, 
                     fontSize: undefined,
                     backgroundColor: '#fff', 
                     borderStyle: 'solid',
                 }
               };
             }
             
             // 2. 如果父节点没有其他子节点了，恢复父节点样式
             if (n.id === oldParentId && !hasOtherChildren) {
                 return {
                     ...n,
                     style: {
                         ...n.style,
                         width: 150,
                         height: 40,
                         backgroundColor: '#fff',
                         borderStyle: 'solid',
                     }
                 };
             }

             return n;
           });
         });
      }
    }
  }, [nodes, setNodes]);


  // 添加节点
  const addNode = () => {
    if (!newNodeName.trim()) return;
    
    const newNodeId = `node-${Date.now()}`;
    const newNode = {
      id: newNodeId,
      data: { label: newNodeName.trim() },
      position: { 
        x: Math.random() * 200 + 50, 
        y: Math.random() * 200 + 50 
      },
      type: 'default', // 或 'input', 'output'
      style: { background: '#fff', border: '1px solid #777', padding: '10px', borderRadius: '5px', width: 150 }
    };
    
    setNodes((nds) => [...nds, newNode]);
    setNewNodeName('');
  };

  const removeNode = (id) => {
    setNodes((nds) => nds.filter((node) => node.id !== id));
    setEdges((eds) => eds.filter((edge) => edge.source !== id && edge.target !== id));
  };

  const clearNodes = () => {
      setNodes([]);
      setEdges([]);
      setMermaidCode('');
      setNewNodeName('');
      setNodeDetails({});
      setSelectedNodeInfo(null);
  };

  const confirmRetainNodes = () => {
      // 确认保留所有节点：将所有节点的 isGenerated 标记移除，并恢复默认样式
      setNodes((nds) => nds.map(node => ({
          ...node,
          data: { ...node.data, isGenerated: false },
          style: {
              ...node.style,
              background: '#fff',
              border: '1px solid #777'
          }
      })));
      setRetainRandomNodes(true);
      // 重置状态以便下次生成
      setTimeout(() => setRetainRandomNodes(false), 500);
  };

  const clearGeneratedNodes = () => {
      // 清除所有自动生成的节点
      setNodes((nds) => nds.filter(node => !node.data.isGenerated));
      // 同时清理相关的边
      // 这一步比较麻烦，因为边没有标记 isGenerated。
      // 但我们可以清理 source 或 target 不存在的边。
      // 由于 setNodes 是异步的，我们应该在 useEffect 或回调中处理边，
      // 或者在这里先计算出保留的节点 ID 列表
      setNodes((currentNodes) => {
          const keptNodes = currentNodes.filter(node => !node.data.isGenerated);
          const keptNodeIds = new Set(keptNodes.map(n => n.id));
          
          setEdges((eds) => eds.filter(edge => keptNodeIds.has(edge.source) && keptNodeIds.has(edge.target)));
          
          return keptNodes;
      });
  };

  const handleLayoutChange = ({ positions, clusters }) => {
    // 接收 Mermaid 渲染后的节点位置和结构信息
    // positions: { "NodeLabel": { x, y, id }, ... }
    // clusters: [ { label, x, y, width, height }, ... ]
    
    // 我们不仅要更新位置，还要确保所有在 Mermaid 中出现的节点都存在于画布中
    
    // 1. 同步节点
    setNodes((prevNodes) => {
      const newNodes = [...prevNodes];
      
      // 标记已存在的节点
      const normalize = (str) => str ? str.trim() : '';
      const existingLabels = new Set(newNodes.map(n => normalize(n.data.label)));
      
      Object.entries(positions).forEach(([label, pos]) => {
        const normalizedLabel = normalize(label);
        if (!existingLabels.has(normalizedLabel)) {
          // 如果是新节点，添加到画布
          // 检查这个节点是否属于某个 Cluster (Subgraph)
          // 这一步比较难，因为 Mermaid DOM 结构没有直接告诉我们节点属于哪个 cluster
          // 但我们可以通过位置判断，或者暂时全部放在顶层，等待用户整理
          
          // 尝试查找父 Cluster
          // 简单碰撞检测：如果节点中心在某个 Cluster 范围内
          // 注意：pos.x/y 是相对于 SVG 的绝对坐标（大概）
          let parentNodeId = undefined;
          
          // 稍微偏移一点，避免重叠
          const newNodeId = `node-gen-${pos.id || Date.now()}-${Math.random().toString(36).substr(2,5)}`;
          
          // 如果是新生成的节点（不在 prevNodes 中），且 allowRandomNodes 为 true，给它一个不同的颜色
          // 注意：这里我们无法直接判断是否是 allowRandomNodes 生成的，但我们可以根据它不在 existingLabels 中推断
          const isNewGenerated = !existingLabels.has(normalizedLabel);
          
          // 如果用户选择不保留随机节点，且该节点是新生成的，则跳过添加
          // 注意：这里需要更细致的逻辑。如果我们在前端只是“显示”了蓝色节点，但用户没有点“保留”，
          // 下次生成时，这些蓝色节点是否应该被视为“预设节点”发送给后端？
          // 根据需求：“用户可以选择是否保留随机生成的蓝色新节点”。
          // 这意味着：生成后，蓝色节点出现。用户如果满意并点击“保留”，它们变白（或保持原样但状态变为永久）；如果不保留（默认），下次生成或清空时它们会消失？
          // 这里的逻辑稍微调整：我们总是显示它们。但我们可以提供一个“一键清除生成节点”的按钮，或者“确认保留”的按钮。
          // 现在的逻辑是：生成后直接添加到画布了。
          
          const newNodeStyle = { 
              background: isNewGenerated && !retainRandomNodes ? '#e6f7ff' : '#fff', // 浅蓝色背景区分新节点，除非已确认保留
              border: isNewGenerated && !retainRandomNodes ? '1px solid #1890ff' : '1px solid #777',
              padding: '10px', 
              borderRadius: '5px', 
              width: 150 
          };

          // 标记这个节点是自动生成的
          const nodeData = { 
              label: label,
              isGenerated: isNewGenerated 
          };

          newNodes.push({
            id: newNodeId,
            data: nodeData,
            position: { x: pos.x, y: pos.y }, // 这里先使用绝对坐标，后续可能需要转换
            type: 'default',
            style: newNodeStyle
          });
        } else {
          // 更新已存在节点的位置
          const nodeIndex = newNodes.findIndex(n => normalize(n.data.label) === normalizedLabel);
          if (nodeIndex !== -1) {
            const node = newNodes[nodeIndex];
            
            // 如果节点已存在，检查是否需要更新样式（例如用户点击了“保留”）
            // 如果 retainRandomNodes 为 true，我们将所有节点的样式恢复为默认
            let newStyle = node.style;
            if (retainRandomNodes && node.data.isGenerated) {
                newStyle = {
                    ...node.style,
                    background: '#fff',
                    border: '1px solid #777'
                };
            }

            // 只有当节点没有父节点时，我们才强制更新位置，避免打乱用户在子图内的布局
            // 或者我们应该更智能地处理
            if (!node.parentNode) {
                newNodes[nodeIndex] = {
                    ...node,
                    position: { x: pos.x, y: pos.y },
                    style: newStyle
                };
            } else {
                // 即使不更新位置，也可能需要更新样式
                newNodes[nodeIndex] = {
                    ...node,
                    style: newStyle
                };
            }
          }
        }
      });
      
      return newNodes;
    });

    // 2. 同步连线 (需要解析 Mermaid 代码)
    // 这里暂时不做，因为从 DOM 获取连线关系比较复杂。
    // 如果需要完全同步，最好是后端返回结构化的 JSON 数据，而不是纯 Mermaid 代码。
    // 目前我们只做节点同步。
  };

  const handleGenerate = async () => {
    if (!description && nodes.length === 0) return;
    setLoading(true);
    setProgress(10); // 初始进度

    // 模拟进度条增加 (因为目前后端没有 WebSocket 推送进度)
    const progressTimer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev;
        return prev + 10;
      });
    }, 500);
    
    // 构建发送给后端的 Prompt 数据
    // 1. 节点列表
    const nodeNames = nodes.map(n => n.data.label);
    
    // 2. 关系列表 (Edges)
    const edgeDescriptions = edges.map(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);
      if (sourceNode && targetNode) {
        return `${sourceNode.data.label} --> ${targetNode.data.label}`;
      }
      return '';
    }).filter(Boolean);

    // 3. 包含关系 (Parent/Child)
    const groupDescriptions = [];
    nodes.forEach(node => {
      if (node.parentNode) {
        const parent = nodes.find(n => n.id === node.parentNode);
        if (parent) {
          groupDescriptions.push(`${parent.data.label} contains ${node.data.label}`);
        }
      }
    });

    const userNodesDescription = [
      ...nodeNames,
      ...edgeDescriptions,
      ...groupDescriptions
    ];

    try {
      const response = await axios.post('http://localhost:8000/generate', {
        prompt: description,
        userNodes: userNodesDescription,
        // 新增参数：是否允许随机节点
        allowRandomNodes: useRandomNodes
      });
      clearInterval(progressTimer);
      setProgress(100);
      
      console.log('Server response:', response.data);
      if (typeof response.data.code === 'string') {
        setMermaidCode(response.data.code);
        // 保存节点详情
        if (response.data.node_details) {
            setNodeDetails(response.data.node_details);
        }
      } else {
        console.error('Unexpected response format:', response.data);
        setMermaidCode(`graph TD\nError[数据格式错误]`);
      }
    } catch (error) {
      clearInterval(progressTimer);
      console.error(error);
      setMermaidCode(`graph TD\nError[生成失败: ${error.message}]`);
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const handleMermaidNodeClick = useCallback((nodeName) => {
      // 阻止冒泡需要在这个回调被调用的地方处理，这里只负责逻辑
      if (nodeName && nodeDetails[nodeName]) {
          setSelectedNodeInfo(nodeDetails[nodeName]);
      } else {
          setSelectedNodeInfo({ name: nodeName, description: '暂无详细描述' });
      }
  }, [nodeDetails]);

  // 开始编辑节点名称
  const startEditing = (node) => {
      setEditingNodeId(node.id);
      setEditingNodeLabel(node.data.label);
  };

  // 完成编辑节点名称
  const finishEditing = () => {
      if (editingNodeId) {
          if (editingNodeLabel.trim()) {
              setNodes((nds) => nds.map((node) => {
                  if (node.id === editingNodeId) {
                      return { ...node, data: { ...node.data, label: editingNodeLabel.trim() } };
                  }
                  return node;
              }));
          }
          setEditingNodeId(null);
          setEditingNodeLabel('');
      }
  };

  const closeNodeDetails = () => {
    setSelectedNodeInfo(null);
  };

  return (
    <div className="app-container" onClick={closeNodeDetails}>
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
              <div className="action-bar" style={{marginTop:0, display: 'flex', alignItems: 'center'}}>
                <input 
                  className="styled-input"
                  style={{
                      marginBottom:0, 
                      flex: 2, 
                      minWidth: 0, 
                      height: '36px',
                      paddingLeft: '10px'
                  }}
                  placeholder="节点名称"
                  value={newNodeName}
                  onChange={(e) => setNewNodeName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addNode()}
                />
                <button 
                    className="btn-add" 
                    onClick={addNode}
                    style={{
                        marginLeft: '5px',
                        whiteSpace: 'nowrap', 
                        flexShrink: 0,
                        height: '36px',
                        padding: '0 10px',
                        backgroundColor: '#217cf3',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: '500',
                        width: 'auto'
                    }}
                >
                    添加
                </button>
                <button 
                    className="btn-clear" 
                    onClick={clearNodes}
                    style={{
                        marginLeft: '5px',
                        whiteSpace: 'nowrap', 
                        flexShrink: 0,
                        height: '36px',
                        padding: '0 10px',
                        backgroundColor: '#ff4d4f',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: '500',
                        width: 'auto'
                    }}
                    title="清空所有节点"
                >
                    清空
                </button>
              </div>

              {/* 随机节点模式开关 */}
              <div style={{marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '5px'}}>
                  <div style={{display: 'flex', alignItems: 'center', fontSize: '14px', color: '#555'}}>
                    <input 
                        type="checkbox" 
                        id="randomMode" 
                        checked={useRandomNodes} 
                        onChange={(e) => setUseRandomNodes(e.target.checked)}
                        style={{marginRight: '8px', cursor: 'pointer'}}
                    />
                    <label htmlFor="randomMode" style={{cursor: 'pointer'}}>
                        允许添加随机节点 (深度生成)
                    </label>
                  </div>
                  
                  {/* 操作生成节点的按钮组 */}
                  {nodes.some(n => n.data.isGenerated) && (
                      <div style={{display: 'flex', gap: '5px', marginTop: '5px'}}>
                          <button 
                            onClick={confirmRetainNodes}
                            style={{
                                flex: 1,
                                fontSize: '12px',
                                padding: '4px',
                                backgroundColor: '#1890ff',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                          >
                            保留新节点
                          </button>
                          <button 
                            onClick={clearGeneratedNodes}
                            style={{
                                flex: 1,
                                fontSize: '12px',
                                padding: '4px',
                                backgroundColor: '#faad14',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                          >
                            移除新节点
                          </button>
                      </div>
                  )}
              </div>
              
              {/* 显示已添加的标签 */}
              <div className="manual-nodes-container" onClick={(e) => e.stopPropagation()}>
                {nodes.length === 0 && <span style={{color:'#ccc', fontSize:'12px'}}>暂无节点</span>}
                {nodes.map(node => (
                  <span key={node.id} className="tag-node" onDoubleClick={() => startEditing(node)}>
                    {editingNodeId === node.id ? (
                        <input 
                            autoFocus
                            value={editingNodeLabel}
                            onChange={(e) => setEditingNodeLabel(e.target.value)}
                            onBlur={finishEditing}
                            onKeyDown={(e) => e.key === 'Enter' && finishEditing()}
                            style={{
                                border: 'none', 
                                outline: 'none', 
                                background: 'transparent', 
                                color: '#333', 
                                width: '60px',
                                fontSize: '12px'
                            }}
                        />
                    ) : (
                        node.data.label
                    )}
                    <span className="tag-remove" onClick={(e) => { e.stopPropagation(); removeNode(node.id); }}>×</span>
                  </span>
                ))}
              </div>

              <button 
                className="btn-primary" 
                style={{marginTop: '15px', position: 'relative', overflow: 'hidden'}}
                onClick={handleGenerate}
                disabled={loading}
              >
                {loading && (
                    <div style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: `${progress}%`,
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        transition: 'width 0.5s ease'
                    }} />
                )}
                {loading ? `生成中 ${progress}%` : '开始生成流程图'}
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

          {/* --- 中间画布 (节点) --- */}
          <div className="panel canvas-panel" ref={reactFlowWrapper} style={{ height: '100%', width: '100%', overflow: 'hidden' }}>
            <div className="card-title" style={{padding: '10px', borderBottom: '1px solid #eee', position: 'absolute', top: 0, left: 0, zIndex: 10, background: 'rgba(255,255,255,0.8)', width: '100%'}}>
              节点编辑区
            </div>
            
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              onPaneClick={closeNodeDetails} // 点击空白处关闭详情
              onNodeDragStop={onNodeDragStop}
              onEdgeDoubleClick={onEdgeDoubleClick}
              onInit={setReactFlowInstance}
              fitView
              attributionPosition="bottom-right"
              deleteKeyCode={['Backspace', 'Delete']}
              nodesDraggable={!loading} // 生成时禁止拖拽
              nodesConnectable={!loading} // 生成时禁止连线
              elementsSelectable={!loading} // 生成时禁止选中
              onNodesDelete={(nodesToDelete) => {
                 // 也可以在这里处理级联删除逻辑
              }}
            >
              <Background color="#aaa" gap={16} />
              <Controls />
              {loading && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, 
                    backgroundColor: 'rgba(255,255,255,0.5)', zIndex: 100, 
                    display: 'flex', justifyContent: 'center', alignItems: 'center'
                }}>
                    <div style={{color: '#333', fontWeight: 'bold'}}>锁定中...</div>
                </div>
              )}
            </ReactFlow>

            {nodes.length === 0 && (
              <div style={{
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', 
                color: '#999', textAlign:'center', zIndex: 5, pointerEvents: 'none'
              }}>
                <p>在左侧添加节点，<br/>然后在画布中拖拽、连线。</p>
              </div>
            )}
          </div>

          {/* --- 右侧边栏 (渲染图) --- */}
          <div className="panel sidebar-right" style={{display: 'flex', flexDirection: 'column'}}>
            <div className="card-title">生成结果</div>
            
            {/* 这里的 MermaidDiagram 是主要的渲染结果 */}
            <div className="mermaid-wrapper" style={{flex: 1, overflow: 'auto'}}>
              {mermaidCode ? (
                <MermaidDiagram 
                    code={mermaidCode} 
                    onLayoutChange={handleLayoutChange} 
                    onNodeClick={handleMermaidNodeClick}
                />
              ) : (
                <div style={{color: '#999', textAlign:'center'}}>
                  <h3>图表空闲</h3>
                  <p>请在左侧输入需求并生成。</p>
                </div>
              )}
            </div>
            
            <div style={{borderTop: '1px solid #eee', padding: '10px', marginTop: 'auto', maxHeight: '300px', overflowY: 'auto'}}>
                 <div className="card-title" style={{fontSize: '0.9rem'}}>节点属性 / 详情</div>
                 <div style={{fontSize: '0.8rem', color: '#666'}}>
                  {selectedNodeInfo ? (
                      <div>
                          <p><strong>名称:</strong> {selectedNodeInfo.name}</p>
                          <p><strong>功能:</strong> {selectedNodeInfo.function || '暂无'}</p>
                          <p><strong>技术方法:</strong> {selectedNodeInfo.technical_methods || '暂无'}</p>
                          {selectedNodeInfo.ports && (
                              <div>
                                  <p><strong>输入:</strong></p>
                                  <ul style={{paddingLeft: '15px', margin: '5px 0'}}>
                                      {selectedNodeInfo.ports.incoming?.map((p, i) => (
                                          <li key={i}>{p.edge_label} (from {p.source_mod || 'Unknown'})</li>
                                      ))}
                                  </ul>
                                  <p><strong>输出:</strong></p>
                                  <ul style={{paddingLeft: '15px', margin: '5px 0'}}>
                                      {selectedNodeInfo.ports.outgoing?.map((p, i) => (
                                          <li key={i}>{p.edge_label} (to {p.target_mod || 'Unknown'})</li>
                                      ))}
                                  </ul>
                              </div>
                          )}
                      </div>
                  ) : (
                      <p>点击图表节点查看详情</p>
                  )}
                 </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;