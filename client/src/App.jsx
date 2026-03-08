// client/src/App.jsx
import React, { useState, useRef, useCallback } from 'react';
import MermaidDiagram from './MermaidDiagram';
import ResultFlowViewer from './ResultFlowViewer';
import './App.css';
import axios from 'axios';
import { toPng } from 'html-to-image';
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
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('开始生成...'); // 进度条状态

  // 多方案状态
  const [plans, setPlans] = useState([]); // 存储所有返回的方案
  const [activePlanIndex, setActivePlanIndex] = useState(0); // 当前激活的方案索引

  const [nodeDetails, setNodeDetails] = useState({}); // 存储后端返回的节点详细信息

  // React Flow 实例
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const reactFlowWrapper = useRef(null);
  const resultFlowRef = useRef(null); // Ref for the right panel viewer

  const [selectedNodeInfo, setSelectedNodeInfo] = useState(null); // 当前选中节点的信息

  // 新增节点详情弹窗状态
  const [isNodeModalOpen, setIsNodeModalOpen] = useState(false);
  const [modalNodeData, setModalNodeData] = useState({ id: null, label: '', description: '', role: '' });

  const [editingNodeId, setEditingNodeId] = useState(null); // 当前正在编辑名称的节点ID
  const [editingNodeLabel, setEditingNodeLabel] = useState(''); // 当前编辑的名称

  const [modificationInput, setModificationInput] = useState(''); // 修改需求的输入

  const [viewMode, setViewMode] = useState('edit'); // 'edit' = 编辑模式, 'result' = 方案全屏展示模式

  // 添加测试
  const [testJsonStr, setTestJsonStr] = useState('');

  const handleTestJson = () => {
    try {
      const json = JSON.parse(testJsonStr);
      let details;
      if (json.module_details) {
        details = json.module_details;
      } else {
        details = json;
      }
      setPlans([{ perspective: '快速测试版', description: json.overall_description || '测试生成的方案', node_details: details }]);
      setActivePlanIndex(0);
      setNodeDetails(details);
      setViewMode('result');
      setTestJsonStr('');
    } catch (err) {
      alert("JSON 解析失败: " + err.message);
    }
  };
  // End

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


  // 打开弹窗准备添加新节点
  const openAddNodeModal = () => {
    setModalNodeData({ id: null, label: newNodeName.trim(), description: '', role: '' });
    setIsNodeModalOpen(true);
  };

  // 保存（新增/编辑）弹窗中的节点信息
  const handleSaveNodeModal = () => {
    if (!modalNodeData.label.trim()) return;

    if (modalNodeData.id) {
      // 更新现有节点
      setNodes((nds) => nds.map((node) => {
        if (node.id === modalNodeData.id) {
          return {
            ...node,
            data: {
              ...node.data,
              label: modalNodeData.label.trim(),
              description: modalNodeData.description.trim(),
              role: modalNodeData.role.trim()
            }
          };
        }
        return node;
      }));
    } else {
      // 插入新节点
      const newNodeId = `node-${Date.now()}`;
      const newNode = {
        id: newNodeId,
        data: {
          label: modalNodeData.label.trim(),
          description: modalNodeData.description.trim(),
          role: modalNodeData.role.trim()
        },
        position: {
          x: Math.random() * 200 + 50,
          y: Math.random() * 200 + 50
        },
        type: 'default',
        style: { background: '#fff', border: '1px solid #777', padding: '10px', borderRadius: '5px', width: 150 }
      };

      setNodes((nds) => [...nds, newNode]);
      setNewNodeName('');
    }
    setIsNodeModalOpen(false);
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
    setPlans([]);
    setActivePlanIndex(0);
    setViewMode('edit');
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
          const newNodeId = `node-gen-${pos.id || Date.now()}-${Math.random().toString(36).substr(2, 5)}`;

          const newNodeStyle = {
            background: '#fff',
            border: '1px solid #777',
            padding: '10px',
            borderRadius: '5px',
            width: 150
          };

          // 标记这个节点
          const nodeData = {
            label: label
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

            let newStyle = node.style;

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
    setProgress(0);
    setPlans([]);
    setMermaidCode('');
    setNodeDetails({});
    setProgressText('开始生成...');
    try {
      const anchorsPayload = nodes.map(n => ({
        user_term: n.data.label,
        description: n.data.description || "",
        role: n.data.role || ""
      }));

      const relationsPayload = edges.map(e => {
        const sourceNode = nodes.find(n => n.id === e.source);
        const targetNode = nodes.find(n => n.id === e.target);
        if (sourceNode && targetNode) {
          return {
            source: sourceNode.data.label,
            target: targetNode.data.label,
            type: "FLOWS_TO"
          };
        }
        return null;
      }).filter(Boolean);

      // 组装 subgraph 关系 (父包含子等同于一种结构)
      nodes.forEach(node => {
        if (node.parentNode) {
          const parent = nodes.find(n => n.id === node.parentNode);
          if (parent) {
            relationsPayload.push({
              source: parent.data.label,
              target: node.data.label,
              type: "CONTAINS"
            });
          }
        }
      });

      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: description,
          anchors: anchorsPayload,
          relations: relationsPayload
        })
      });

      if (!response.body) throw new Error('ReadableStream not yet supported in this browser.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let newPlans = [];
      let isFirstPlanLoaded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunkStr = decoder.decode(value, { stream: true });
        const lines = chunkStr.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const data = JSON.parse(line);

            if (data.type === 'progress') {
              setProgress(data.percent || 0);
              setProgressText(data.message || '生成中...');
            } else if (data.type === 'plan') {
              // Add plan to state
              const initialHistory = [
                { role: 'system', content: `初始方案已生成: ${data.plan_data.perspective}`, timestamp: Date.now() }
              ];
              const planWithHistory = { ...data.plan_data, history: initialHistory };

              newPlans = [...newPlans, planWithHistory];
              setPlans(newPlans);

              if (!isFirstPlanLoaded) {
                isFirstPlanLoaded = true;
                setActivePlanIndex(0);
                setMermaidCode(data.plan_data.mermaid_code || '');
                if (data.plan_data.node_details) {
                  setNodeDetails(data.plan_data.node_details);
                }
                setViewMode('result');
              }
            } else if (data.type === 'error') {
              console.error('Server error:', data.message);
              setMermaidCode(`graph TD\nError[${data.message}]`);
            }
          } catch (e) {
            console.error('Error parsing streaming chunk:', e, line);
          }
        }
      }

    } catch (error) {
      console.error(error);
      setMermaidCode(`graph TD\nError[生成失败: ${error.message}]`);
    } finally {
      setLoading(false);
      setProgressText('');
    }
  };

  const handleEdit = async () => {
    if (!modificationInput.trim()) return;
    if (plans.length === 0 || !plans[activePlanIndex]) return;

    // Store user message immediately
    const userMsg = { role: 'user', content: modificationInput, timestamp: Date.now() };

    // Optimistically update UI to show user message
    setPlans(prevPlans => {
      const updated = [...prevPlans];
      const current = updated[activePlanIndex];
      updated[activePlanIndex] = {
        ...current,
        history: [...(current.history || []), userMsg]
      };
      return updated;
    });

    setLoading(true);
    setProgress(0);
    setProgressText('正在根据您的需求修改方案...');

    try {
      const currentPlan = plans[activePlanIndex];
      // Construct payload
      const payload = {
        plan_id: currentPlan.plan_id, // Pass plan_id
        current_plan: {
          module_details: currentPlan.node_details,
          overall_description: currentPlan.description,
        },
        instruction: modificationInput,
        original_query: description
      };

      // Clear input immediately
      setModificationInput('');

      const response = await fetch('http://localhost:8000/edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.body) throw new Error('ReadableStream not yet supported.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunkStr = decoder.decode(value, { stream: true });
        const lines = chunkStr.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.type === 'progress') {
              setProgress(data.percent || 0);
              setProgressText(data.message || '修改中...');
            } else if (data.type === 'plan') {
              // Update the current plan with the new one
              const newPlanData = data.plan_data;

              setPlans(prevPlans => {
                const updatedPlans = [...prevPlans];
                // Append modification history to perspective if needed, or just update
                // For now, let's keep the perspective but mark as modified
                let newPerspective = newPlanData.perspective;
                if (!newPerspective.includes('(已修改)')) {
                  newPerspective = `(已修改) ${currentPlan.perspective}`;
                }

                // Add system success message
                const sysMsg = { role: 'system', content: `修改完成: ${data.plan_data.description ? data.plan_data.description.substring(0, 50) + '...' : '架构已更新'}`, timestamp: Date.now() };
                const currentHistory = updatedPlans[activePlanIndex].history || [];

                updatedPlans[activePlanIndex] = {
                  ...newPlanData,
                  perspective: newPerspective,
                  history: [...currentHistory, sysMsg]
                };
                return updatedPlans;
              });

              // Update current view immediately
              setMermaidCode(newPlanData.mermaid_code || '');
              if (newPlanData.node_details) {
                setNodeDetails(newPlanData.node_details);
              }

            } else if (data.type === 'error') {
              console.error('Edit error:', data.message);
              // Add error message to history
              setPlans(prevPlans => {
                const updated = [...prevPlans];
                const current = updated[activePlanIndex];
                updated[activePlanIndex] = {
                  ...current,
                  history: [...(current.history || []), { role: 'system', content: `Error: ${data.message}`, isError: true, timestamp: Date.now() }]
                };
                return updated;
              });
              alert(`修改失败: ${data.message}`);
            }
          } catch (e) {
            console.error('Error parsing chunk', e);
          }
        }
      }

    } catch (error) {
      console.error(error);
      alert(`请求失败: ${error.message}`);
      // Add error message to history
      setPlans(prevPlans => {
        const updated = [...prevPlans];
        const current = updated[activePlanIndex];
        updated[activePlanIndex] = {
          ...current,
          history: [...(current.history || []), { role: 'system', content: `Request Error: ${error.message}`, isError: true, timestamp: Date.now() }]
        };
        return updated;
      });
    } finally {
      setLoading(false);
      setProgressText('');
    }
  };

  const handleMermaidNodeClick = useCallback((nodeName) => {
    // 阻止冒泡需要在这个回调被调用的地方处理，这里只负责逻辑
    if (!nodeName) return;

    let foundModule = null;
    if (Array.isArray(nodeDetails)) {
      foundModule = nodeDetails.find(mod => mod.name === nodeName || mod.id === nodeName);
    } else {
      foundModule = nodeDetails[nodeName];
    }

    if (foundModule) {
      setSelectedNodeInfo(foundModule);
      return;
    }

    // Try finding the node as a sub_node in any module
    const entries = Array.isArray(nodeDetails) ? nodeDetails.map(m => [m.name, m]) : Object.entries(nodeDetails);
    for (const [moduleName, moduleInfo] of entries) {
      if (moduleInfo.sub_nodes && Array.isArray(moduleInfo.sub_nodes)) {
        const subNode = moduleInfo.sub_nodes.find(sn => sn.name === nodeName || sn.id === nodeName);
        if (subNode) {
          setSelectedNodeInfo({
            ...subNode,
            isSubNode: true,
            parentModule: moduleName || 'Unknown',
            reference: moduleInfo.reference, // Inherit from parent module
            references: moduleInfo.references,
          });
          return;
        }
      }
    }

    setSelectedNodeInfo({ name: nodeName, description: '暂无详细描述' });
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

  const handleExport = async () => {
    if (!resultFlowRef.current) return;
    if (plans.length === 0 || !plans[activePlanIndex]) return;

    setLoading(true);
    setProgress(0);
    setProgressText('Generating Document...');
    try {
      // 1. Capture React Flow component to PNG
      const dataUrl = await resultFlowRef.current.exportImage();

      const currentPlan = plans[activePlanIndex];

      // 2. Fetch the /export endpoint
      const response = await fetch('http://localhost:8000/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          plan: nodeDetails
        })
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      // 3. Download the resulting document
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `architecture_plan_${new Date().getTime()}.docx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

    } catch (err) {
      console.error("Export Error: ", err);
      alert(`Failed to export document: ${err.message}`);
    } finally {
      setLoading(false);
      setProgressText('');
    }
  };

  return (
    <div className="app-container" onClick={closeNodeDetails}>
      <header>
        <h1>DiaMaid</h1>
      </header>

      <main className="main-content">
        <div className="generator-layout" style={viewMode === 'result' ? { gridTemplateColumns: '320px 1fr' } : {}}>

          {/* --- 左侧边栏 (输入 + 工具 + 代码) --- */}
          <div className="sidebar-left">

            {/* 1. 需求 / 对话 面板 */}
            <div className="left-card" style={viewMode === 'result' ? { flex: '1 1 auto', display: 'flex', flexDirection: 'column' } : { flex: '0 0 auto' }}>
              <div className="card-title">{viewMode === 'result' ? '对话交互 (优化方案)' : '需求描述'}</div>

              {viewMode === 'result' && (
                <div style={{ flex: 1, backgroundColor: '#f5f7fa', borderRadius: '4px', padding: '10px', marginBottom: '10px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', border: '1px solid #eee' }}>
                  {/* Render History */}
                  {plans.length > 0 && plans[activePlanIndex] && plans[activePlanIndex].history && plans[activePlanIndex].history.map((msg, idx) => (
                    <div key={idx} style={{
                      alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                      backgroundColor: msg.role === 'user' ? '#e3f2fd' : (msg.isError ? '#ffebee' : '#fff'),
                      border: msg.role === 'user' ? 'none' : '1px solid #ddd',
                      padding: '8px 12px',
                      borderRadius: msg.role === 'user' ? '8px 8px 0 8px' : '8px 8px 8px 0',
                      maxWidth: '85%',
                      fontSize: '0.9rem',
                      color: msg.isError ? '#c62828' : '#333',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                    }}>
                      {msg.role === 'system' && <strong style={{ display: 'block', fontSize: '0.8rem', color: '#666', marginBottom: '2px' }}>System</strong>}
                      {msg.content}
                    </div>
                  ))}

                  {/* Fallback for empty history or loading state */}
                  {(!plans[activePlanIndex] || !plans[activePlanIndex].history || plans[activePlanIndex].history.length === 0) && (
                    <div style={{ alignSelf: 'center', color: '#999', fontSize: '0.85rem', marginTop: '20px' }}>
                      暂无修改记录
                    </div>
                  )}
                </div>
              )}

              <textarea
                className="styled-input"
                rows={viewMode === 'result' ? 3 : 4}
                placeholder={viewMode === 'result' ? "输入方案优化需求..." : "例如：设计一个包含RAG和LLM的历史教育系统..."}
                value={viewMode === 'result' ? modificationInput : description}
                onChange={(e) => {
                  if (viewMode === 'edit') setDescription(e.target.value);
                  else setModificationInput(e.target.value);
                }}
                disabled={loading}
                style={viewMode === 'result' ? { flexShrink: 0, resize: 'none' } : {}}
              />

              {viewMode === 'result' && (
                <button
                  className="btn-primary"
                  style={{ marginTop: '10px', flexShrink: 0 }}
                  disabled={loading || !modificationInput.trim()}
                  onClick={handleEdit}
                >
                  {loading ? '修改中...' : '发送修改指令'}
                </button>
              )}
            </div>

            {/* 2. 功能面板 (手动节点) */}
            <div className="left-card" style={{ flex: '0 0 auto', display: viewMode === 'result' ? 'none' : 'flex' }}>
              <div className="card-title">功能面板 (预设节点)</div>
              <div className="action-bar" style={{ marginTop: 0, display: 'flex', alignItems: 'center' }}>
                <input
                  className="styled-input"
                  style={{
                    marginBottom: 0,
                    flex: 2,
                    minWidth: 0,
                    height: '36px',
                    paddingLeft: '10px'
                  }}
                  placeholder="节点名称"
                  value={newNodeName}
                  onChange={(e) => setNewNodeName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && openAddNodeModal()}
                />
                <button
                  className="btn-add"
                  onClick={openAddNodeModal}
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


              {/* 显示已添加的标签 */}
              <div className="manual-nodes-container" onClick={(e) => e.stopPropagation()}>
                {nodes.length === 0 && <span style={{ color: '#ccc', fontSize: '12px' }}>暂无节点</span>}
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
                style={{ marginTop: '15px', position: 'relative', overflow: 'hidden' }}
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
                {loading ? `${progressText} ${progress}%` : '开始生成流程图'}
              </button>
            </div>

            <div className="left-card" style={{ flex: '0 0 auto', display: viewMode === 'result' ? 'none' : 'flex' }}>
              <div className="card-title" style={{ fontSize: '0.9rem', color: '#666' }}>快速测试渲染 JSON (跳过后端)</div>
              <textarea
                className="styled-input"
                rows={4}
                value={testJsonStr}
                onChange={(e) => setTestJsonStr(e.target.value)}
                placeholder='在此粘贴包含 module_details 的 arch_plan 直接渲染'
                spellCheck="false"
              />
              <button
                className="btn-primary"
                style={{ marginTop: '10px', backgroundColor: '#f39c12', border: 'none' }}
                onClick={handleTestJson}
              >
                直接渲染测试
              </button>
            </div>
          </div>

          {/* --- 中间画布 (节点) --- */}
          <div className="panel canvas-panel" ref={reactFlowWrapper} style={{ height: '100%', width: '100%', overflow: 'hidden', display: viewMode === 'result' ? 'none' : undefined }}>
            <div className="card-title" style={{ padding: '10px', borderBottom: '1px solid #eee', position: 'absolute', top: 0, left: 0, zIndex: 10, background: 'rgba(255,255,255,0.8)', width: '100%' }}>
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
                  <div style={{ color: '#333', fontWeight: 'bold' }}>锁定中...</div>
                </div>
              )}
            </ReactFlow>

            {nodes.length === 0 && (
              <div style={{
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                color: '#999', textAlign: 'center', zIndex: 5, pointerEvents: 'none'
              }}>
                <p>在左侧添加节点，<br />然后在画布中拖拽、连线。</p>
              </div>
            )}
          </div>

          {/* --- 右侧边栏 (多方案展示区) --- */}
          <div className="panel sidebar-right" style={{ display: 'flex', flexDirection: 'column', flex: viewMode === 'result' ? 1 : '0 0 60%', position: 'relative' }}>
            {/* 工作表 Tabs */}
            <div style={{
              display: 'flex',
              borderBottom: '1px solid #ccc',
              backgroundColor: '#f5f5f5',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div style={{ display: 'flex' }}>
                {plans.length > 0 ? plans.map((plan, index) => (
                  <div
                    key={index}
                    onClick={() => {
                      setActivePlanIndex(index);
                      setMermaidCode(plan.mermaid_code || '');
                      setNodeDetails(plan.node_details || {});
                      setSelectedNodeInfo(null);
                    }}
                    style={{
                      padding: '10px 20px',
                      cursor: 'pointer',
                      backgroundColor: activePlanIndex === index ? '#fff' : 'transparent',
                      borderBottom: activePlanIndex === index ? '2px solid #217cf3' : '2px solid transparent',
                      fontWeight: activePlanIndex === index ? 'bold' : 'normal',
                      color: activePlanIndex === index ? '#217cf3' : '#666'
                    }}
                  >
                    {plan.perspective || `方案 ${index + 1}`}
                  </div>
                )) : (
                  <div style={{ padding: '10px 20px', color: '#999' }}>暂无方案</div>
                )}
              </div>

              {/* 返回/全屏 按钮 */}
              {plans.length > 0 && (
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <button
                    onClick={handleExport}
                    disabled={loading}
                    style={{
                      padding: '6px 15px',
                      backgroundColor: '#28a745',
                      color: '#fff',
                      border: '1px solid #28a745',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: '500'
                    }}
                  >
                    导出为 Word
                  </button>
                  <button
                    onClick={() => setViewMode(viewMode === 'result' ? 'edit' : 'result')}
                    style={{
                      marginRight: '15px',
                      padding: '6px 15px',
                      backgroundColor: viewMode === 'result' ? '#fff' : '#217cf3',
                      color: viewMode === 'result' ? '#217cf3' : '#fff',
                      border: '1px solid #217cf3',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: '500'
                    }}
                  >
                    {viewMode === 'result' ? '← 返回工作区' : '全屏查看方案'}
                  </button>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
              {/* 渲染图部分 */}
              <div className="mermaid-wrapper" style={{ flex: 2, overflow: 'auto', borderRight: '1px solid #eee', padding: '20px', position: 'relative' }}>
                {Object.keys(nodeDetails).length > 0 ? (
                  <div style={{ width: '100%', height: '100%' }}>
                    <ResultFlowViewer
                      ref={resultFlowRef}
                      nodeDetails={nodeDetails}
                      onNodeClick={handleMermaidNodeClick}
                    />
                  </div>
                ) : (
                  <div style={{ color: '#999', textAlign: 'center', marginTop: '50px' }}>
                    <h3>图表空闲</h3>
                    <p>请在左侧输入需求并生成。</p>
                  </div>
                )}
              </div>

              {/* 节点详情部分 */}
              <div style={{ flex: 1, padding: '20px', overflowY: 'auto', backgroundColor: '#fafafa' }}>
                <div className="card-title" style={{ fontSize: '1rem', borderBottom: '1px solid #ddd', paddingBottom: '10px', marginBottom: '15px' }}>
                  总体描述
                </div>
                {plans.length > 0 && plans[activePlanIndex] && plans[activePlanIndex].description && (
                  <div style={{ fontSize: '0.9rem', color: '#555', marginBottom: '20px', whiteSpace: 'pre-wrap' }}>
                    {plans[activePlanIndex].description}
                  </div>
                )}

                <div className="card-title" style={{ fontSize: '1rem', borderBottom: '1px solid #ddd', paddingBottom: '10px', marginBottom: '15px' }}>
                  节点详情
                </div>
                <div style={{ fontSize: '0.9rem', color: '#444' }}>
                  {selectedNodeInfo ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                        <strong style={{ color: '#333' }}>名称:</strong>
                        <div style={{ marginTop: '5px' }}>
                          {selectedNodeInfo.name}
                          {selectedNodeInfo.isSubNode && (
                            <span style={{ fontSize: '0.85rem', color: '#666', marginLeft: '8px' }}>(所属模块: {selectedNodeInfo.parentModule})</span>
                          )}
                        </div>
                      </div>
                      <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                        <strong style={{ color: '#333' }}>功能:</strong>
                        <div style={{ marginTop: '5px' }}>{selectedNodeInfo.function || '暂无'}</div>
                      </div>
                      <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                        <strong style={{ color: '#333' }}>技术方法:</strong>
                        <div style={{ marginTop: '5px' }}>{selectedNodeInfo.technical_methods || '暂无'}</div>
                      </div>
                      {selectedNodeInfo.ports && (
                        <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                          <p><strong style={{ color: '#333' }}>输入:</strong></p>
                          <ul style={{ paddingLeft: '20px', margin: '5px 0', color: '#666' }}>
                            {selectedNodeInfo.ports.incoming?.map((p, i) => (
                              <li key={i}>{p.edge_label} (from {p.source_mod || 'Unknown'})</li>
                            ))}
                            {(!selectedNodeInfo.ports.incoming || selectedNodeInfo.ports.incoming.length === 0) && <li>无</li>}
                          </ul>
                          <p style={{ marginTop: '10px' }}><strong style={{ color: '#333' }}>输出:</strong></p>
                          <ul style={{ paddingLeft: '20px', margin: '5px 0', color: '#666' }}>
                            {selectedNodeInfo.ports.outgoing?.map((p, i) => (
                              <li key={i}>{p.edge_label} (to {p.target_mod || 'Unknown'})</li>
                            ))}
                            {(!selectedNodeInfo.ports.outgoing || selectedNodeInfo.ports.outgoing.length === 0) && <li>无</li>}
                          </ul>
                        </div>
                      )}
                      {/* 展示参考来源 */}
                      {(() => {
                        // 1. Backend injected directly:
                        if (selectedNodeInfo.reference) {
                          return (
                            <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                              <strong style={{ color: '#333' }}>来源:</strong>
                              <div style={{ marginTop: '5px' }}>
                                <div><strong>{selectedNodeInfo.reference.name}</strong></div>
                                {selectedNodeInfo.reference.source && (
                                  <div style={{ fontSize: '0.85rem', color: '#217cf3', marginTop: '5px' }}><strong>Source File: </strong>{selectedNodeInfo.reference.source}</div>
                                )}
                                <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '5px', whiteSpace: 'pre-wrap' }}>{selectedNodeInfo.reference.technical_details}</div>
                              </div>
                            </div>
                          );
                        }

                        // 2. Fallback to plan_data.used_references
                        if (plans.length > 0 && plans[activePlanIndex] && plans[activePlanIndex].used_references) {
                          const modName = selectedNodeInfo.isSubNode ? selectedNodeInfo.parentModule : selectedNodeInfo.name;
                          const usedRefs = plans[activePlanIndex].used_references[modName];
                          if (usedRefs && usedRefs.length > 0) {
                            return (
                              <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                                <strong style={{ color: '#333' }}>来源:</strong>
                                <div style={{ marginTop: '5px' }}>
                                  {usedRefs.map((r, idx) => (
                                    <div key={idx} style={{ marginBottom: '10px' }}>
                                      <div><strong>{r.name}</strong></div>
                                      {r.source && (
                                        <div style={{ fontSize: '0.85rem', color: '#217cf3', marginTop: '5px' }}><strong>Souce File: </strong>{r.source}</div>
                                      )}
                                      <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '5px', whiteSpace: 'pre-wrap' }}>{r.text}</div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            );
                          }
                        }

                        // 3. Fallback to raw string value from json
                        const rawRefs = selectedNodeInfo.references;
                        if (rawRefs && String(rawRefs).toLowerCase() !== 'none' && String(rawRefs).toLowerCase() !== 'null') {
                          return (
                            <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                              <strong style={{ color: '#333' }}>来源 (仅名称匹配):</strong>
                              <div style={{ marginTop: '5px' }}>{rawRefs}</div>
                            </div>
                          );
                        }

                        // 4. No references found
                        return (
                          <div style={{ backgroundColor: '#fff', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                            <strong style={{ color: '#333' }}>来源:</strong>
                            <div style={{ marginTop: '5px' }}>无相关源论文</div>
                          </div>
                        );
                      })()}
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: '#999', padding: '20px' }}>
                      点击左侧图表节点查看详情
                    </div>
                  )}
                </div>
              </div>
            </div>


          </div>

        </div>
      </main>

      {/* Node Details Modal */}
      {isNodeModalOpen && (
        <div className="modal-overlay" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
          <div className="modal-content" style={{ backgroundColor: '#fff', padding: '20px', borderRadius: '8px', width: '400px', maxWidth: '90%' }}>
            <h3 style={{ marginTop: 0 }}>{modalNodeData.id ? '编辑节点' : 'Add Node'}</h3>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>Name:</label>
              <input
                className="styled-input"
                value={modalNodeData.label}
                onChange={e => setModalNodeData({ ...modalNodeData, label: e.target.value })}
                autoFocus
              />
            </div>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>Description:</label>
              <textarea
                className="styled-input"
                rows={3}
                value={modalNodeData.description}
                onChange={e => setModalNodeData({ ...modalNodeData, description: e.target.value })}
              />
            </div>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>Role (Optional):</label>
              <input
                className="styled-input"
                value={modalNodeData.role}
                onChange={e => setModalNodeData({ ...modalNodeData, role: e.target.value })}
                placeholder="例如: processing_core"
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button className="btn-clear" style={{ padding: '8px 15px', border: '1px solid #ccc', background: '#f5f5f5', borderRadius: '4px', cursor: 'pointer' }} onClick={() => setIsNodeModalOpen(false)}>Cancel</button>
              <button className="btn-primary" style={{ padding: '8px 15px', background: '#217cf3', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }} onClick={handleSaveNodeModal}>Save</button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;