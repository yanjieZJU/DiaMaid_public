// client/src/App.jsx
import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import X6FlowViewer from './X6FlowViewer';
import './App.css';
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '/api';

const plan_placeholder =  {
        "perspective": "多模态异常上报型",
        "overall_description": "本系统为现代化物流仓库一线操作员打造高流畅度、低中断的智能拣货补货协作工作流。其核心创新体现在多模态异常一键上报、智能任务优先级排序、动态任务可视化、操作员实时压力反馈四大能力的深度融合，通过智能排序与视觉引导，动态适配仓库现场状态和操作员工作节奏，有效减少任务打断、优先级混乱与异常积压，全面提升拣补货高峰期的作业效率与体验。",
        "module_details": [
          {
            "name": "多模态异常快速上报输入",
            "brief_description": "支持语音、扫码、触摸等多方式让操作员快速上一键报告异常。",
            "function": "为操作员提供语音、扫码、触摸等多模态输入入口，极大简化缺货、错放、溢出等异常的上报流程，确保异常情况在作业高压下能被及时、低成本记入系统。",
            "technical_methods": "集成安卓基础触控与语音识别引擎，允许在任意主屏或任务节点上快速呼出异常上报入口。扫码可由专用硬件或手机摄像头调用，语音输入则由实时语音识别模块处理，触控通过UI快速按钮完成。所有输入转为结构化异常事件，统一推送至异常处理流。",
            "role": "input",
            "references": "Multimodal_Input_Handler,User_Interaction",
            "sub_nodes": [
              {
                "name": "语音快速上报",
                "function": "识别并解析操作员的语音异常说明。",
                "technical_methods": "采用本地和云端ASR混合模型实时转写异常内容，关键词抽取辅助分类异常类型。"
              },
              {
                "name": "扫码/拍照快速上报",
                "function": "通过扫码或拍照捕获商品/货架异常。",
                "technical_methods": "调用摄像头读取商品条码或拍照，结合OCR/条码解析快速定位异常目标。"
              },
              {
                "name": "触摸一键异常上报",
                "function": "通过触控界面快速选择常见异常类型。",
                "technical_methods": "弹出预设异常选项的快捷触控面板，支持一键多选及备注补充。"
              }
            ],
            "ports": {
              "incoming": [],
              "outgoing": [
                {
                  "target_mod": "异常事件识别与归档",
                  "target_node": null,
                  "originates_from_internal_node": null,
                  "edge_label": "结构化异常输入事件"
                }
              ],
              "internal_flow": [
                {
                  "source_node": "语音快速上报",
                  "target_node": null,
                  "edge_label": "语音转为异常事件"
                },
                {
                  "source_node": "扫码/拍照快速上报",
                  "target_node": null,
                  "edge_label": "扫码/拍照转为异常事件"
                },
                {
                  "source_node": "触摸一键异常上报",
                  "target_node": null,
                  "edge_label": "触控转为异常事件"
                }
              ]
            },
            "reference": {
              "name": "Multimodal_Input_Handler",
              "technical_details": "Module: Multimodal_Input_Handler\nSource: Voice and Touch Based Error-tolerant Multimodal Text Editing and Correction for Smartphones\nCommunity Context: ```\nThe community collectively functions as an Interactive Control and Input Interface, enabling users to engage with various systems through diverse input modalities, including physical gestures, voice commands, and tactile interactions. Typical Inputs: User movements, voice commands, and physical actions; Outputs: Control signals, crafted objects, and visual feedback. Innovative Highlights: Utilizes advanced deep learning models for speech recognition and integrates real-time physical crafting tools, such as 3D printing pens, to create tangible artifacts, setting it apart from conventional input systems. The assigned role: 'sensing_input:direct_input'\n```\nText content: Module: Multimodal_Input_Handler\nRole: sensing_input:direct_input\nFunction: Captures user's touch gestures (t) and voice utterances (s) from the smartphone interface. Touch input can be tapping to reposition the caret or gliding a finger to select text. Voice input is the spoken command or content.\nTechnical Details: Supports common touch input actions provided by Android EditText view. Voice input can start before, during, or after touch actions. The user can signal voice input by tapping/long-clicking text or using a microphone button.\nInput: User touch (tap, glide) and voice utterance.\nOutput: Multimodal input event X = <t, s> (touch input 't', voice input 's').\n--- Context ---\nProceded By: [None]\nFollowed By: ['Speech_Recognizer', 'Operation_Classifier']\n",
              "source": "Voice and Touch Based Error-tolerant Multimodal Text Editing and Correction for Smartphones"
            }
          },
          {
            "name": "异常事件识别与归档",
            "brief_description": "统一处理各类异常上报，生成结构化可追踪异常任务。",
            "function": "对多模态输入的异常进行分类、去重与关联货物/订单，生成可追踪的异常任务流，供后续任务编排与优先级排序。",
            "technical_methods": "结合关键词匹配、目标ID识别（如货架/商品码），自动归类异常（缺货、错放、损坏等），并与当前订单和货位状态关联。实现异常事件去重与状态更新，保障异常响应的时效性和唯一性。",
            "role": "processing",
            "references": "None",
            "sub_nodes": [
              {
                "name": "异常类型归类",
                "function": "将输入的异常分为标准类型。",
                "technical_methods": "基于关键词库和文本分类，对语音/文本描述进行自动类型判别。"
              },
              {
                "name": "异常任务生成",
                "function": "为每条归类异常生成可追踪的异常任务。",
                "technical_methods": "提取货物、货架、时间、责任人等元数据，生成标准异常任务记录。"
              }
            ],
            "ports": {
              "incoming": [
                {
                  "source_mod": "多模态异常快速上报输入",
                  "source_node": null,
                  "connects_to_internal_node": null,
                  "edge_label": "结构化异常输入事件"
                }
              ],
              "outgoing": [
                {
                  "target_mod": "智能任务优先级动态分析",
                  "target_node": null,
                  "originates_from_internal_node": null,
                  "edge_label": "新增异常任务流"
                }
              ],
              "internal_flow": [
                {
                  "source_node": "异常类型归类",
                  "target_node": "异常任务生成",
                  "edge_label": "归类后的异常转为任务"
                }
              ]
            },
            "reference": null
          },
          {
            "name": "智能任务优先级动态分析",
            "brief_description": "实时分析任务优先级和顺序，动态推荐最优任务流。",
            "function": "分析当前所有拣货、补货、异常等任务，结合仓库状态、操作员位置、订单紧急度，动态生成最优任务执行序列并实时调整，最大化操作流畅性与效率。",
            "technical_methods": "核心采用层级有限状态自动机管理任务状态，融合强化学习（如PPO或DQN）实现任务流优选。综合实时位置、订单时效、操作员状态和异常堆积度，以规则和机器学习混合方式动态决策任务排序。支持“人-机”子任务协同分配，确保高优先级任务和易中断任务优先推送。",
            "role": "processing",
            "references": "Task_Supervisor,Task_Collaboration",
            "sub_nodes": [
              {
                "name": "任务状态感知",
                "function": "实时收集任务、仓库、人员与异常状态。",
                "technical_methods": "对接仓库管理系统，周期性拉取任务队列、货架/库存状态与操作员位置。"
              },
              {
                "name": "优先级排序与动态决策",
                "function": "依据多维度信息生成最优任务排序。",
                "technical_methods": "基于多条件决策树+机器学习优化，动态调整任务流并自动插入高优异常/补货任务。"
              },
              {
                "name": "子任务分配与协作建议",
                "function": "合理划分任务，给出协作建议（如人机分工）。",
                "technical_methods": "分析任务物理位置、操作员状态与剩余负载，自动分派/建议分工。"
              }
            ],
            "ports": {
              "incoming": [
                {
                  "source_mod": "异常事件识别与归档",
                  "source_node": null,
                  "connects_to_internal_node": "任务状态感知",
                  "edge_label": "异常任务流"
                },
                {
                  "source_mod": "操作员压力自评与节奏反馈",
                  "source_node": null,
                  "connects_to_internal_node": "优先级排序与动态决策",
                  "edge_label": "压力与节奏反馈"
                }
              ],
              "outgoing": [
                {
                  "target_mod": "动态可视化任务流程板",
                  "target_node": null,
                  "originates_from_internal_node": null,
                  "edge_label": "优化排序后任务流"
                }
              ],
              "internal_flow": [
                {
                  "source_node": "任务状态感知",
                  "target_node": "优先级排序与动态决策",
                  "edge_label": "实时状态供排序"
                },
                {
                  "source_node": "优先级排序与动态决策",
                  "target_node": "子任务分配与协作建议",
                  "edge_label": "排序结果用于协作建议"
                }
              ]
            },
            "reference": {
              "name": "Task_Supervisor",
              "technical_details": "Module: Task_Supervisor\nSource: Human Decision-Making Modeling and Cooperative Controller Design for Human–Agent Interaction Systems\nCommunity Context: ```\nThe community collectively functions as an Adaptive Decision-Making Framework, integrating advanced reinforcement learning and real-time adaptation strategies to optimize interactions and actions in dynamic environments. Typical Inputs: User touch locations, conversational cues, and environmental data; Outputs: Contextualized actions and adaptive responses for humanoid robots and conversational agents. Innovative Highlights: Utilizes Proximal Policy Optimization and Deep Q-Learning for adaptive learning, alongside advanced event-based control mechanisms to ensure stability and responsiveness in real-time applications. The assigned role: 'processing_core: decision_logic'\n```\nText content: Module: Task_Supervisor\nRole: processing_core:decision_logic\nFunction: Manages the priority of different tasks (human intervention, move-to-target, collision avoidance) using a hierarchical finite state automata supervisor. It determines which task is active based on predefined triggering conditions and human intervention timing.\nTechnical Details: Employs a hierarchical finite state automata supervisor with states S1 (human intervention), S2 (move-to-target), and S3 (collision avoidance). State transitions are governed by conditions Rθ2 (e.g., R1 for collision avoidance, R3 for local minima, R4/R5 for human task completion). It receives TDj to activate human intervention (S1).\nInput: TDj (human intervention timing), xpj (agent position for collision conditions)\nOutput: Task_Priority_Output (active task and its priority)\n--- Context ---\nProceded By: ['HDDM_Threshold']\nFollowed By: ['NSBC']\n",
              "source": "Human Decision-Making Modeling and Cooperative Controller Design for Human–Agent Interaction Systems"
            }
          },
          {
            "name": "操作员压力自评与节奏反馈",
            "brief_description": "操作员可一键反馈压力和节奏，系统自适应调整任务推送。",
            "function": "为操作员提供简易快速的压力与节奏自评入口（如滑块/表情），系统根据反馈自动调整后续任务推送频率和难度，防止持续超负荷工作。",
            "technical_methods": "界面呈现可一键评价的压力与节奏选择（如当前舒适/有点快/压力大），后台记录后计算滚动均值及趋势。当压力高于阈值自动降低任务推送频率，必要时弹窗建议休息或临时调岗。反馈数据实时回流到任务排序引擎，影响优先级决策。",
            "role": "input",
            "references": "None",
            "sub_nodes": [],
            "ports": {
              "incoming": [],
              "outgoing": [
                {
                  "target_mod": "智能任务优先级动态分析",
                  "target_node": null,
                  "originates_from_internal_node": null,
                  "edge_label": "压力与节奏反馈"
                }
              ],
              "internal_flow": []
            },
            "reference": null
          },
          {
            "name": "动态可视化任务流程板",
            "brief_description": "以图形化界面高亮展示任务、优先级与异常提醒。",
            "function": "将当前任务列表、状态、优先级变化和异常事件以可视化方式（如甘特图、任务卡片、状态高亮）实时展示，支持交互查看详情、异常一键跳转，显著降低认知负担。",
            "technical_methods": "基于Web（D3.js/HTML5）或App原生实现，实时渲染任务序列，异常任务高亮提示，支持多种视图切换（列表、时间轴、热力图等）。与输入端联动，允许操作员点击任务查看详情或一键上报新异常。任务状态与优先级变动采用动画/色彩高亮反馈，增强信息直观性。",
            "role": "output",
            "references": "Display_Interface,Overview_Dashboard",
            "sub_nodes": [
              {
                "name": "任务与优先级展示",
                "function": "清晰展示所有任务及排序优先级。",
                "technical_methods": "使用颜色、排序、图标等多维度视觉编码，辅以动态刷新。"
              },
              {
                "name": "异常与堆积高亮提醒",
                "function": "对异常和任务堆积情况进行高亮及弹窗提示。",
                "technical_methods": "异常任务以警告色标注，并可点击快速查看/处理。"
              },
              {
                "name": "交互式详情与快捷操作",
                "function": "支持点击任务卡片查看详情或一键触发异常上报等操作。",
                "technical_methods": "交互式UI组件，数据联动至多模态上报入口。"
              }
            ],
            "ports": {
              "incoming": [
                {
                  "source_mod": "智能任务优先级动态分析",
                  "source_node": null,
                  "connects_to_internal_node": "任务与优先级展示",
                  "edge_label": "优化排序后任务流"
                }
              ],
              "outgoing": [
                {
                  "target_mod": "多模态异常快速上报输入",
                  "target_node": null,
                  "originates_from_internal_node": "交互式详情与快捷操作",
                  "edge_label": "一键异常上报入口"
                },
                {
                  "target_mod": "操作员压力自评与节奏反馈",
                  "target_node": null,
                  "originates_from_internal_node": null,
                  "edge_label": "压力自评快捷入口"
                }
              ],
              "internal_flow": [
                {
                  "source_node": "任务与优先级展示",
                  "target_node": "异常与堆积高亮提醒",
                  "edge_label": "异常高亮传递"
                },
                {
                  "source_node": "异常与堆积高亮提醒",
                  "target_node": "交互式详情与快捷操作",
                  "edge_label": "弹窗至交互入口"
                }
              ]
            },
            "reference": {
              "name": "Display_Interface",
              "technical_details": "Module: Display_Interface\nSource: Effect of Imperfect Information and Action Automation on Attentional Allocation\nCommunity Context: ```\nThe community collectively functions as an Augmented Reality Visualization and Interaction Suite, enabling dynamic display and manipulation of 3D models across various applications, from cultural artifacts to human poses. Typical Inputs: 3D model data, user interactions; Outputs: Interactive visual representations and real-time feedback on 3D designs. Innovative Highlights: Utilizes close-range photogrammetry for high-fidelity model rendering and integrates advanced AR techniques for synchronized visual overlays and real-time pose manipulation, setting it apart from conventional 3D display systems. The assigned role: 'interface_output:gui_display'\n```\nText content: Module: Display_Interface\nRole: interface_output:gui_display\nFunction: Presents the visual output of the MATB simulation environment to the human operator, including the states of the tracking, resources management, and system monitoring tasks, as well as the 'Warning' signal from the information automation.\nTechnical Details: A 17-inch monitor (1440 x 900 pixels) displaying the OpenMATB interface. The 'Warning' button turns red when information automation is functioning and a gauge failure occurs.\nInput: Tracking_Task_Visual_State (from OpenMATB_Simulation), Resources_Management_Visual_State (from OpenMATB_Simulation), Monitoring_Task_Visual_State (from OpenMATB_Simulation), Warning_Signal (from Information_Automation_System)\nOutput: Visual_Feedback_to_Human (to Human Operator - implicit)\n--- Context ---\nProceded By: ['OpenMATB_Simulation', 'Information_Automation_System']\nFollowed By: [None]\n",
              "source": "Effect of Imperfect Information and Action Automation on Attentional Allocation"
            }
          },
          {
            "name": "任务执行与状态反馈",
            "brief_description": "操作员按任务指引完成操作，并实时反馈进展状态。",
            "function": "指导操作员按推荐顺序执行拣货、补货、异常处理任务，提供任务完成、暂停、遇阻等多状态反馈按钮，实现任务闭环。",
            "technical_methods": "每个任务节点均有“完成/暂停/失败/异常”一键反馈入口。状态变化自动同步至后端，实时更新任务流与优先级决策。支持扫码/拍照确认任务完成，提升数据可靠性。",
            "role": "input",
            "references": "None",
            "sub_nodes": [],
            "ports": {
              "incoming": [
                {
                  "source_mod": "动态可视化任务流程板",
                  "source_node": null,
                  "connects_to_internal_node": null,
                  "edge_label": "任务详情与操作入口"
                }
              ],
              "outgoing": [
                {
                  "target_mod": "智能任务优先级动态分析",
                  "target_node": null,
                  "originates_from_internal_node": null,
                  "edge_label": "最新任务状态与反馈"
                }
              ],
              "internal_flow": []
            },
            "reference": null
          }
        ]
      }

const plan_placeholder2 = {
  "perspective": "智能调度优先型",
  "overall_description": "本系统以智能任务调度与动态优先级引擎为核心，为仓库一线操作员提供高效的拣货补货决策支持。通过实时感知仓库状态、操作员位置与任务积压情况，动态重排任务队列，最大化作业吞吐量，减少空走路程与等待时间，同时以简洁的可视化界面让操作员始终掌握任务全局视野。",
  "module_details": [
    {
      "name": "实时仓库状态感知",
      "brief_description": "持续采集货架、库存、操作员位置等实时数据，作为调度决策的输入源。",
      "function": "通过IoT传感器、扫码记录和操作员位置追踪，实时构建仓库状态快照，为调度引擎提供决策基础数据。",
      "technical_methods": "集成RFID/条码扫描事件流、操作员手持设备GPS/蓝牙定位、货架重量传感器数据，统一接入消息队列，以100ms级延迟更新状态快照。",
      "role": "input",
      "references": "Warehouse_State_Monitor,IoT_Integration",
      "sub_nodes": [
        {
          "name": "操作员位置追踪",
          "function": "实时获取每位操作员在仓库中的位置。",
          "technical_methods": "蓝牙信标+手持设备融合定位，精度2米以内。"
        },
        {
          "name": "库存变动监听",
          "function": "监听拣货、补货、退货等事件引起的库存变化。",
          "technical_methods": "订阅WMS库存变动事件流，实时同步至状态快照。"
        }
      ],
      "ports": {
        "incoming": [],
        "outgoing": [
          {
            "target_mod": "智能任务调度引擎",
            "target_node": null,
            "originates_from_internal_node": null,
            "edge_label": "仓库状态快照"
          }
        ],
        "internal_flow": [
          {
            "source_node": "操作员位置追踪",
            "target_node": "库存变动监听",
            "edge_label": "位置与库存联合更新"
          }
        ]
      }
    },
    {
      "name": "智能任务调度引擎",
      "brief_description": "动态计算最优任务分配方案，最小化路径与等待时间。",
      "function": "综合任务紧急度、操作员位置、货架距离等因素，实时生成最优任务队列，并在状态变化时自动重调度。",
      "technical_methods": "采用贪心+动态规划混合算法进行任务路径优化，支持软硬约束配置（如优先级锁定、区域隔离）。调度结果以差量推送方式下发至操作员终端，避免全量刷新。",
      "role": "processing_core",
      "references": "Task_Scheduler,Priority_Engine",
      "sub_nodes": [
        {
          "name": "优先级评分模块",
          "function": "为每个待处理任务计算综合优先级分数。",
          "technical_methods": "加权评分：订单截止时间权重40%，货架距离权重30%，任务类型权重30%。"
        },
        {
          "name": "路径优化模块",
          "function": "在优先级排序基础上进一步优化拣货路径，减少空走。",
          "technical_methods": "改良TSP算法，结合仓库通道地图实时计算最短可行路径。"
        }
      ],
      "ports": {
        "incoming": [
          {
            "source_mod": "实时仓库状态感知",
            "source_node": null,
            "edge_label": "仓库状态快照"
          }
        ],
        "outgoing": [
          {
            "target_mod": "任务可视化与操作员界面",
            "target_node": null,
            "originates_from_internal_node": null,
            "edge_label": "优化后任务队列"
          }
        ],
        "internal_flow": [
          {
            "source_node": "优先级评分模块",
            "target_node": "路径优化模块",
            "edge_label": "优先级排序结果"
          }
        ]
      }
    },
    {
      "name": "任务可视化与操作员界面",
      "brief_description": "以清晰直观的方式将调度结果呈现给操作员，支持快速确认与反馈。",
      "function": "将调度结果以列表、地图、进度条等多维方式展示，操作员可查看当前任务、下一任务及全局进度，并可标记完成或反馈异常。",
      "technical_methods": "React Native移动端应用，WebSocket实时接收任务更新推送，支持离线缓存保障弱网下可用。地图视图基于仓库SVG底图叠加动态任务标注。",
      "role": "output",
      "references": "Task_Visualization,Operator_UI",
      "sub_nodes": [
        {
          "name": "任务列表视图",
          "function": "显示操作员当前待完成任务的有序列表。",
          "technical_methods": "优先级排序列表，支持滑动完成手势与一键跳过。"
        },
        {
          "name": "仓库地图导航",
          "function": "在仓库地图上高亮标注下一目标货架位置。",
          "technical_methods": "SVG矢量仓库地图，动态绘制推荐路径箭头。"
        }
      ],
      "ports": {
        "incoming": [
          {
            "source_mod": "智能任务调度引擎",
            "source_node": null,
            "edge_label": "优化后任务队列"
          }
        ],
        "outgoing": [],
        "internal_flow": [
          {
            "source_node": "任务列表视图",
            "target_node": "仓库地图导航",
            "edge_label": "当前任务定位"
          }
        ]
      }
    }
  ]
};

// Extract top N processing modules from node_details for plan preview
function getProcessingModules(nodeDetails, maxCount = 3) {
  if (!nodeDetails) return [];
  const entries = Array.isArray(nodeDetails)
    ? nodeDetails.map(m => [m.id || m.name, m])
    : Object.entries(nodeDetails);
  const processing = entries
    .map(([, m]) => m)
    .filter(m => {
      const role = String(m.role || '').toLowerCase();
      return role.includes('processing') || role.includes('core') || role.includes('inference') || role.includes('reasoning');
    });
  return processing.slice(0, maxCount).map(m => m.name || m.id || '');
}

// Get first sentence of a description string
function firstSentence(text) {
  if (!text) return '';
  const m = String(text).match(/^[^。！？.!?]+[。！？.!?]?/);
  return m ? m[0].trim() : String(text).slice(0, 60);
}

// 初始节点
const initialNodes = [];
const initialEdges = [];

function App() {
  const [description, setDescription] = useState('');
  const [sessionId, setSessionId] = useState(''); // Added session ID

  // 使用 React Flow 的 hook 管理状态
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const [newNodeName, setNewNodeName] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('开始生成...'); // 进度条状态

  // 多方案状态
  const [plans, setPlans] = useState([]); // 存储所有返回的方案
  const [activePlanIndex, setActivePlanIndex] = useState(0); // 当前激活的方案索引
  const [isMorePerspectivesOpen, setIsMorePerspectivesOpen] = useState(false);
  const [isCompareOpen, setIsCompareOpen] = useState(false);
  const [comparePanelWidth, setComparePanelWidth] = useState(320);
  const compareDragRef = React.useRef(null);
  // Modules quoted from other plans, to be attached as context to the next edit message
  // Each entry: { planIndex, planName, moduleName, moduleData }
  const [quotedModules, setQuotedModules] = useState([]);
  const plansRef = useRef(plans);
  const nodeDetailsSigRef = useRef('');

  const [nodeDetails, setNodeDetails] = useState({}); // 存储后端返回的节点详细信息
  const resolvedNodeDetails = useMemo(() => {
    const hasNodeDetails = Array.isArray(nodeDetails)
      ? nodeDetails.length > 0
      : !!(nodeDetails && Object.keys(nodeDetails).length > 0);
    if (hasNodeDetails) return nodeDetails;
    const currentPlan = plans[activePlanIndex];
    const fromPlan = currentPlan?.node_details ?? currentPlan?.module_details;
    if (typeof fromPlan !== 'undefined') return fromPlan;
    return nodeDetails;
  }, [nodeDetails, plans, activePlanIndex]);
  const detailsRef = useRef(resolvedNodeDetails);
  useEffect(() => {
    detailsRef.current = resolvedNodeDetails;
  }, [resolvedNodeDetails]);
  
  useEffect(() => {
    plansRef.current = plans;
  }, [plans]);

  // Trigger KG warmup when the generate page loads
  useEffect(() => {
    fetch(`${API_BASE}/warmup/kg`, { method: 'POST' }).catch(() => {});
  }, []);

  useEffect(() => {
    const currentPlan = plansRef.current[activePlanIndex];
    if (currentPlan) {
      const detailsFromPlan = currentPlan.node_details ?? currentPlan.module_details;
      console.log('Syncing plan to UI:', activePlanIndex, currentPlan, 'detailsFromPlan:', detailsFromPlan ? 'present' : 'absent');
      if (typeof detailsFromPlan !== 'undefined') {
        let sig = '';
        try {
          sig = JSON.stringify(detailsFromPlan);
        } catch (e) { void e; }
        if (!sig || sig !== nodeDetailsSigRef.current) {
          nodeDetailsSigRef.current = sig;
          setNodeDetails(detailsFromPlan);
        }
      }
      setSelectedNodeInfo(null);
    }
  }, [activePlanIndex]);

  // React Flow 实例
  const reactFlowWrapper = useRef(null);
  const resultFlowRef = useRef(null); // Ref for the right panel viewer

  const [selectedNodeInfo, setSelectedNodeInfo] = useState(null); // 当前选中节点的信息
  const [nodeDetailDraft, setNodeDetailDraft] = useState({ function: '', technical_methods: '' });
  const [isNodeDetailEditing, setIsNodeDetailEditing] = useState(false);

  // 新增节点详情弹窗状态
  const [isNodeModalOpen, setIsNodeModalOpen] = useState(false);
  const [modalNodeData, setModalNodeData] = useState({ id: null, label: '', description: '', role: '' });

  const [editingNodeId, setEditingNodeId] = useState(null); // 当前正在编辑名称的节点ID
  const [editingNodeLabel, setEditingNodeLabel] = useState(''); // 当前编辑的名称

  const [modificationInput, setModificationInput] = useState(''); // 修改需求的输入

  const [viewMode, setViewMode] = useState('edit'); // 'edit' = 编辑+视角卡片, 'result' = 全屏方案展示

  // Divergent perspectives state (populated before full plan generation)
  const [divergentPerspectives, setDivergentPerspectives] = useState([]); // [{perspective, rationale, additional_anchors}]
  const [planProgressMap, setPlanProgressMap] = useState({}); // index -> {percent, done}

  // 添加测试
  const [testJsonStr, setTestJsonStr] = useState('');
  const [planThumbnails, setPlanThumbnails] = useState({});

  const handleTestJson = (overrideStr) => {
    try {
      const json = JSON.parse(overrideStr !== undefined ? overrideStr : testJsonStr);
      const items = Array.isArray(json) ? json : [json];
      const testPlans = items.map((item, i) => {
        const details = item.module_details || item;
        return {
          perspective: item.perspective || `测试方案${i + 1}`,
          description: item.overall_description || item.description || '测试生成的方案',
          node_details: details,
          mermaid_code: item.mermaid_code || '',
          plan_id: item.plan_id || `test-plan-${Date.now()}-${i}`,
          history: item.history || [],
          used_references: item.used_references || {}
        };
      });

      setPlans(testPlans);
      setActivePlanIndex(0);
      setNodeDetails(testPlans[0].node_details);
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
    setNewNodeName('');
    setNodeDetails({});
    setSelectedNodeInfo(null);
    setPlans([]);
    setActivePlanIndex(0);
    setViewMode('edit');
  };



  const handleGenerate = async () => {
    console.log('[DEBUG] handleGenerate called, description:', description, 'nodes:', nodes.length);
    if (!description && nodes.length === 0) { console.log('[DEBUG] early return — empty'); return; }
    setLoading(true);
    setProgress(0);
    setViewMode('edit');
    setPlans([]);
    setNodeDetails({});
    nodeDetailsSigRef.current = '';
    setProgressText('开始生成...');
    setDivergentPerspectives([]);
    setPlanProgressMap({});
    thumbnailCycleRef.current = false;
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

      console.log('[DEBUG] about to fetch, API_BASE:', API_BASE);
      const response = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId.trim() || undefined,
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
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);

            if (data.type === 'progress') {
              setProgress(data.percent || 0);
              setProgressText(data.message || '生成中...');
              // Track per-plan progress: estimate based on overall progress and plans completed
              setPlanProgressMap(prev => {
                const completedCount = Object.values(prev).filter(v => v && v.done).length;
                const inProgressIdx = completedCount; // next slot being worked on
                const updated = { ...prev };
                if (!updated[inProgressIdx] || !updated[inProgressIdx].done) {
                  updated[inProgressIdx] = { percent: data.percent || 0, done: false };
                }
                return updated;
              });
            } else if (data.type === 'divergent_perspectives') {
              // Show perspectives side-by-side in the right panel
              setDivergentPerspectives(data.perspectives || []);
            } else if (data.type === 'warning') {
              // Loud surface for KG/embedding failures — without this, silent
              // degradation looks like a "normal" plan with no references.
              setProgressText(data.message || '');
              console.warn('[server warning]', data.message);
            } else if (data.type === 'plan') {
              // Add plan to state
              const initialHistory = [
                { role: 'system', content: `初始方案已生成: ${data.plan_data.perspective}`, timestamp: Date.now() }
              ];
              const planWithHistory = { ...data.plan_data, history: initialHistory };

              newPlans = [...newPlans, planWithHistory];
              setPlans(newPlans);
              // Mark this perspective slot as done
              const planIdx = newPlans.length - 1;
              setPlanProgressMap(prev => ({ ...prev, [planIdx]: { percent: 100, done: true } }));

              if (!isFirstPlanLoaded) {
                isFirstPlanLoaded = true;
                setActivePlanIndex(0);
              }
            } else if (data.type === 'error') {
              console.error('Server error:', data.message);
            }
          } catch (e) {
            console.error('Error parsing streaming chunk:', e, line);
          }
        }
      }

    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
      setProgressText('');
    }
  };

  const handleEdit = async () => {
    if (!modificationInput.trim()) return;
    if (plans.length === 0 || !plans[activePlanIndex]) return;

    // Snapshot quoted modules before clearing
    const referencedModules = quotedModules.map(q => ({
      plan_name: q.planName,
      module_name: q.moduleName,
      module_data: q.moduleData,
    }));

    // Store user message with quoted module context shown inline
    const quotedSummary = referencedModules.length > 0
      ? `\n\n[引用模块: ${referencedModules.map(r => `${r.plan_name}·${r.module_name}`).join('、')}]`
      : '';
    const userMsg = { role: 'user', content: modificationInput + quotedSummary, timestamp: Date.now() };

    // Store user message immediately and update the CURRENT branch/tab
    setPlans(prevPlans => {
      const updated = [...prevPlans];
      const current = updated[activePlanIndex];
      updated[activePlanIndex] = {
        ...current,
        history: [...(current.history || []), userMsg],
        perspective: current.perspective.includes('(修改中...)') ? current.perspective : `${current.perspective} (修改中...)`
      };
      return updated;
    });

    setLoading(true);
    setProgress(0);
    setProgressText('正在根据您的需求修改方案...');

    try {
      const currentPlan = plans[activePlanIndex];
      // Construct payload
      const latestNodeDetails = nodeDetails && Object.keys(nodeDetails).length > 0 ? nodeDetails : currentPlan.node_details;

      const payload = {
        session_id: sessionId.trim() || undefined,
        plan_id: currentPlan.plan_id,
        current_plan: {
          module_details: latestNodeDetails,
          overall_description: currentPlan.description,
        },
        instruction: modificationInput,
        original_query: description,
        referenced_modules: referencedModules.length > 0 ? referencedModules : undefined,
        // Scope deictic edits ("this module") to the node the user has selected on the canvas.
        focused_modules: selectedNodeInfo?.name ? [selectedNodeInfo.name] : undefined,
      };

      // Clear input and quoted modules immediately
      setModificationInput('');
      setQuotedModules([]);

      const response = await fetch(`${API_BASE}/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.body) throw new Error('ReadableStream not yet supported.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let editBuf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        editBuf += decoder.decode(value, { stream: true });
        const lines = editBuf.split('\n');
        editBuf = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === 'progress') {
              setProgress(data.percent || 0);
              setProgressText(data.message || '修改中...');
            } else if (data.type === 'assistant') {
              const modeLabelMap = {
                plan_explain: '方案解读',
                evidence_answer: '证据答疑',
                direct_answer: '答疑',
                plan_update_chat: '方案更新说明',
              };
              const modeLabel = modeLabelMap[data.mode] || '答疑';
              const baseText = data.message || '';
              const citations = Array.isArray(data.citations) ? data.citations : [];
              const citationText = citations.length
                ? `\n\n引用:\n${citations.map((c, i) => {
                    const title = c.title || c.name || c.source || `Reference ${i + 1}`;
                    const snippet = c.snippet ? ` - ${c.snippet}` : '';
                    return `[${i + 1}] ${title}${snippet}`;
                  }).join('\n')}`
                : '';
              const finalText = `${modeLabel}${baseText ? `: ${baseText}` : ''}${citationText}`;
              const meta = {
                addressedExplorations: Array.isArray(data.addressed_explorations) ? data.addressed_explorations : [],
                addressedQuestions: Array.isArray(data.addressed_questions) ? data.addressed_questions : [],
                intentAnalysis: data.intent_analysis || null,
                unhandledIntents: Array.isArray(data.unhandled_intents) ? data.unhandled_intents : [],
                llmCritique: data.llm_critique || null,
              };
              setPlans(prevPlans => {
                const updated = [...prevPlans];
                const targetIndex = activePlanIndex;
                const current = updated[targetIndex];
                const updatedPerspective = current.perspective.replace(' (修改中...)', '');
                updated[targetIndex] = {
                  ...current,
                  history: [...(current.history || []), { role: 'assistant', content: finalText, meta, timestamp: Date.now() }],
                  perspective: updatedPerspective
                };
                return updated;
              });
            } else if (data.type === 'plan') {
              // Update the current plan with the new one
              const newPlanData = data.plan_data;

              setPlans(prevPlans => {
                const updatedPlans = [...prevPlans];
                const targetIndex = activePlanIndex; // Update the active branch
                const currentHistory = updatedPlans[targetIndex].history || [];

                // Add system success message
                const sysMsg = { role: 'system', content: `修改完成: ${data.plan_data.description ? data.plan_data.description : '架构已更新'}`, timestamp: Date.now() };

                // Remove "(修改中...)" from perspective
                const updatedPerspective = updatedPlans[targetIndex].perspective.replace(' (修改中...)', '');

                updatedPlans[targetIndex] = {
                  ...newPlanData,
                  history: [...currentHistory, sysMsg],
                  perspective: updatedPerspective // Cleaned
                };
                return updatedPlans;
              });

              // Update current view immediately
              if (newPlanData.node_details) {
                setNodeDetails(newPlanData.node_details);
              } else if (newPlanData.module_details) {
                setNodeDetails(newPlanData.module_details);
              }

            } else if (data.type === 'critique') {
              // Deferred critique arrives AFTER the plan/answer is already on screen.
              // Attach it to the most recent assistant message (fall back to the last
              // message, e.g. the "修改完成" system note) so the 质量/反思 panel fills in.
              const critique = data.llm_critique || null;
              if (critique) {
                setPlans(prevPlans => {
                  const updated = [...prevPlans];
                  const targetIndex = activePlanIndex;
                  const current = updated[targetIndex];
                  if (!current) return prevPlans;
                  const history = [...(current.history || [])];
                  let attachIdx = -1;
                  for (let i = history.length - 1; i >= 0; i--) {
                    if (history[i].role === 'assistant') { attachIdx = i; break; }
                  }
                  if (attachIdx === -1) attachIdx = history.length - 1;
                  if (attachIdx >= 0) {
                    const target = history[attachIdx];
                    history[attachIdx] = { ...target, meta: { ...(target.meta || {}), llmCritique: critique } };
                    updated[targetIndex] = { ...current, history };
                  }
                  return updated;
                });
              }
            } else if (data.type === 'error') {
              console.error('Edit error:', data.message);
              // Add error message to history
              setPlans(prevPlans => {
                const updated = [...prevPlans];
                const targetIndex = activePlanIndex; // Update active branch
                const current = updated[targetIndex];
                const updatedPerspective = current.perspective.replace(' (修改中...)', '');

                updated[targetIndex] = {
                  ...current,
                  history: [...(current.history || []), { role: 'system', content: `Error: ${data.message}`, isError: true, timestamp: Date.now() }],
                  perspective: updatedPerspective
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
        const targetIndex = activePlanIndex; // Update active branch
        const current = updated[targetIndex];
        const updatedPerspective = current.perspective.replace(' (修改中...)', '');

        updated[targetIndex] = {
          ...current,
          history: [...(current.history || []), { role: 'system', content: `Request Error: ${error.message}`, isError: true, timestamp: Date.now() }],
          perspective: updatedPerspective
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
    const currentDetails = detailsRef.current || {};

    let foundModule = null;
    if (Array.isArray(currentDetails)) {
      foundModule = currentDetails.find(mod => mod.name === nodeName || mod.id === nodeName);
    } else {
      foundModule = currentDetails[nodeName];
    }

    if (foundModule) {
      setSelectedNodeInfo(foundModule);
      return;
    }

    // Try finding the node as a sub_node in any module
    const entries = Array.isArray(currentDetails) ? currentDetails.map(m => [m.name, m]) : Object.entries(currentDetails);
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
  }, []);

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

  const closeNodeDetails = useCallback(() => {
    setSelectedNodeInfo(null);
    setIsNodeDetailEditing(false);
  }, []);

  useEffect(() => {
    if (!selectedNodeInfo) {
      setNodeDetailDraft({ function: '', technical_methods: '' });
      setIsNodeDetailEditing(false);
      return;
    }
    setNodeDetailDraft({
      function: selectedNodeInfo.function || '',
      technical_methods: selectedNodeInfo.technical_methods || ''
    });
  }, [selectedNodeInfo]);

  const handleNodeDetailsChange = useCallback((newDetails) => {
    // Merge: preserve sub_nodes from current nodeDetails if newDetails lost them (e.g. collapsed state)
    let merged = newDetails;
    try {
      const current = detailsRef.current;
      if (current && newDetails) {
        const isArr = Array.isArray(newDetails);
        const entries = isArr ? newDetails.map(m => [m.id || m.name, m]) : Object.entries(newDetails);
        const currentEntries = Array.isArray(current) ? current.map(m => [m.id || m.name, m]) : Object.entries(current);
        const currentMap = Object.fromEntries(currentEntries);
        const mergedEntries = entries.map(([key, mod]) => {
          const curMod = currentMap[key];
          if (curMod && curMod.sub_nodes && curMod.sub_nodes.length > 0 && (!mod.sub_nodes || mod.sub_nodes.length === 0)) {
            return [key, { ...mod, sub_nodes: curMod.sub_nodes }];
          }
          return [key, mod];
        });
        merged = isArr ? mergedEntries.map(([, m]) => m) : Object.fromEntries(mergedEntries);
      }
    } catch (e) { void e; }

    let sig = '';
    try {
      sig = JSON.stringify(merged);
    } catch (e) { void e; }
    if (sig && sig === nodeDetailsSigRef.current) return;
    nodeDetailsSigRef.current = sig;
    setNodeDetails(merged);
    setPlans(prevPlans => {
      const updated = [...prevPlans];
      if (updated[activePlanIndex]) {
        updated[activePlanIndex] = {
          ...updated[activePlanIndex],
          node_details: merged
        };
      }
      return updated;
    });
  }, [activePlanIndex]);

  const updateSelectedNodeDetails = useCallback((detailsSource, selectedInfo, draft) => {
    if (!detailsSource || !selectedInfo) return detailsSource;
    let cloned;
    try {
      cloned = JSON.parse(JSON.stringify(detailsSource));
    } catch (e) {
      void e;
      return detailsSource;
    }

    const normalize = (value) => String(value || '').trim();
    const draftFunction = normalize(draft.function);
    const draftTechnicalMethods = normalize(draft.technical_methods);
    const selectedName = normalize(selectedInfo.name);
    const selectedId = normalize(selectedInfo.id);
    const parentName = normalize(selectedInfo.parentModule);

    const applyDraft = (target) => {
      if (!target) return false;
      target.function = draftFunction;
      target.technical_methods = draftTechnicalMethods;
      return true;
    };

    if (Array.isArray(cloned)) {
      if (selectedInfo.isSubNode) {
        const parent = cloned.find((module) =>
          normalize(module?.name) === parentName || normalize(module?.id) === parentName
        );
        if (!parent || !Array.isArray(parent.sub_nodes)) return detailsSource;
        const subNode = parent.sub_nodes.find((sn) =>
          normalize(sn?.name) === selectedName || normalize(sn?.id) === selectedId
        );
        if (!applyDraft(subNode)) return detailsSource;
        return cloned;
      }
      const module = cloned.find((item) =>
        normalize(item?.name) === selectedName ||
        normalize(item?.id) === selectedId
      );
      if (!applyDraft(module)) return detailsSource;
      return cloned;
    }

    const moduleEntries = Object.entries(cloned);

    if (selectedInfo.isSubNode) {
      const parentEntry = moduleEntries.find(([key, module]) =>
        normalize(key) === parentName ||
        normalize(module?.name) === parentName ||
        normalize(module?.id) === parentName
      );
      if (!parentEntry) return detailsSource;
      const parent = parentEntry[1];
      if (!Array.isArray(parent.sub_nodes)) return detailsSource;
      const subNode = parent.sub_nodes.find((sn) =>
        normalize(sn?.name) === selectedName || normalize(sn?.id) === selectedId
      );
      if (!applyDraft(subNode)) return detailsSource;
      return cloned;
    }

    const moduleEntry = moduleEntries.find(([key, module]) =>
      normalize(key) === selectedName ||
      normalize(module?.name) === selectedName ||
      normalize(module?.id) === selectedId
    );
    if (!moduleEntry) return detailsSource;
    const module = moduleEntry[1];
    if (!applyDraft(module)) return detailsSource;
    return cloned;
  }, []);

  const handleExport = async () => {
    if (!resultFlowRef.current) return;
    if (plans.length === 0 || !plans[activePlanIndex]) return;

    setLoading(true);
    setProgress(0);
    setProgressText('Generating Document...');
    try {
      // Auto-save snapshot before exporting
      await saveSnapshotToSessionTree(undefined, '已保存当前架构（导出前自动保存）', '');

      // 1. Capture React Flow component to PNG
      const dataUrl = await resultFlowRef.current.exportImage();

      const currentPlan = plans[activePlanIndex];

      // 2. Fetch the /export endpoint
      const response = await fetch(`${API_BASE}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          plan: nodeDetails,
          overall_description: currentPlan.description || '',
          plan_id: currentPlan.plan_id || ''
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

  const saveSnapshotToSessionTree = useCallback(async (detailsOverride, messageContent, successToast) => {
    if (plans.length === 0 || !plans[activePlanIndex]) return false;
    try {
      const currentPlan = plans[activePlanIndex];
      const latestNodeDetails = detailsOverride ?? (nodeDetails && Object.keys(nodeDetails).length > 0 ? nodeDetails : currentPlan.node_details);
      const payload = {
        session_id: sessionId.trim() || undefined,
        plan_id: currentPlan.plan_id,
        current_plan: {
          module_details: latestNodeDetails,
          overall_description: currentPlan.description
        },
        original_query: description
      };
      const res = await fetch(`${API_BASE}/sessions/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '保存失败');
      setPlans(prev => {
        const u = [...prev];
        const idx = activePlanIndex;
        if (u[idx]) {
          const msg = { role: 'system', content: messageContent || '已保存当前架构', timestamp: Date.now() };
          u[idx] = { ...u[idx], plan_id: data.new_plan_id || u[idx].plan_id, history: [...(u[idx].history || []), msg] };
        }
        return u;
      });
      if (successToast) {
        alert(successToast);
      }
      return true;
    } catch (e) {
      alert(`保存失败: ${e.message}`);
      return false;
    }
  }, [plans, activePlanIndex, nodeDetails, sessionId, description]);

  const handleSaveNodeDetailDraft = useCallback(async () => {
    if (!selectedNodeInfo) return;
    const updatedDetails = updateSelectedNodeDetails(resolvedNodeDetails, selectedNodeInfo, nodeDetailDraft);
    handleNodeDetailsChange(updatedDetails);
    setSelectedNodeInfo(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        function: (nodeDetailDraft.function || '').trim(),
        technical_methods: (nodeDetailDraft.technical_methods || '').trim()
      };
    });
    await saveSnapshotToSessionTree(
      updatedDetails,
      `已保存节点「${selectedNodeInfo.name}」详情`,
      `已保存节点「${selectedNodeInfo.name}」并写入方案快照`
    );
    setIsNodeDetailEditing(false);
  }, [selectedNodeInfo, nodeDetailDraft, resolvedNodeDetails, updateSelectedNodeDetails, handleNodeDetailsChange, saveSnapshotToSessionTree]);

  const handleSaveSnapshot = async () => {
    await saveSnapshotToSessionTree(undefined, '已保存当前架构', '已保存当前架构');
  };

  const handleRevertLast = async () => {
    try {
      if (plans.length === 0 || !plans[activePlanIndex]) throw new Error('无可用方案');
      const currentPlan = plans[activePlanIndex];
      const params = new URLSearchParams();
      params.set('session_id', sessionId.trim() || 'anonymous_session');
      if (currentPlan.plan_id) params.set('plan_id', currentPlan.plan_id);
      const res = await fetch(`${API_BASE}/sessions/revert?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '无法回退');
      const p = data.plan;
      const cp = p?.arch_json || {};
      const details = cp.module_details || cp;
      if (!details) throw new Error('快照缺少模块数据');
      setNodeDetails(details);
      setPlans(prev => {
        const u = [...prev];
        const idx = activePlanIndex;
        if (u[idx]) {
          u[idx] = {
            ...u[idx],
            plan_id: p.plan_id || u[idx].plan_id,
            node_details: details,
            description: p.description || cp.overall_description || u[idx].description,
            perspective: p.perspective || u[idx].perspective,
            history: [...(u[idx].history || []), { role: 'system', content: '已回退到上一版本', timestamp: Date.now() }]
          };
        }
        return u;
      });
      alert('已回退到上一版本');
    } catch (e) {
      alert(`恢复失败: ${e.message}`);
    }
  };

  const activePlan = plans[activePlanIndex] || null;

  // Capture thumbnail for a plan after its diagram renders
  const captureThumbnail = useCallback(async (index) => {
    if (!resultFlowRef.current) return;
    try {
      const dataUrl = await resultFlowRef.current.exportImage();
      setPlanThumbnails(prev => ({ ...prev, [index]: dataUrl }));
    } catch (e) { void e; }
  }, []);

  // Capture thumbnail when active plan changes in result mode (after a short delay for render)
  useEffect(() => {
    if (plans.length === 0 || viewMode !== 'result') return;
    const timer = setTimeout(() => captureThumbnail(activePlanIndex), 800);
    return () => clearTimeout(timer);
  }, [activePlanIndex, plans.length, viewMode, captureThumbnail]);

  const thumbnailCycleRef = useRef(false);

  const otherPlans = plans
    .map((plan, index) => ({ plan, index }))
    .filter((item) => item.index !== activePlanIndex);

  return (
    <div className="app-container">
      <div style={{ position: 'fixed', top: 8, right: 16, zIndex: 9999 }}>
        <a
          href="#/chat"
          style={{
            fontSize: '0.82rem', fontWeight: 500, color: '#217cf3',
            background: '#e8f0fe', padding: '4px 12px', borderRadius: 6,
            textDecoration: 'none', border: '1px solid #c5d8fb',
          }}
        >
          💬 LLM Chat
        </a>
      </div>

      <main className="main-content">
        <div className="generator-layout" style={viewMode === 'result' ? { gridTemplateColumns: '320px 1fr' } : {}}>

          {/* --- 左侧边栏 (输入 + 工具 + 代码) --- */}
          <div className="sidebar-left">

            {/* 1. 需求 / 对话 面板 */}
            <div className="left-card" style={(viewMode === 'result') ? { flex: '1 1 auto', display: 'flex', flexDirection: 'column' } : { flex: '0 0 auto' }}>
              <div style={{ marginBottom: '10px' }}>
                <input
                  className="styled-input"
                  placeholder="被试 ID"
                  value={sessionId}
                  onChange={(e) => setSessionId(e.target.value)}
                  disabled={loading || plans.length > 0}
                  style={{ marginBottom: '5px', height: '32px' }}
                />
              </div>
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
                      {msg.role === 'assistant' && <strong style={{ display: 'block', fontSize: '0.8rem', color: '#666', marginBottom: '2px' }}>Assistant</strong>}
                      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                      {msg.meta && (msg.meta.addressedExplorations?.length > 0 || msg.meta.addressedQuestions?.length > 0 || msg.meta.unhandledIntents?.length > 0 || msg.meta.llmCritique) && (
                        <details style={{ marginTop: '6px', fontSize: '0.78rem' }}>
                          <summary style={{ cursor: 'pointer', color: '#666' }}>
                            意图分析与反思
                            {msg.meta.llmCritique?.quality_score != null && (
                              <span style={{ marginLeft: '6px', padding: '1px 6px', borderRadius: '8px', backgroundColor: msg.meta.llmCritique.quality_score >= 4 ? '#e8f5e9' : (msg.meta.llmCritique.quality_score >= 3 ? '#fff8e1' : '#ffebee'), color: '#333' }}>
                                质量 {msg.meta.llmCritique.quality_score}/5
                              </span>
                            )}
                            {msg.meta.unhandledIntents?.length > 0 && (
                              <span style={{ marginLeft: '6px', padding: '1px 6px', borderRadius: '8px', backgroundColor: '#ffebee', color: '#c62828' }}>
                                未处理 {msg.meta.unhandledIntents.length}
                              </span>
                            )}
                          </summary>
                          <div style={{ marginTop: '6px', paddingLeft: '6px', borderLeft: '2px solid #e0e0e0' }}>
                            {msg.meta.addressedQuestions?.length > 0 && (
                              <div style={{ marginBottom: '6px' }}>
                                <div style={{ fontWeight: 600, color: '#1565c0', marginBottom: '2px' }}>问答</div>
                                {msg.meta.addressedQuestions.map((q, i) => (
                                  <div key={i} style={{ marginBottom: '4px' }}>
                                    <div style={{ color: '#666' }}>Q: {q.question}</div>
                                    <div style={{ color: '#333' }}>A: {q.answer}</div>
                                  </div>
                                ))}
                              </div>
                            )}
                            {msg.meta.addressedExplorations?.length > 0 && (
                              <div style={{ marginBottom: '6px' }}>
                                <div style={{ fontWeight: 600, color: '#6a1b9a', marginBottom: '2px' }}>探索建议</div>
                                {msg.meta.addressedExplorations.map((ex, i) => (
                                  <div key={i} style={{ marginBottom: '4px' }}>
                                    <div style={{ color: '#666' }}>· {ex.topic}{ex.applied_to_plan ? '（已应用）' : '（仅参考）'}</div>
                                    <div style={{ color: '#333' }}>{ex.response}</div>
                                  </div>
                                ))}
                              </div>
                            )}
                            {msg.meta.unhandledIntents?.length > 0 && (
                              <div style={{ marginBottom: '6px' }}>
                                <div style={{ fontWeight: 600, color: '#c62828', marginBottom: '2px' }}>未处理项</div>
                                {msg.meta.unhandledIntents.map((u, i) => (
                                  <div key={i} style={{ color: '#666', marginBottom: '2px' }}>
                                    · [{u.kind}] {u.detail}
                                  </div>
                                ))}
                              </div>
                            )}
                            {msg.meta.llmCritique?.critique && (
                              <div style={{ marginBottom: '6px' }}>
                                <div style={{ fontWeight: 600, color: '#4527a0', marginBottom: '2px' }}>Critic 点评</div>
                                <div style={{ color: '#333' }}>{msg.meta.llmCritique.critique}</div>
                                {msg.meta.llmCritique.structural_issues?.length > 0 && (
                                  <div style={{ color: '#c62828', marginTop: '2px' }}>
                                    结构问题: {msg.meta.llmCritique.structural_issues.join('；')}
                                  </div>
                                )}
                                {msg.meta.llmCritique.hallucinations?.length > 0 && (
                                  <div style={{ color: '#ef6c00', marginTop: '2px' }}>
                                    可能虚构: {msg.meta.llmCritique.hallucinations.join('；')}
                                  </div>
                                )}
                                {msg.meta.llmCritique.suggested_fixes?.length > 0 && (
                                  <div style={{ marginTop: '4px' }}>
                                    <div style={{ color: '#666' }}>建议修复:</div>
                                    {msg.meta.llmCritique.suggested_fixes.map((fix, i) => (
                                      <div key={i} style={{ color: '#333', marginLeft: '8px' }}>· {fix}</div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </details>
                      )}
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

              {viewMode === 'result' && quotedModules.length > 0 && (
                <div style={{ flexShrink: 0, marginBottom: '6px', padding: '6px 8px', backgroundColor: '#eef6ff', border: '1px solid #c5d8fb', borderRadius: '6px' }}>
                  <div style={{ fontSize: '10px', color: '#888', fontWeight: 600, marginBottom: '4px' }}>已引用模块（将附加到下一条消息）</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                    {quotedModules.map((q, i) => (
                      <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '3px', backgroundColor: '#fff', border: '1px solid #217cf3', borderRadius: '4px', padding: '2px 6px', fontSize: '11px', color: '#1a56db' }}>
                        <span style={{ fontSize: '10px', color: '#888' }}>{q.planName}·</span>
                        {q.moduleName}
                        <span
                          style={{ cursor: 'pointer', color: '#aaa', marginLeft: '1px', fontSize: '12px', lineHeight: 1 }}
                          onClick={() => setQuotedModules(prev => prev.filter((_, idx) => idx !== i))}
                        >×</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <textarea
                className="styled-input"
                value={viewMode === 'result' ? modificationInput : description}
                placeholder={viewMode === 'result' ? '输入修改指令...' : '描述您的系统需求...'}
                onChange={(e) => {
                  if (viewMode !== 'result') setDescription(e.target.value);
                  else setModificationInput(e.target.value);
                }}
                disabled={loading}
                style={viewMode === 'result' ? { flexShrink: 0, resize: 'vertical', minHeight: '80px' } : {}}
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
                {loading ? `${progressText} ${progress}%` : '开始生成'}
              </button>
            </div>

            <div className="left-card" style={{ flex: '0 0 auto', display: viewMode === 'result' ? 'none' : 'flex', minHeight: 0 }}>
              <div className="card-title" style={{ fontSize: '0.9rem', color: '#666' }}>快速测试渲染 JSON (跳过后端)</div>
              <textarea
                className="styled-input"
                rows={4}
                value={testJsonStr}
                onChange={(e) => setTestJsonStr(e.target.value)}
                placeholder={JSON.stringify(plan_placeholder)}
                spellCheck="false"
              />
              <button
                className="btn-primary"
                style={{ marginTop: '10px', backgroundColor: '#f39c12', border: 'none' }}
                onClick={() => {
                  const str = testJsonStr.trim() ? testJsonStr : JSON.stringify([plan_placeholder, plan_placeholder2]);
                  if (!testJsonStr.trim()) setTestJsonStr(str);
                  handleTestJson(str);
                }}
              >
                直接渲染测试
              </button>
            </div>
          </div>

          {/* --- 中间画布 (节点) --- */}
          {viewMode === 'edit' && (
            <div className="panel canvas-panel" ref={reactFlowWrapper} style={{ height: '100%', width: '100%', overflow: 'hidden' }}>
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
                onPaneClick={closeNodeDetails}
                onNodeDragStop={onNodeDragStop}
                onEdgeDoubleClick={onEdgeDoubleClick}
                fitView
                attributionPosition="bottom-right"
                deleteKeyCode={['Backspace', 'Delete']}
                nodesDraggable={!loading}
                nodesConnectable={!loading}
                elementsSelectable={!loading}
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
          )}

          {/* --- 右侧边栏 (视角卡片 / 全屏方案) --- */}
            <div className="panel sidebar-right" style={{ display: 'flex', flexDirection: 'column', flex: viewMode === 'result' ? 1 : '0 0 60%', position: 'relative', minHeight: 0 }}>

            {/* === 编辑模式：视角/方案卡片横排 === */}
            {viewMode === 'edit' && (divergentPerspectives.length > 0 || plans.length > 0) && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {/* Header */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 16px', borderBottom: '1px solid #e0e0e0', backgroundColor: '#f5f5f5', flexShrink: 0 }}>
                  <div style={{ fontWeight: 600, color: '#333', fontSize: '14px' }}>
                    {loading ? `${progressText} ${progress}%` : `方案生成完成 — 点击卡片进入全屏查看`}
                  </div>
                  {!loading && plans.length > 0 && (
                    <div style={{ fontSize: '12px', color: '#888' }}>{plans.length} / {divergentPerspectives.length || plans.length} 个方案已完成</div>
                  )}
                </div>
                {/* Cards — 竖排 */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {(divergentPerspectives.length > 0 ? divergentPerspectives : plans.map(p => ({ perspective: p.perspective, rationale: p.description, additional_anchors: [] }))).map((persp, idx) => {
                    const completedPlan = plans[idx];
                    const prog = planProgressMap[idx] || { percent: 0, done: false };
                    const doneCount = Object.values(planProgressMap).filter(v => v && v.done).length;
                    const isWaiting = !prog.done && idx > doneCount;
                    const thumb = planThumbnails[idx];

                    return (
                      <div
                        key={idx}
                        onClick={completedPlan ? () => {
                          nodeDetailsSigRef.current = ''; // force re-sync
                          const details = completedPlan.node_details ?? completedPlan.module_details;
                          if (details) setNodeDetails(details);
                          setActivePlanIndex(idx);
                          setViewMode('result');
                        } : undefined}
                        style={{
                          border: `1px solid ${completedPlan ? '#c5d8fb' : '#e0e7f0'}`,
                          borderRadius: '10px',
                          backgroundColor: '#fff',
                          display: 'flex',
                          flexDirection: 'column',
                          overflow: 'hidden',
                          flexShrink: 0,
                          cursor: completedPlan ? 'pointer' : 'default',
                          boxShadow: completedPlan ? '0 2px 12px rgba(33,124,243,0.10)' : '0 1px 4px rgba(0,0,0,0.05)',
                          transition: 'box-shadow 0.15s, border-color 0.15s, transform 0.15s',
                        }}
                        onMouseEnter={completedPlan ? e => { e.currentTarget.style.boxShadow = '0 4px 18px rgba(33,124,243,0.18)'; e.currentTarget.style.borderColor = '#217cf3'; } : undefined}
                        onMouseLeave={completedPlan ? e => { e.currentTarget.style.boxShadow = '0 2px 12px rgba(33,124,243,0.10)'; e.currentTarget.style.borderColor = '#c5d8fb'; } : undefined}
                      >
                        {/* Card label */}
                        <div style={{ padding: '14px 18px 10px', borderBottom: '1px solid #f0f0f0' }}>
                          <span style={{
                            fontSize: '13px', fontWeight: 700, padding: '3px 10px', borderRadius: '12px',
                            backgroundColor: completedPlan ? '#e8f0fe' : '#f5f5f5',
                            color: completedPlan ? '#1a56db' : '#999',
                            marginRight: '8px',
                          }}>方案{['一','二','三','四'][idx] || idx+1}</span>
                          <span style={{ fontWeight: 700, fontSize: '18px', color: '#1a2436' }}>
                            {persp ? persp.perspective : `方案 ${idx + 1}`}
                          </span>
                        </div>
                        {/* Rationale */}
                        {persp && persp.rationale && (
                          <div style={{ padding: '10px 18px 8px', fontSize: '14px', color: '#505a6b', lineHeight: 1.7 }}>
                            <div style={{ fontWeight: 600, color: '#888', fontSize: '13px', marginBottom: '5px' }}>设计视角</div>
                            <div>{persp.rationale}</div>
                          </div>
                        )}
                        {/* Anchor pills */}
                        {persp && persp.additional_anchors && persp.additional_anchors.length > 0 && (
                          <div style={{ padding: '0 18px 10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {persp.additional_anchors.map((a, ai) => (
                              <span key={ai} style={{
                                fontSize: '13px', padding: '3px 10px', borderRadius: '4px',
                                backgroundColor: '#f0f4ff', color: '#3b5fc0', border: '1px solid #c9d8f8'
                              }}>{a.user_term}</span>
                            ))}
                          </div>
                        )}
                        {/* Thumbnail */}
                        {thumb && (
                          <div style={{ padding: '0 18px 10px' }}>
                            <img src={thumb} alt="方案预览" style={{ width: '100%', maxHeight: '180px', objectFit: 'contain', borderRadius: '6px', border: '1px solid #e8edf5', display: 'block' }} />
                          </div>
                        )}
                        {/* Progress / CTA */}
                        <div style={{ marginTop: 'auto', padding: '10px 18px 12px', borderTop: '1px solid #f0f0f0', backgroundColor: '#fafbfc' }}>
                          {completedPlan ? (
                            <div style={{ textAlign: 'center', color: '#217cf3', fontWeight: 600, fontSize: '14px' }}>点击进入全屏方案 →</div>
                          ) : isWaiting ? (
                            <div style={{ textAlign: 'center', color: '#bbb', fontSize: '13px' }}>等待生成...</div>
                          ) : (
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                <span style={{ fontSize: '13px', color: '#666' }}>完整方案生成中</span>
                                <span style={{ fontSize: '13px', color: '#666' }}>{prog.percent}%</span>
                              </div>
                              <div style={{ height: '5px', backgroundColor: '#e8edf5', borderRadius: '3px', overflow: 'hidden' }}>
                                <div style={{ height: '100%', borderRadius: '3px', backgroundColor: '#217cf3', width: `${prog.percent}%`, transition: 'width 0.5s ease' }} />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* === 编辑模式：初始空白占位 === */}
            {viewMode === 'edit' && divergentPerspectives.length === 0 && plans.length === 0 && (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ccc', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '36px' }}>💡</div>
                <div style={{ fontSize: '14px', color: '#aaa' }}>此处将展示三个不同的设计视角</div>
              </div>
            )}

            {/* === 全屏结果模式（始终保留在DOM中，仅用display控制可见性，避免X6重建） === */}
            <div style={{ display: viewMode === 'result' ? 'flex' : 'none', flex: 1, flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
            {/* 工作表 Tabs */}
            <div style={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderBottom: '1px solid #ccc',
              backgroundColor: '#f5f5f5',
              flexShrink: 0,
              flexWrap: 'wrap',
              gap: '4px',
              padding: '0 4px',
            }}>
              {/* 当前方案名 */}
              <div style={{ display: 'flex', alignItems: 'center', minWidth: 0, flex: '1 1 200px', overflowX: 'hidden' }}>
                {activePlan ? (
                  <div style={{ padding: '10px 16px', fontWeight: 'bold', color: '#217cf3', borderBottom: '2px solid #217cf3', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 280 }}>
                    {activePlan.perspective || `方案 ${activePlanIndex + 1}`}
                  </div>
                ) : (
                  <div style={{ padding: '10px 16px', color: '#999' }}>暂无方案</div>
                )}
              </div>
              {/* 右侧按钮组 */}
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap', padding: '6px 8px' }}>
                {otherPlans.length > 0 && (
                  <button
                    onClick={() => setIsCompareOpen(prev => !prev)}
                    style={{
                      padding: '7px 16px', fontSize: '14px', fontWeight: 600,
                      backgroundColor: isCompareOpen ? '#e8f0fe' : '#fff',
                      color: isCompareOpen ? '#1a56db' : '#3e4a5a',
                      border: `1px solid ${isCompareOpen ? '#c5d8fb' : '#d5dbe7'}`,
                      borderRadius: '6px', cursor: 'pointer', whiteSpace: 'nowrap', flexShrink: 0,
                    }}
                  >对比视图 ◫</button>
                )}
                {plans.length > 0 && (<>
                  <button onClick={handleSaveSnapshot} disabled={loading} style={{ padding: '5px 12px', backgroundColor: '#fff', color: '#2d76e2', border: '1px solid #2d76e2', borderRadius: '4px', cursor: 'pointer', fontSize: '13px', fontWeight: '500', whiteSpace: 'nowrap', flexShrink: 0 }}>Save</button>
                  <button onClick={handleRevertLast} disabled={loading} style={{ padding: '5px 12px', backgroundColor: '#fff', color: '#2d76e2', border: '1px solid #2d76e2', borderRadius: '4px', cursor: 'pointer', fontSize: '13px', fontWeight: '500', whiteSpace: 'nowrap', flexShrink: 0 }}>Revert</button>
                  <button onClick={handleExport} disabled={loading} style={{ padding: '5px 12px', backgroundColor: '#28a745', color: '#fff', border: '1px solid #28a745', borderRadius: '4px', cursor: 'pointer', fontSize: '13px', fontWeight: '500', whiteSpace: 'nowrap', flexShrink: 0 }}>Export (.docx)</button>
                  <button onClick={() => setViewMode('edit')} style={{ padding: '5px 12px', backgroundColor: '#fff', color: '#217cf3', border: '1px solid #217cf3', borderRadius: '4px', cursor: 'pointer', fontSize: '13px', fontWeight: '500', whiteSpace: 'nowrap', flexShrink: 0 }}>← 返回方案选择</button>
                </>)}
              </div>
            </div>

            {/* 主内容：画布 + 右侧对比栏 */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>
              {/* 左侧：主画布 + 节点详情 */}
              <div style={{ display: 'flex', flex: 1, overflow: 'hidden', flexDirection: 'column', minWidth: 0 }}>
              {/* 渲染图部分 */}
              <div className="mermaid-wrapper" style={{ flex: 1, overflow: 'hidden', borderBottom: '1px solid #eee', position: 'relative', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <div style={{ flex: 1, minHeight: 0, width: '100%' }}>
                  <X6FlowViewer
                    ref={resultFlowRef}
                    nodeDetails={resolvedNodeDetails}
                    onNodeClick={handleMermaidNodeClick}
                    onPaneClick={closeNodeDetails}
                    onNodeDetailsChange={handleNodeDetailsChange}
                  />
                </div>
                {(Array.isArray(resolvedNodeDetails) ? resolvedNodeDetails.length === 0 : Object.keys(resolvedNodeDetails || {}).length === 0) && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
                    <div style={{ color: '#999', textAlign: 'center', marginTop: '50px' }}>
                      <h3>图表空闲</h3>
                      <p>请在左侧输入需求并生成。</p>
                    </div>
                  </div>
                )}
              </div>

              {/* 节点详情部分 — 下方滚动面板 */}
              <div style={{ flexShrink: 0, maxHeight: '280px', overflowY: 'auto', backgroundColor: '#fafafa', borderTop: '1px solid #e0e0e0', padding: '12px 20px' }}>
                {!selectedNodeInfo ? (
                  plans.length > 0 && plans[activePlanIndex] && plans[activePlanIndex].description ? (
                    <div style={{ fontSize: '0.85rem', color: '#555', lineHeight: 1.6 }}>
                      <strong style={{ color: '#333', display: 'block', marginBottom: '5px', fontSize: '0.88rem' }}>总体描述</strong>
                      <div style={{ whiteSpace: 'pre-wrap' }}>{plans[activePlanIndex].description}</div>
                    </div>
                  ) : (
                    <div style={{ color: '#bbb', fontSize: '0.85rem' }}>点击图表节点查看详情</div>
                  )
                ) : (
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                      <div>
                        <strong style={{ color: '#222', fontSize: '1.15rem' }}>{selectedNodeInfo.name}</strong>
                        {selectedNodeInfo.isSubNode && (
                          <span style={{ fontSize: '0.78rem', color: '#888', marginLeft: '8px' }}>所属: {selectedNodeInfo.parentModule}</span>
                        )}
                      </div>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        {!isNodeDetailEditing ? (
                          <button onClick={() => setIsNodeDetailEditing(true)} disabled={loading} style={{ padding: '3px 10px', background: '#fff', border: '1px solid #d0d7de', color: '#555', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}>编辑</button>
                        ) : (
                          <>
                            <button onClick={handleSaveNodeDetailDraft} disabled={loading} style={{ padding: '3px 10px', background: '#217cf3', border: '1px solid #217cf3', color: '#fff', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}>保存</button>
                            <button onClick={() => { setNodeDetailDraft({ function: selectedNodeInfo.function || '', technical_methods: selectedNodeInfo.technical_methods || '' }); setIsNodeDetailEditing(false); }} disabled={loading} style={{ padding: '3px 10px', background: '#fff', border: '1px solid #d0d7de', color: '#666', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}>取消</button>
                          </>
                        )}
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '10px' }}>
                      <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: '6px', padding: '8px 10px' }}>
                        <strong style={{ color: '#444', fontSize: '0.8rem', display: 'block', marginBottom: '4px' }}>功能</strong>
                        {isNodeDetailEditing ? (
                          <textarea className="styled-input" rows={3} value={nodeDetailDraft.function} onChange={(e) => setNodeDetailDraft((prev) => ({ ...prev, function: e.target.value }))} style={{ marginTop: 0, marginBottom: 0, resize: 'vertical', fontSize: '0.82rem', lineHeight: 1.4 }} />
                        ) : (
                          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.5, color: '#333', fontSize: '0.82rem' }}>{nodeDetailDraft.function || '暂无'}</div>
                        )}
                      </div>
                      <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: '6px', padding: '8px 10px' }}>
                        <strong style={{ color: '#444', fontSize: '0.8rem', display: 'block', marginBottom: '4px' }}>技术方法</strong>
                        {isNodeDetailEditing ? (
                          <textarea className="styled-input" rows={3} value={nodeDetailDraft.technical_methods} onChange={(e) => setNodeDetailDraft((prev) => ({ ...prev, technical_methods: e.target.value }))} style={{ marginTop: 0, marginBottom: 0, resize: 'vertical', fontSize: '0.82rem', lineHeight: 1.4 }} />
                        ) : (
                          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.5, color: '#333', fontSize: '0.82rem' }}>{nodeDetailDraft.technical_methods || '暂无'}</div>
                        )}
                      </div>
                      <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: '6px', padding: '8px 10px' }}>
                        <strong style={{ color: '#444', fontSize: '0.8rem', display: 'block', marginBottom: '4px' }}>输入/输出</strong>
                        <div style={{ fontSize: '0.82rem', color: '#333', lineHeight: 1.5 }}>
                          {(() => {
                            const ports = selectedNodeInfo.ports;
                            if (!ports) return <span style={{ color: '#bbb' }}>暂无</span>;
                            const incoming = ports.incoming || [];
                            const outgoing = ports.outgoing || [];
                            if (incoming.length === 0 && outgoing.length === 0) return <span style={{ color: '#bbb' }}>暂无</span>;
                            return (
                              <>
                                {incoming.length > 0 && (<div><span style={{ color: '#3b82f6' }}>输入: </span>{incoming.map((p, i) => <span key={i}>{p.edge_label || p.source_node || JSON.stringify(p)}{i < incoming.length - 1 ? '、' : ''}</span>)}</div>)}
                                {outgoing.length > 0 && (<div><span style={{ color: '#22c55e' }}>输出: </span>{outgoing.map((p, i) => <span key={i}>{p.edge_label || p.target_mod || p.target_node || JSON.stringify(p)}{i < outgoing.length - 1 ? '、' : ''}</span>)}</div>)}
                              </>
                            );
                          })()}
                        </div>
                      </div>
                    </div>
                    {(() => {
                      const usedRefs = plans[activePlanIndex]?.used_references;
                      const nodeName = selectedNodeInfo.name;
                      const fullEntries = usedRefs && nodeName && usedRefs[nodeName];
                      // selectedNodeInfo.references is the LLM's comma-separated string;
                      // selectedNodeInfo.reference is the singular {name, technical_details, source}
                      // attached by pipeline_runner.collect_references. Mixing them via `||`
                      // used to leak the object into the fallback <div>{refStr}</div> render
                      // and produce "[object Object]" — keep them strictly typed here.
                      const refStrRaw = typeof selectedNodeInfo.references === 'string' ? selectedNodeInfo.references : '';
                      const refObj = (selectedNodeInfo.reference && typeof selectedNodeInfo.reference === 'object') ? selectedNodeInfo.reference : null;
                      const refStr = refStrRaw && refStrRaw !== 'None' ? refStrRaw : '';
                      const hasAny = (fullEntries && fullEntries.length > 0) || refStr || refObj;
                      if (!hasAny) return null;
                      return (
                        <div style={{ background: '#f7f9fc', border: '1px solid #e8edf5', borderRadius: '6px', padding: '8px 10px' }}>
                          <strong style={{ color: '#444', fontSize: '0.8rem', display: 'block', marginBottom: '6px' }}>来源</strong>
                          {fullEntries && fullEntries.length > 0 ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                              {fullEntries.map((entry, i) => (
                                <div key={i} style={{ borderLeft: '3px solid #c5d8fb', paddingLeft: '8px' }}>
                                  <div style={{ fontWeight: 600, fontSize: '0.8rem', color: '#2a3550', marginBottom: '2px' }}>
                                    [{i + 1}] {entry.name || '未知来源'}
                                    {entry.source && <span style={{ fontWeight: 400, color: '#7a8a9e', marginLeft: '6px', fontSize: '0.75rem' }}>{entry.source}</span>}
                                  </div>
                                  {entry.text && (<div style={{ fontSize: '0.78rem', color: '#444', lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{entry.text}</div>)}
                                </div>
                              ))}
                            </div>
                          ) : refObj ? (
                            <div style={{ borderLeft: '3px solid #c5d8fb', paddingLeft: '8px' }}>
                              <div style={{ fontWeight: 600, fontSize: '0.8rem', color: '#2a3550', marginBottom: '2px' }}>
                                {refObj.name || '未知来源'}
                                {refObj.source && <span style={{ fontWeight: 400, color: '#7a8a9e', marginLeft: '6px', fontSize: '0.75rem' }}>{refObj.source}</span>}
                              </div>
                              {refObj.technical_details && (<div style={{ fontSize: '0.78rem', color: '#444', lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{refObj.technical_details}</div>)}
                            </div>
                          ) : (
                            <div style={{ fontSize: '0.8rem', color: '#555', lineHeight: 1.6 }}>{refStr}</div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>

              {/* 右侧对比栏 */}
              {isCompareOpen && otherPlans.length > 0 && (
                <div style={{
                  width: comparePanelWidth + 'px', flexShrink: 0,
                  borderLeft: '1px solid #e0e7f0', backgroundColor: '#f8faff',
                  display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative',
                }}>
                  {/* Drag handle */}
                  <div
                    style={{ position: 'absolute', top: 0, left: 0, width: '5px', height: '100%', cursor: 'col-resize', zIndex: 10 }}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      const startX = e.clientX;
                      const startW = comparePanelWidth;
                      const onMove = (mv) => {
                        const delta = startX - mv.clientX;
                        setComparePanelWidth(Math.min(500, Math.max(220, startW + delta)));
                      };
                      const onUp = () => {
                        window.removeEventListener('mousemove', onMove);
                        window.removeEventListener('mouseup', onUp);
                      };
                      window.addEventListener('mousemove', onMove);
                      window.addEventListener('mouseup', onUp);
                    }}
                  />
                  {/* 对比栏标题 */}
                  <div style={{ padding: '10px 14px 8px', fontWeight: 600, fontSize: '13px', color: '#555', borderBottom: '1px solid #e0e7f0', flexShrink: 0 }}>
                    其他视角
                  </div>
                  {/* 各方案区块 */}
                  <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
                    {otherPlans.map(({ plan, index }) => {
                      const thumb = planThumbnails[index];
                      const persp = divergentPerspectives[index];
                      const planDetails = plan.node_details ?? plan.module_details;
                      const moduleEntries = planDetails
                        ? (Array.isArray(planDetails)
                            ? planDetails.map(m => [m.id || m.name || '', m])
                            : Object.entries(planDetails))
                        : [];
                      // First sentence of overall_description / description
                      const descText = plan.description || '';
                      const firstSentence = descText.match(/^[^。！？.!?]+[。！？.!?]?/)?.[0] || descText.slice(0, 60);
                      return (
                        <div key={index} style={{ display: 'flex', flexDirection: 'column', borderBottom: '1px solid #e8edf5', flexShrink: 0 }}>
                          {/* Plan header — click to switch main canvas */}
                          <div
                            style={{ padding: '10px 14px 8px', backgroundColor: '#eef3ff', borderBottom: '1px solid #dde7f8', flexShrink: 0, cursor: 'pointer', display: 'flex', alignItems: 'flex-start', gap: '8px' }}
                            onClick={() => {
                              nodeDetailsSigRef.current = '';
                              const details = plan.node_details ?? plan.module_details;
                              if (details) setNodeDetails(details);
                              setActivePlanIndex(index);
                            }}
                            title="点击切换为主画布"
                          >
                            <span style={{ fontSize: '12px', fontWeight: 700, padding: '2px 8px', borderRadius: '10px', backgroundColor: '#e8f0fe', color: '#1a56db', flexShrink: 0, marginTop: '2px' }}>
                              方案{['一','二','三','四'][index] || index+1}
                            </span>
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div style={{ fontWeight: 700, fontSize: '15px', color: '#1a2436', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {plan.perspective || `方案 ${index + 1}`}
                              </div>
                              {firstSentence && (
                                <div style={{ fontSize: '12px', color: '#666', marginTop: '3px', lineHeight: 1.5 }}>{firstSentence}</div>
                              )}
                            </div>
                            <span style={{ fontSize: '11px', color: '#888', flexShrink: 0, marginTop: '2px' }}>切换 ↗</span>
                          </div>
                          {/* Thumbnail */}
                          {thumb && (
                            <div style={{ padding: '6px 10px', flexShrink: 0, borderBottom: '1px solid #f0f0f0' }}>
                              <img src={thumb} alt="方案预览" style={{ width: '100%', maxHeight: '120px', objectFit: 'contain', borderRadius: '5px', border: '1px solid #e8edf5', display: 'block' }} />
                            </div>
                          )}
                          {/* Module list with 引用 buttons */}
                          <div style={{ maxHeight: '240px', overflowY: 'auto', padding: '4px 0' }}>
                            <div style={{ padding: '6px 14px 4px', fontSize: '11px', color: '#aaa', fontWeight: 600, letterSpacing: '0.05em' }}>模块列表 · 点击引用</div>
                            {moduleEntries.length === 0 && (
                              <div style={{ padding: '8px 14px', color: '#bbb', fontSize: '11px' }}>暂无模块数据</div>
                            )}
                            {moduleEntries.map(([key, mod]) => {
                              const modName = mod.name || key;
                              const isQuoted = quotedModules.some(q => q.planIndex === index && q.moduleName === modName);
                              return (
                                <div
                                  key={key}
                                  style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                    padding: '7px 14px', gap: '8px',
                                    backgroundColor: isQuoted ? '#eef6ff' : 'transparent',
                                    borderLeft: isQuoted ? '3px solid #217cf3' : '3px solid transparent',
                                    borderBottom: '1px solid #f0f4ff',
                                  }}
                                >
                                  <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '13px', fontWeight: 600, color: '#2a3550', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{modName}</div>
                                    {/* {mod.brief_description && (
                                      <div style={{ fontSize: '11px', color: '#888', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: '2px' }}>{mod.brief_description}</div>
                                    )}                                   */}
                                  </div>
                                  <button
                                    onClick={() => {
                                      if (isQuoted) {
                                        setQuotedModules(prev => prev.filter(q => !(q.planIndex === index && q.moduleName === modName)));
                                      } else {
                                        setQuotedModules(prev => [...prev, {
                                          planIndex: index,
                                          planName: plan.perspective || `方案 ${index + 1}`,
                                          moduleName: modName,
                                          moduleData: mod,
                                        }]);
                                      }
                                    }}
                                    style={{
                                      flexShrink: 0, padding: '3px 10px', fontSize: '12px', borderRadius: '4px', cursor: 'pointer', fontWeight: 500,
                                      backgroundColor: isQuoted ? '#217cf3' : '#fff',
                                      color: isQuoted ? '#fff' : '#217cf3',
                                      border: '1px solid #217cf3',
                                    }}
                                  >{isQuoted ? '已引用' : '引用'}</button>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
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
