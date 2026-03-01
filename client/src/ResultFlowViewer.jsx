import React, { useEffect, useState } from 'react';
import ReactFlow, { Background, Controls, MarkerType, useNodesState, useEdgesState } from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';

const nodeWidth = 220;
const nodeHeight = 60;

const getLayoutedElements = (nodes, edges, direction = 'TB') => {
    const dagreGraph = new dagre.graphlib.Graph({ compound: true });
    dagreGraph.setDefaultEdgeLabel(() => ({}));

    // Set graph direction and margins
    dagreGraph.setGraph({ rankdir: direction, ranksep: 100, nodesep: 100, marginx: 50, marginy: 50 });

    // 1. Add nodes to dagre
    nodes.forEach((node) => {
        if (node.isGroup) {
            dagreGraph.setNode(node.id, { label: String(node.id), clusterLabelPos: 'top', paddingTop: 60, paddingBottom: 20, paddingLeft: 20, paddingRight: 20 });
        } else {
            dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
            if (node.parentNode) {
                dagreGraph.setParent(node.id, node.parentNode);
            }
        }
    });

    // 2. Add safe edges to dagre (prevent dagre crash with self or parent-child loops)
    const isSafeEdge = (source, target) => {
        if (source === target) return false;
        let p1 = dagreGraph.parent(source);
        while (p1) {
            if (p1 === target) return false;
            p1 = dagreGraph.parent(p1);
        }
        let p2 = dagreGraph.parent(target);
        while (p2) {
            if (p2 === source) return false;
            p2 = dagreGraph.parent(p2);
        }
        return true;
    };

    const isGroupNode = new Set(nodes.filter(n => n.isGroup).map(n => n.id));
    const groupToFirstChild = {};
    nodes.forEach(n => {
        if (n.parentNode && !groupToFirstChild[n.parentNode]) {
            groupToFirstChild[n.parentNode] = n.id;
        }
    });

    edges.forEach((edge) => {
        if (isSafeEdge(edge.source, edge.target)) {
            let dagreSource = edge.source;
            let dagreTarget = edge.target;

            // Dagre crashes if an edge points directly to/from a compound node that has children
            if (isGroupNode.has(dagreSource) && groupToFirstChild[dagreSource]) {
                dagreSource = groupToFirstChild[dagreSource];
            }
            if (isGroupNode.has(dagreTarget) && groupToFirstChild[dagreTarget]) {
                dagreTarget = groupToFirstChild[dagreTarget];
            }

            dagreGraph.setEdge(dagreSource, dagreTarget);
        }
    });

    // 3. Run layout
    dagre.layout(dagreGraph);

    // 4. Map back positions
    const layoutedNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        let newNode = { ...node };

        if (node.isGroup) {
            // For groups, dagre center-aligns the bounding box, but ReactFlow sets top-left.
            newNode.position = {
                x: nodeWithPosition.x - nodeWithPosition.width / 2,
                y: nodeWithPosition.y - nodeWithPosition.height / 2,
            };
            // We must explicitly set width and height for the parent to bound its children correctly
            newNode.style = {
                ...newNode.style,
                width: nodeWithPosition.width,
                height: nodeWithPosition.height,
            };
        } else {
            newNode.position = {
                x: nodeWithPosition.x - nodeWidth / 2,
                y: nodeWithPosition.y - nodeHeight / 2,
            };

            // If node has parent, coordinates in ReactFlow must be relative to parent.
            if (newNode.parentNode) {
                const parentPos = dagreGraph.node(newNode.parentNode);
                const parentX = parentPos.x - parentPos.width / 2;
                const parentY = parentPos.y - parentPos.height / 2;
                newNode.position.x -= parentX;
                newNode.position.y -= parentY;
                newNode.extent = 'parent';
            }
        }

        return newNode;
    });

    return { nodes: layoutedNodes, edges };
};

export default function ResultFlowViewer({ nodeDetails, onNodeClick }) {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    useEffect(() => {
        if (!nodeDetails || Object.keys(nodeDetails).length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        let nextNodes = [];
        let nextEdges = [];

        // Parse modules, subnodes and edges
        let normalizedDetails = {};
        if (Array.isArray(nodeDetails)) {
            nodeDetails.forEach(mod => {
                if (mod && mod.name) {
                    normalizedDetails[mod.name] = mod;
                }
            });
        } else {
            normalizedDetails = nodeDetails;
        }

        const modNames = Object.keys(normalizedDetails);

        modNames.forEach((modName) => {
            const mod = normalizedDetails[modName];
            const hasSubNodes = mod.sub_nodes && mod.sub_nodes.length > 0;

            if (hasSubNodes) {
                // Create container node
                nextNodes.push({
                    id: modName,
                    type: 'default', // Using default to support label
                    isGroup: true,
                    data: {
                        label: (
                            <div style={{ fontWeight: 'bold', fontSize: '14px', paddingTop: '5px' }}>
                                {mod.name || modName}
                            </div>
                        )
                    },
                    position: { x: 0, y: 0 },
                    style: {
                        zIndex: -1,
                        backgroundColor: 'rgba(33, 124, 243, 0.05)',
                        border: '2px dashed #217cf3',
                        borderRadius: '8px',
                        padding: 0,
                        paddingTop: '10px',
                        display: 'flex',
                        alignItems: 'flex-start',
                        justifyContent: 'center',
                        color: '#217cf3'
                    }
                });

                // Create inner nodes
                mod.sub_nodes.forEach((sn) => {
                    const snId = sn.name || sn.id;
                    nextNodes.push({
                        id: snId,
                        parentNode: modName,
                        data: { label: sn.name },
                        position: { x: 0, y: 0 },
                        style: {
                            width: nodeWidth,
                            height: nodeHeight,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: '#fff',
                            border: '1px solid #666',
                            borderRadius: '4px',
                            fontSize: '12px',
                            padding: '10px',
                            textAlign: 'center',
                            wordBreak: 'break-word'
                        }
                    });
                });
            } else {
                // Single node module
                nextNodes.push({
                    id: modName,
                    data: { label: mod.name || modName },
                    position: { x: 0, y: 0 },
                    style: {
                        width: nodeWidth,
                        height: nodeHeight,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: '#e6f7ff',
                        border: '1px solid #1890ff',
                        borderRadius: '4px',
                        fontSize: '13px',
                        padding: '10px',
                        fontWeight: 'bold',
                        textAlign: 'center',
                        wordBreak: 'break-word'
                    }
                });
            }

            // Process outgoing edges
            if (mod.ports && mod.ports.outgoing) {
                mod.ports.outgoing.forEach((outEdge, idx) => {
                    let sourceId = outEdge.originates_from_internal_node || outEdge.source_node || modName;
                    let targetId = outEdge.connects_to_internal_node || outEdge.target_node || outEdge.target_mod;

                    if (sourceId === 'null' || sourceId === 'None') sourceId = modName;
                    if (targetId === 'null' || targetId === 'None') targetId = outEdge.target_mod;

                    if (sourceId && targetId) {
                        nextEdges.push({
                            id: `edge-out-${modName}-${idx}`,
                            source: sourceId,
                            target: targetId,
                            label: outEdge.edge_label,
                            type: 'smoothstep',
                            markerEnd: { type: MarkerType.ArrowClosed, color: '#999' },
                            style: { stroke: '#999', strokeWidth: 1.5 },
                            labelStyle: { fill: '#666', fontSize: 10, fontWeight: 500 },
                            labelBgStyle: { fill: 'rgba(255,255,255,0.8)' }
                        });
                    }
                });
            }

            // Process internal flow
            if (mod.ports && mod.ports.internal_flow) {
                mod.ports.internal_flow.forEach((inEdge, idx) => {
                    let sourceId = inEdge.source_node;
                    let targetId = inEdge.target_node;

                    if (sourceId === 'null' || sourceId === 'None') sourceId = null;
                    if (targetId === 'null' || targetId === 'None') targetId = null;

                    if (sourceId && targetId) {
                        nextEdges.push({
                            id: `edge-in-${modName}-${idx}`,
                            source: sourceId,
                            target: targetId,
                            label: inEdge.edge_label,
                            type: 'smoothstep',
                            markerEnd: { type: MarkerType.ArrowClosed, color: '#aaa' },
                            style: { stroke: '#aaa', strokeWidth: 1.5, strokeDasharray: '4 4' },
                            labelStyle: { fill: '#888', fontSize: 9 },
                            labelBgStyle: { fill: 'rgba(255,255,255,0.7)' }
                        });
                    }
                });
            }
        });

        // Handle missing edge targets that might cause layout crashes
        const validNodeIds = new Set(nextNodes.map(n => n.id));
        nextEdges = nextEdges.filter(e => validNodeIds.has(e.source) && validNodeIds.has(e.target));

        try {
            const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(nextNodes, nextEdges);
            setNodes([...layoutedNodes]);
            setEdges([...layoutedEdges]);
        } catch (e) {
            console.error("Layout error:", e);
            alert("布局计算引擎遭遇严重失败: " + String(e));
            setNodes(nextNodes);
            setEdges(nextEdges);
        }

    }, [nodeDetails]);

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={(e, node) => {
                e.stopPropagation(); // Prevent App.jsx from catching this and closing the panel immediately
                // Determine original node name passed to onNodeClick
                onNodeClick(node.id);
            }}
            fitView
            attributionPosition="bottom-right"
            nodesDraggable={true}
        >
            <Background color="#ccc" gap={16} />
            <Controls />
        </ReactFlow>
    );
}
