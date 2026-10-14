import { useEffect, useRef, useImperativeHandle, forwardRef, useCallback, useState } from 'react';
import { Graph, Shape, Selection, Export } from '@antv/x6';

const NODE_W = 280;
const NODE_H = 88;
const GROUP_W = 360;

const ROLE_THEME = {
  input: { fill: 'rgba(56,132,255,0.1)', stroke: '#3b82f6', text: '#2456b3', light: '#e8f1ff' },
  processing: { fill: 'rgba(245,158,11,0.11)', stroke: '#f59e0b', text: '#9a6700', light: '#fff4df' },
  output: { fill: 'rgba(34,197,94,0.12)', stroke: '#22c55e', text: '#1f7a3c', light: '#e7f9ee' },
  data: { fill: 'rgba(156,163,175,0.1)', stroke: '#9ca3af', text: '#4b5563', light: '#f3f4f6' },
  unknown: { fill: 'rgba(245,158,11,0.11)', stroke: '#f59e0b', text: '#9a6700', light: '#fff4df' },
};

function normalizeRole(role) {
  return String(role || '').trim().toLowerCase();
}

function getRoleLane(role) {
  const raw = normalizeRole(role);
  if (!raw) return 'unknown';
  const category = raw.includes(':') ? raw.split(':')[0].trim() : raw;
  if (category.includes('data') || category.includes('storage') || category.includes('database') || category.includes('knowledge')) return 'data';
  if (category.includes('input') || category.includes('sensing')) return 'input';
  if (category.includes('output') || category.includes('actuation') || category.includes('presentation')) return 'output';
  if (category.includes('processing') || category.includes('core') || category.includes('inference') || category.includes('reasoning')) return 'processing';
  if (raw.includes('data') || raw.includes('storage') || raw.includes('database')) return 'data';
  if (raw.includes('input')) return 'input';
  if (raw.includes('output')) return 'output';
  if (raw.includes('processing')) return 'processing';
  return 'unknown';
}

function parseImportance(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.min(1, value));
  const t = String(value || '').trim().toLowerCase();
  if (!t) return 0.5;
  if (t === 'high' || t === 'critical' || t === 'p0') return 1;
  if (t === 'medium' || t === 'normal' || t === 'p1') return 0.66;
  if (t === 'low' || t === 'minor' || t === 'p2') return 0.33;
  const n = Number.parseFloat(t);
  if (!Number.isNaN(n)) return Math.max(0, Math.min(1, n));
  return 0.5;
}

function inferNodeIcons(node) {
  const text = [
    node.label || '',
    node.desc || '',
    node.role || '',
    node.functionText || '',
    node.technicalText || '',
  ].join(' ').toLowerCase();
  const role = normalizeRole(node.role);
  const icons = [];
  const hasDataSignal =
    role.includes('data') ||
    role.includes('storage') ||
    role.includes('database') ||
    role.includes('personalization_store') ||
    /\b(database|knowledge graph|vector db|vector store|data lake|repository)\b/.test(text) ||
    /\bdb\b/.test(text) ||
    /\bstore\b/.test(text);
  if (hasDataSignal) icons.push('🗄️');
  const hasLlmSignal =
    /\b(llm|gpt|large language model|ai model|foundation model|genai|ai inference|inference engine)\b/.test(text);
  if (hasLlmSignal) icons.push('🤖');
  if (role.includes('input') || role.includes('sensing')) icons.push('📥');
  if (role.includes('output') || role.includes('actuation') || role.includes('presentation')) icons.push('📤');
  return Array.from(new Set(icons)).slice(0, 2);
}

function makeTextMeasurer() {
  let canvas;
  let ctx;
  return (text, fontSize = 12, fontFamily = 'Arial, sans-serif', fontWeight = 'normal') => {
    if (!canvas) {
      canvas = document.createElement('canvas');
      ctx = canvas.getContext('2d');
    }
    ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
    return ctx.measureText(String(text)).width;
  };
}
const measureTextWidth = makeTextMeasurer();

function wrapTextToLines(text, maxWidth, fontSize = 12, fontWeight = 'normal') {
  const t = String(text || '').trim();
  if (!t) return [];
  const hasSpace = /\s/.test(t);
  const byChars = (str) => {
    const out = [];
    let cur = '';
    for (const ch of str) {
      const cand = cur + ch;
      if (measureTextWidth(cand, fontSize, 'Arial, sans-serif', fontWeight) <= maxWidth) {
        cur = cand;
      } else {
        if (cur) out.push(cur);
        cur = ch;
      }
    }
    if (cur) out.push(cur);
    return out;
  };
  if (!hasSpace) return byChars(t);
  const words = t.split(/\s+/);
  const lines = [];
  let cur = '';
  words.forEach(w => {
    const candidate = cur ? cur + ' ' + w : w;
    if (measureTextWidth(candidate, fontSize, 'Arial, sans-serif', fontWeight) <= maxWidth) {
      cur = candidate;
    } else {
      if (measureTextWidth(w, fontSize, 'Arial, sans-serif', fontWeight) > maxWidth) {
        if (cur) { lines.push(cur); cur = ''; }
        const broken = byChars(w);
        lines.push(...broken.slice(0, -1));
        cur = broken[broken.length - 1] || '';
      } else {
        if (cur) lines.push(cur);
        cur = w;
      }
    }
  });
  if (cur) lines.push(cur);
  return lines;
}

function computeTopologicalLevels(topNodes, edges, childToParent) {
  const topSet = new Set(topNodes.map(n => n.id));
  const incoming = {};
  const indegree = {};
  topNodes.forEach(n => {
    incoming[n.id] = [];
    indegree[n.id] = 0;
  });

  edges.forEach(e => {
    const srcTop = childToParent[e.source] || e.source;
    const tgtTop = childToParent[e.target] || e.target;
    if (!topSet.has(srcTop) || !topSet.has(tgtTop) || srcTop === tgtTop) return;
    incoming[tgtTop].push(srcTop);
    indegree[tgtTop] += 1;
  });

  const queue = topNodes.filter(n => indegree[n.id] === 0).map(n => n.id);
  const order = [];
  while (queue.length) {
    const cur = queue.shift();
    order.push(cur);
    edges.forEach(e => {
      const srcTop = childToParent[e.source] || e.source;
      const tgtTop = childToParent[e.target] || e.target;
      if (srcTop !== cur || !topSet.has(tgtTop) || srcTop === tgtTop) return;
      indegree[tgtTop] -= 1;
      if (indegree[tgtTop] === 0) queue.push(tgtTop);
    });
  }
  topNodes.forEach(n => {
    if (!order.includes(n.id)) order.push(n.id);
  });

  const levelMap = {};
  order.forEach(id => {
    const upstreamLevels = incoming[id].map(up => levelMap[up] ?? 0);
    levelMap[id] = upstreamLevels.length > 0 ? Math.max(...upstreamLevels) + 1 : 0;
  });

  return levelMap;
}

function runRoleLayout(nodes, edges, preservedPositions = {}) {
  const positions = {};
  const topNodes = nodes.filter(n => !n.parent);
  const childToParent = {};
  nodes.forEach(n => {
    if (n.parent) childToParent[n.id] = n.parent;
  });

  // Compute topological levels for all top-level nodes (drives X position)
  const levelMap = computeTopologicalLevels(topNodes, edges, childToParent);

  const COL_GAP = 120;   // horizontal gap between columns (increased for better flow visibility)
  const ROW_GAP = 60;   // vertical gap between nodes in same column (increased to reduce stacking feel)
  const DATA_ROW_GAP = 60; // gap below data row
  const childGap = 16;

  // Pre-calculate sub-node dynamic heights
  const subnodeDims = {};
  nodes.filter(n => n.parent).forEach(n => {
    const innerW = Math.max(10, NODE_W - 24);
    const rawDesc = String(n.desc || '').replace(/\s+/g, ' ').trim();
    const titleLines = wrapTextToLines(n.label, innerW, 14, 'bold');
    const descLines = wrapTextToLines(rawDesc, innerW, 12, 'normal');
    const lines = descLines.length > 0 ? [...titleLines, ...descLines] : [...titleLines];
    const dynH = Math.max(NODE_H, 18 + lines.length * 14);
    subnodeDims[n.id] = { w: NODE_W, h: dynH };
  });

  function computeNodeDims(n) {
    const w = n.isGroup ? GROUP_W : NODE_W;
    const innerW = Math.max(10, w - 24);
    const titleText = String(n.label || '').trim();
    const rawDesc = String(n.desc || '').replace(/\s+/g, ' ').trim();
    let h = NODE_H;
    let overlayTop = 0;
    if (n.isGroup) {
      const titleLines = wrapTextToLines(titleText, innerW, 20, 'bold');
      const titleH = Math.max(24, titleLines.length * 22);
      const descH = 0; // description hidden
      overlayTop = 10 + titleH;

      // If collapsed, use minimal height; if expanded, calculate based on children
      if (n.collapsed) {
        h = Math.max(NODE_H, overlayTop + 20);
      } else {
        const childTop = overlayTop + 16;
        const groupChildren = nodes.filter(c => c.parent === n.id).sort((a, b) => String(a.id).localeCompare(String(b.id)));
        let childrenTotalH = 0;
        groupChildren.forEach((c, idx) => {
          childrenTotalH += subnodeDims[c.id].h;
          if (idx < groupChildren.length - 1) childrenTotalH += childGap;
        });
        h = Math.max(160, childTop + childrenTotalH + 24);
      }
    } else {
      const titleLines = wrapTextToLines(titleText, innerW, 20, 'bold');
      const titleH = Math.max(24, titleLines.length * 22);
      const descH = 0; // description hidden
      h = Math.max(NODE_H, 20 + titleH + 10);
    }
    return { w, h, overlayTop };
  }

  const nodeDims = {};
  topNodes.forEach(n => { nodeDims[n.id] = computeNodeDims(n); });

  // Separate data nodes (shown as a horizontal row at top) from flow nodes
  const dataNodes = topNodes.filter(n => n.lane === 'data');
  const flowNodes = topNodes.filter(n => n.lane !== 'data');

  // Group flow nodes by their topological level (column index)
  const levelGroups = {};
  flowNodes.forEach(n => {
    const lvl = levelMap[n.id] ?? 0;
    if (!levelGroups[lvl]) levelGroups[lvl] = [];
    levelGroups[lvl].push(n);
  });

  // Sort nodes within each column: input first, then processing, then output/unknown
  const laneOrder = { input: 0, processing: 1, output: 2, unknown: 3 };
  Object.values(levelGroups).forEach(group => {
    group.sort((a, b) => {
      const lo = (laneOrder[a.lane] ?? 3) - (laneOrder[b.lane] ?? 3);
      if (lo !== 0) return lo;
      return (b.importance || 0) - (a.importance || 0);
    });
  });

  // Compute column widths (widest node in each column)
  const levels = Object.keys(levelGroups).map(Number).sort((a, b) => a - b);
  const colWidths = {};
  levels.forEach(lvl => {
    colWidths[lvl] = Math.max(...levelGroups[lvl].map(n => nodeDims[n.id].w));
  });

  // Compute X positions for each column (left-to-right)
  const colX = {};
  let curX = 60;
  levels.forEach(lvl => {
    colX[lvl] = curX;
    curX += colWidths[lvl] + COL_GAP;
  });

  // Compute column heights to vertically center nodes
  const colHeights = {};
  levels.forEach(lvl => {
    const group = levelGroups[lvl];
    const totalH = group.reduce((sum, n) => sum + nodeDims[n.id].h, 0)
      + ROW_GAP * (group.length - 1);
    colHeights[lvl] = totalH;
  });
  const maxColH = Math.max(...Object.values(colHeights), 0);

  // Place flow nodes: preserve existing positions; append new nodes below existing ones in their column
  const FLOW_TOP = 40;
  levels.forEach(lvl => {
    const group = levelGroups[lvl];
    const x0 = colX[lvl];
    const colW = colWidths[lvl];

    // First pass: restore preserved positions for existing nodes
    group.forEach(n => {
      const { w, h, overlayTop } = nodeDims[n.id];
      if (preservedPositions[n.id]) {
        positions[n.id] = { x: preservedPositions[n.id].x, y: preservedPositions[n.id].y, w, h, overlayTop };
      }
    });

    // Find the bottom edge of all already-placed nodes in this column
    let colBottomY = FLOW_TOP;
    group.forEach(n => {
      if (positions[n.id]) {
        colBottomY = Math.max(colBottomY, positions[n.id].y + positions[n.id].h + ROW_GAP);
      }
    });

    // Second pass: place new nodes (no preserved position) below existing ones
    group.forEach(n => {
      if (positions[n.id]) return; // already placed
      const { w, h, overlayTop } = nodeDims[n.id];
      const x = x0 + (colW - w) / 2;
      positions[n.id] = { x, y: colBottomY, w, h, overlayTop };
      colBottomY += h + ROW_GAP;
    });
  });

  // Place data nodes as a horizontal row below all flow nodes
  if (dataNodes.length > 0) {
    // Find the actual bottom of all placed flow nodes
    let flowBottom = FLOW_TOP;
    flowNodes.forEach(n => {
      if (positions[n.id]) {
        flowBottom = Math.max(flowBottom, positions[n.id].y + positions[n.id].h);
      }
    });
    const dataRowTop = flowBottom + DATA_ROW_GAP;

    // Find the rightmost edge of already-placed data nodes to append new ones after them
    let dataAppendX = 60;
    dataNodes.forEach(n => {
      if (preservedPositions[n.id]) {
        dataAppendX = Math.max(dataAppendX, preservedPositions[n.id].x + (nodeDims[n.id]?.w || NODE_W) + 40);
      }
    });

    dataNodes.forEach(n => {
      const { w, h, overlayTop } = nodeDims[n.id];
      if (preservedPositions[n.id]) {
        positions[n.id] = { x: preservedPositions[n.id].x, y: preservedPositions[n.id].y, w, h, overlayTop };
      } else {
        positions[n.id] = { x: dataAppendX, y: dataRowTop, w, h, overlayTop };
        dataAppendX += w + 40;
      }
    });
  }

  // Place child nodes inside their parents
  const childrenByParent = {};
  nodes.filter(n => n.parent).forEach(n => {
    if (!childrenByParent[n.parent]) childrenByParent[n.parent] = [];
    childrenByParent[n.parent].push(n);
  });

  Object.entries(childrenByParent).forEach(([parentId, children]) => {
    const parentPos = positions[parentId] || { x: 0, y: 0, w: GROUP_W, h: 160, overlayTop: 60 };
    const sorted = children.sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const overlayTop = parentPos.overlayTop || 60;
    const padding = 16;

    // First pass: restore preserved positions for existing children
    sorted.forEach(n => {
      if (preservedPositions[n.id]) {
        positions[n.id] = {
          x: preservedPositions[n.id].x,
          y: preservedPositions[n.id].y,
          w: NODE_W,
          h: subnodeDims[n.id]?.h || NODE_H,
        };
      }
    });

    // Find the bottom of already-placed children
    let childBottomY = parentPos.y + overlayTop + padding;
    sorted.forEach(n => {
      if (positions[n.id]) {
        childBottomY = Math.max(childBottomY, positions[n.id].y + (positions[n.id].h || NODE_H) + childGap);
      }
    });

    // Second pass: place new children below existing ones
    sorted.forEach(n => {
      if (positions[n.id]) return;
      positions[n.id] = {
        x: parentPos.x + (parentPos.w - NODE_W) / 2,
        y: childBottomY,
        w: NODE_W,
        h: subnodeDims[n.id]?.h || NODE_H,
      };
      childBottomY += (subnodeDims[n.id]?.h || NODE_H) + childGap;
    });
  });

  return { positions, levelMap, childToParent };
}

// ─── JSON → graph nodes/edges ─────────────────────────────────────────────────
function parseNodeDetails(nodeDetails, options = {}) {
  const { collapsedMap = new Map() } = options;
  const nodes = [];
  const edges = [];
  const isOverlayId = (val) => {
    const id = String(val || '');
    return id.endsWith('__hdr') || id.endsWith('__sub') || id.endsWith('__ico');
  };

  let normalized = {};
  if (Array.isArray(nodeDetails)) {
    nodeDetails.forEach((m, idx) => {
      if (!m) return;
      const key =
        (m.id && String(m.id).trim()) ||
        (m.name && String(m.name).trim()) ||
        `Module_${idx + 1}`;
      normalized[key] = { ...m, id: m.id || key, name: m.name || key };
    });
  } else {
    normalized = nodeDetails || {};
  }

  Object.entries(normalized).forEach(([modName, mod]) => {
    if (isOverlayId(modName) || isOverlayId(mod?.id) || isOverlayId(mod?.name)) return;
    const hasChildren = mod.sub_nodes?.length > 0;
    if (hasChildren) {
      const role = mod.role || '';
      const lane = getRoleLane(role);
      const importance = parseImportance(mod.importance ?? mod.priority ?? mod.criticality ?? mod.weight);
      const isCollapsed = collapsedMap.get(modName) ?? false; // Default to expanded
      nodes.push({
        id: modName,
        label: mod.name || modName,
        desc: mod.brief_description || '',
        isGroup: true,
        parent: null,
        role,
        lane,
        importance,
        functionText: mod.function || '',
        technicalText: mod.technical_methods || '',
        collapsed: isCollapsed,
      });
      // Only add sub_nodes if this specific module is not collapsed
      if (!isCollapsed) {
        mod.sub_nodes.forEach((sn, sIdx) => {
          const snId =
            (sn.id && String(sn.id).trim()) ||
            (sn.name && String(sn.name).trim()) ||
            `${modName}_sub_${sIdx + 1}`;
          nodes.push({
            id: snId,
            label: sn.name || snId,
            desc: sn.brief_description || '',
            isGroup: false,
            parent: modName,
            role,
            lane,
            importance,
            functionText: sn.function || '',
            technicalText: sn.technical_methods || '',
          });
        });
      }
    } else {
      const role = mod.role || '';
      const lane = getRoleLane(role);
      const importance = parseImportance(mod.importance ?? mod.priority ?? mod.criticality ?? mod.weight);
      nodes.push({
        id: modName,
        label: mod.name || modName,
        desc: mod.brief_description || '',
        isGroup: false,
        parent: null,
        role,
        lane,
        importance,
        functionText: mod.function || '',
        technicalText: mod.technical_methods || '',
        collapsed: false,
      });
    }

    mod.ports?.outgoing?.forEach((e, i) => {
      let src = e.originates_from_internal_node || e.source_node || modName;
      let tgt = e.connects_to_internal_node || e.target_node || e.target_mod;
      if (!src || src === 'null' || src === 'None') src = modName;
      if (!tgt || tgt === 'null' || tgt === 'None') tgt = e.target_mod;

      // If this module is collapsed, only show module-level edges (skip internal_flow edges)
      const isCollapsed = collapsedMap.get(modName) ?? false;
      const isInternalEdge = (e.originates_from_internal_node && e.originates_from_internal_node !== modName);
      if (isCollapsed && isInternalEdge) return;

      if (src && tgt) edges.push({ id: `out-${modName}-${i}`, source: src, target: tgt, label: e.edge_label || '', dashed: false });
    });

    // Skip internal_flow edges when this module is collapsed
    const isCollapsed = collapsedMap.get(modName) ?? false;
    if (!isCollapsed) {
      mod.ports?.internal_flow?.forEach((e, i) => {
        const src = e.source_node, tgt = e.target_node;
        if (src && tgt && src !== 'null' && src !== 'None' && tgt !== 'null' && tgt !== 'None')
          edges.push({ id: `in-${modName}-${i}`, source: src, target: tgt, label: e.edge_label || '', dashed: true });
      });
    }
  });

  const validIds = new Set(nodes.map(n => n.id));

  // Build a map from sub-node ID → parent module ID for all modules (including collapsed ones)
  // so that edges pointing to/from a collapsed sub-node can be redirected to the parent module.
  const subNodeToParent = {};
  Object.entries(normalized).forEach(([modName, mod]) => {
    mod.sub_nodes?.forEach(sn => {
      const snId =
        (sn.id && String(sn.id).trim()) ||
        (sn.name && String(sn.name).trim());
      if (snId) subNodeToParent[snId] = modName;
    });
  });

  // Resolve an endpoint: if it's not a valid (rendered) node, try to redirect to its parent module
  const resolve = (id) => {
    if (validIds.has(id)) return id;
    const parent = subNodeToParent[id];
    if (parent && validIds.has(parent)) return parent;
    return id;
  };

  const resolvedEdges = edges
    .map(e => ({ ...e, source: resolve(e.source), target: resolve(e.target) }))
    .filter(e => validIds.has(e.source) && validIds.has(e.target) && e.source !== e.target);

  return { nodes, edges: resolvedEdges };
}

// ─── Graph → JSON ─────────────────────────────────────────────────────────────
function graphToNodeDetails(graph, originalNodeDetails, collapsedMap = new Map()) {
  const normalizeText = (text) => String(text || '').replace(/\s+/g, ' ').trim();
  let normalized = {};
  if (Array.isArray(originalNodeDetails)) {
    originalNodeDetails.forEach(m => {
      if (!m) return;
      const raw = JSON.parse(JSON.stringify(m));
      const key = m.id || m.name;
      if (key) normalized[key] = raw;
      if (m.name) normalized[m.name] = raw;
    });
  } else {
    normalized = JSON.parse(JSON.stringify(originalNodeDetails || {}));
  }

  const allNodes = graph.getNodes();
  const allEdges = graph.getEdges();

  const childParentMap = {};
  const groupChildren = {};
  const titleTextMap = {};
  const descTextMap = {};
  allNodes.forEach(n => {
    const id = String(n.id || '');
    if (id.endsWith('__hdr')) {
      const baseId = id.slice(0, -5);
      const text = n.attr('label/text') || n.getProp('label') || '';
      if (text) titleTextMap[baseId] = normalizeText(text);
      return;
    }
    if (id.endsWith('__ico')) {
      return;
    }
    if (id.endsWith('__sub')) {
      const baseId = id.slice(0, -5);
      const text = n.attr('label/text') || n.getProp('label') || '';
      if (text) descTextMap[baseId] = normalizeText(text);
      return;
    }
    const parent = n.getParent();
    if (parent && parent.isNode()) {
      // Only track real child nodes, not overlay helpers
      if (!id.endsWith('__hdr') && !id.endsWith('__sub') && !id.endsWith('__ico') && !id.endsWith('__toggle')) {
        childParentMap[n.id] = parent.id;
        if (!groupChildren[parent.id]) groupChildren[parent.id] = [];
        groupChildren[parent.id].push(n.id);
      }
    }
  });

  const groupIds = new Set(Object.keys(groupChildren));
  const childIds = new Set(Object.keys(childParentMap));
  const newDetails = {};

  allNodes.forEach(node => {
    const nodeId = String(node.id || '');
    if (nodeId.endsWith('__hdr') || nodeId.endsWith('__sub') || nodeId.endsWith('__ico') || nodeId.endsWith('__toggle')) return;
    if (childIds.has(node.id)) return;
    const label = node.attr('label/text') || node.getProp('label') || nodeId;
    const displayName = titleTextMap[nodeId] || normalizeText(label) || nodeId;
    const displayDesc = descTextMap[nodeId] || '';
    const isGroup = groupIds.has(nodeId);
    // Also treat as group if the original data has sub_nodes (collapsed state: children not in graph)
    const origForGroup = normalized[nodeId] || normalized[label] || {};
    const isGroupOrCollapsed = isGroup || (origForGroup.sub_nodes && origForGroup.sub_nodes.length > 0);
    // 优先使用 id 查找，如果没有再用 label
    const orig = normalized[nodeId] || normalized[label] || {};

    const entry = {
      id: nodeId,
      name: displayName,
      role: orig.role || 'processing', // 保留 role，默认为 processing
      brief_description: displayDesc || orig.brief_description || orig.desc || '',
      function: orig.function || '',
      technical_methods: orig.technical_methods || '',
      ports: { incoming: [], outgoing: [], internal_flow: [] },
      ...(orig.reference ? { reference: orig.reference } : {}),
      ...(orig.references ? { references: orig.references } : { references: 'None' }),
    };

    if (isGroupOrCollapsed) {
      // If there are children in the graph (expanded), derive sub_nodes from them
      if (groupChildren[nodeId] && groupChildren[nodeId].length > 0) {
        entry.sub_nodes = groupChildren[nodeId].map(childId => {
          const childNode = graph.getCellById(childId);
          let childLabel = childNode?.attr('label/text') || childNode?.getProp('label') || childId;
          if (childLabel && typeof childLabel === 'string') {
            childLabel = childLabel.split('\n')[0].trim();
          }
          const origChild = orig.sub_nodes?.find(sn => sn.id === childId) || orig.sub_nodes?.find(sn => sn.name === childId) || {};
          return {
            id: childId,
            name: childLabel,
            brief_description: origChild.brief_description || '',
            function: origChild.function || '',
            technical_methods: origChild.technical_methods || ''
          };
        });
      } else {
        // Collapsed — always preserve sub_nodes. Try multiple sources in priority order:
        // 1. orig (from normalized originalNodeDetails)
        // 2. raw originalNodeDetails lookup (in case normalized lost sub_nodes due to a prior stale emit)
        let origSubNodes = orig.sub_nodes;
        if (!origSubNodes || origSubNodes.length === 0) {
          if (Array.isArray(originalNodeDetails)) {
            const rawMod = originalNodeDetails.find(m => (m.id || m.name) === nodeId || m.name === nodeId);
            origSubNodes = rawMod?.sub_nodes;
          } else if (originalNodeDetails && typeof originalNodeDetails === 'object') {
            origSubNodes = originalNodeDetails[nodeId]?.sub_nodes;
          }
        }
        if (origSubNodes && origSubNodes.length > 0) {
          entry.sub_nodes = origSubNodes;
        }
      }
    }

    newDetails[nodeId] = entry;
  });

  allEdges.forEach(edge => {
    const srcId = edge.getSourceCellId();
    const tgtId = edge.getTargetCellId();
    if (!srcId || !tgtId) return;
    const labelFromLabels = edge.getLabels?.()[0]?.attrs?.label?.text || edge.getLabels?.()[0]?.attrs?.text?.text || '';
    const label = edge.attr('label/text') || edge.getProp('label') || labelFromLabels || '';
    const isDashed = edge.getProp('dashed') === true;
    const srcParent = childParentMap[srcId];
    const tgtParent = childParentMap[tgtId];

    if (isDashed && srcParent && srcParent === tgtParent) {
      const parentEntry = newDetails[srcParent];
      if (parentEntry) parentEntry.ports.internal_flow.push({ source_node: srcId, target_node: tgtId, edge_label: label });
    } else {
      const ownerModId = srcParent || srcId;
      const ownerEntry = newDetails[ownerModId];
      if (ownerEntry) {
        const tgtModId = tgtParent || tgtId;
        ownerEntry.ports.outgoing.push({ originates_from_internal_node: srcParent ? srcId : null, connects_to_internal_node: tgtParent ? tgtId : null, source_node: srcId, target_node: tgtId, target_mod: tgtModId, edge_label: label });
      }
    }
  });

  return newDetails;
}

// ─── Render nodes+edges into an existing graph instance ──────────────────────
function renderGraph(graph, nodeDetails, options = {}) {
  const { collapsedMap = new Map(), skipZoom = false, preservedPositions = {} } = options;
  try {
    const t = Array.isArray(nodeDetails) ? 'array' : typeof nodeDetails;
    const n = Array.isArray(nodeDetails) ? nodeDetails.length : (nodeDetails ? Object.keys(nodeDetails).length : 0);
    console.debug('[X6] renderGraph input:', t, n, 'collapsedMap size:', collapsedMap.size);
  } catch (e) { void e; }

  // Suspend view updates to prevent any intermediate repaints / scroll jumps
  try { graph.startBatch('render'); } catch (e) { void e; }

  try { graph.resetCells([]); } catch (e) { void e; }
  try { graph.clearCells(); } catch (e) { void e; }
  try {
    const cells = graph.getCells();
    if (cells && cells.length) graph.removeCells(cells);
  } catch (e) { void e; }
  if (
    !nodeDetails ||
    (Array.isArray(nodeDetails) ? nodeDetails.length === 0 : Object.keys(nodeDetails).length === 0)
  ) {
    return;
  }

  let parsedNodes = [];
  let parsedEdges = [];
  try {
    const parsed = parseNodeDetails(nodeDetails, { collapsedMap });
    parsedNodes = parsed.nodes;
    parsedEdges = parsed.edges;
  } catch (err) {
    console.error("Parse error:", err);
    return;
  }

  if (parsedNodes.length === 0) return;

  let positions = {};
  let levelMap = {};
  let childToParent = {};
  try {
    const layout = runRoleLayout(parsedNodes, parsedEdges, preservedPositions);
    positions = layout.positions;
    levelMap = layout.levelMap || {};
    childToParent = layout.childToParent || {};
  } catch (e) {
    console.error('Layout error:', e);
    parsedNodes.forEach((n, i) => { positions[n.id] = { x: (i % 5) * 260, y: Math.floor(i / 5) * 100, w: NODE_W, h: NODE_H }; });
  }

  const colorOf = (n) => ROLE_THEME[n.lane] || ROLE_THEME.unknown;

  const portsDef = (id, lane = 'unknown') => {
    const color = (ROLE_THEME[lane] || ROLE_THEME.unknown).stroke;
    return ({
      groups: {
        top: { position: 'top', attrs: { circle: { r: 4, magnet: true, stroke: color, fill: '#fff', strokeWidth: 1.5 } } },
        bottom: { position: 'bottom', attrs: { circle: { r: 4, magnet: true, stroke: color, fill: '#fff', strokeWidth: 1.5 } } },
        left: { position: 'left', attrs: { circle: { r: 4, magnet: true, stroke: color, fill: '#fff', strokeWidth: 1.5 } } },
        right: { position: 'right', attrs: { circle: { r: 4, magnet: true, stroke: color, fill: '#fff', strokeWidth: 1.5 } } },
      },
      items: [
        { id: `${id}-top`, group: 'top' },
        { id: `${id}-bottom`, group: 'bottom' },
        { id: `${id}-left`, group: 'left' },
        { id: `${id}-right`, group: 'right' },
      ],
    });
  };

  const clampInside = (child, parent) => {
    if (!child || !parent) return;
    const pb = parent.getBBox();
    const data = parent.getData ? parent.getData() : {};
    const overlayTop = (data && data.overlayTop) || 0;
    const pad = 12;
    const b = child.getBBox();
    let nx = b.x;
    let ny = b.y;
    const minX = pb.x + pad;
    const maxX = pb.x + pb.width - b.width - pad;
    const minY = pb.y + overlayTop + pad;
    const maxY = pb.y + pb.height - b.height - pad;
    if (nx < minX) nx = minX;
    if (nx > maxX) nx = Math.max(minX, maxX);
    if (ny < minY) ny = minY;
    if (ny > maxY) ny = Math.max(minY, maxY);
    if (nx !== b.x || ny !== b.y) child.position(nx, ny);
  };

  // Groups first
  const overlayTopMap = {};
  parsedNodes.filter(n => n.isGroup).forEach(n => {
    const pos = positions[n.id] || { x: 0, y: 0, w: 240, h: 120 };
    const theme = colorOf(n);
    const icons = inferNodeIcons(n);
    const iconText = icons.join(' ');
    const titleText = n.label;
    const innerW = Math.max(10, pos.w - 24);
    const titleLines = wrapTextToLines(titleText, innerW, 20, 'bold');
    const rawDesc = (n.desc || '').replace(/\s+/g, ' ').trim();
    const descLines = wrapTextToLines(rawDesc, innerW, 13, 'normal');
    const titleH = Math.max(24, titleLines.length * 22);
    const descH = 0; // description hidden
    const overlayTop = 10 + titleH + (descH ? 6 + descH : 0);
    overlayTopMap[n.id] = overlayTop;

    // If collapsed, render as a simple node instead of a group
    const isCollapsed = n.collapsed === true;
    const groupHeight = isCollapsed ? Math.max(pos.h, overlayTop + 20) : Math.max(pos.h, overlayTop + 80);

    const group = graph.addNode({
      id: n.id, shape: 'rect', x: pos.x, y: pos.y, width: pos.w, height: groupHeight, zIndex: 0,
      attrs: {
        body: {
          fill: theme.fill,
          stroke: theme.stroke,
          strokeWidth: 2.2,
          strokeDasharray: isCollapsed ? '0' : '6 3', // Solid border when collapsed
          rx: 10,
          ry: 10
        },
        label: { text: '', fontSize: 16, fontWeight: 'bold', fill: theme.text, refX: 0.5, refY: 0.5, textAnchor: 'middle', textVerticalAnchor: 'middle' },
      },
      ports: portsDef(n.id, n.lane),
    });
    try { group.setData && group.setData({ overlayTop: overlayTop, collapsed: isCollapsed }); } catch (e) { void e; }
    const titleNode = graph.addNode({
      id: `${n.id}__hdr`,
      shape: 'rect',
      x: pos.x + 12,
      y: pos.y + 10,
      width: innerW,
      height: titleH,
      zIndex: 5,
      attrs: {
        body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
        label: { text: titleLines.join('\n'), fontSize: 20, fontWeight: 'bold', fill: theme.text, textAnchor: 'middle', textVerticalAnchor: 'top', refX: 0.5, refY: 0, pointerEvents: 'none' },
      },
      ports: { groups: {}, items: [] },
    });
    const iconNode = iconText ? graph.addNode({
      id: `${n.id}__ico`,
      shape: 'rect',
      x: pos.x + pos.w - 38,
      y: pos.y + 8,
      width: 26,
      height: 22,
      zIndex: 6,
      attrs: {
        body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
        label: { text: iconText, fontSize: 22, fill: theme.text, textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5, pointerEvents: 'none' },
      },
      ports: { groups: {}, items: [] },
    }) : null;

    // Add collapse/expand indicator
    const collapseIndicator = graph.addNode({
      id: `${n.id}__toggle`,
      shape: 'rect',
      x: pos.x + pos.w - (iconText ? 68 : 38),
      y: pos.y + 8,
      width: 24,
      height: 22,
      zIndex: 6,
      attrs: {
        body: { fill: 'rgba(255,255,255,0.8)', stroke: theme.stroke, strokeWidth: 1, rx: 4, ry: 4, cursor: 'pointer' },
        label: { text: isCollapsed ? '▶' : '▼', fontSize: 14, fill: theme.text, textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5, cursor: 'pointer' },
      },
      ports: { groups: {}, items: [] },
    });

    const descNode = graph.addNode({
      id: `${n.id}__sub`,
      shape: 'rect',
      x: pos.x + 12,
      y: pos.y + 10 + titleH + 6,
      width: innerW,
      height: descH || 14,
      zIndex: 5,
      attrs: {
        body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
        label: { text: '' /* description hidden */, fontSize: 13, fill: '#666', textAnchor: 'middle', textVerticalAnchor: 'top', refX: 0.5, refY: 0, pointerEvents: 'none' },
      },
      ports: { groups: {}, items: [] },
    });

    group.addChild(titleNode);
    group.addChild(collapseIndicator);
    if (iconNode) group.addChild(iconNode);
    group.addChild(descNode);
    group.resize(pos.w, groupHeight);
  });

  // Leaf nodes
  parsedNodes.filter(n => !n.isGroup).forEach(n => {
    const pos = positions[n.id] || { x: 0, y: 0, w: NODE_W, h: NODE_H };
    const theme = colorOf(n);
    const oy = 0;
    const isTopLevelLeaf = !n.parent;
    if (isTopLevelLeaf) {
      const icons = inferNodeIcons(n);
      const iconText = icons.join(' ');
      const titleText = n.label;
      const innerW = Math.max(10, pos.w - 24);
      const rawDesc = (n.desc || '').replace(/\s+/g, ' ').trim();
      const titleLines = wrapTextToLines(titleText, innerW, 20, 'bold');
      const descLines = wrapTextToLines(rawDesc, innerW, 15, 'normal');
      const gap = 12;
      const titleH = Math.max(24, titleLines.length * 22);
      const descH = 0; // description hidden
      const dynH = Math.max(pos.h, 20 + titleH + 10);
      const group = graph.addNode({
        id: n.id,
        shape: 'rect',
        x: pos.x,
        y: pos.y + oy,
        width: pos.w,
        height: dynH,
        zIndex: 1,
        attrs: {
          body: { fill: theme.fill, stroke: theme.stroke, strokeWidth: 2.2, strokeDasharray: '6 3', rx: 10, ry: 10 },
          label: { text: '', fontSize: 17, fill: theme.text, textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5 },
        },
        ports: portsDef(n.id, n.lane),
      });
      const overlayTop = 12 + titleH + (descH ? gap + descH : 0);
      try { group.setData && group.setData({ overlayTop }); } catch (e) { void e; }
      const titleNode = graph.addNode({
        id: `${n.id}__hdr`,
        shape: 'rect',
        x: pos.x + 12,
        y: pos.y + oy + 12,
        width: innerW,
        height: titleH,
        zIndex: 5,
        attrs: {
          body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
          label: { text: titleLines.join('\n'), fontSize: 20, fontWeight: 'bold', fill: theme.text, textAnchor: 'middle', textVerticalAnchor: 'top', refX: 0.5, refY: 0, pointerEvents: 'none' },
        },
        ports: { groups: {}, items: [] },
      });
      const iconNode = iconText ? graph.addNode({
        id: `${n.id}__ico`,
        shape: 'rect',
        x: pos.x + pos.w - 38,
        y: pos.y + oy + 8,
        width: 26,
        height: 22,
        zIndex: 6,
        attrs: {
          body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
          label: { text: iconText, fontSize: 23, fill: theme.text, textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5, pointerEvents: 'none' },
        },
        ports: { groups: {}, items: [] },
      }) : null;
      const descNode = graph.addNode({
        id: `${n.id}__sub`,
        shape: 'rect',
        x: pos.x + 12,
        y: pos.y + oy + 12 + titleH + gap,
        width: innerW,
        height: descH || 14,
        zIndex: 5,
        attrs: {
          body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
          label: { text: '' /* description hidden */, fontSize: 15, fill: '#666', textAnchor: 'middle', textVerticalAnchor: 'top', refX: 0.5, refY: 0, pointerEvents: 'none' },
        },
        ports: { groups: {}, items: [] },
      });
      group.addChild(titleNode);
      if (iconNode) group.addChild(iconNode);
      group.addChild(descNode);
    } else {
      const innerW = Math.max(10, pos.w - 24);
      const rawDesc = (n.desc || '').replace(/\s+/g, ' ').trim();
      const titleLines = wrapTextToLines(n.label, innerW, 16, 'bold');
      const descLines = wrapTextToLines(rawDesc, innerW, 14, 'normal');
      const lines = descLines.length > 0 ? [...titleLines, ...descLines] : [...titleLines];
      const labelText = lines.join('\n');
      const dynH = Math.max(pos.h, 18 + lines.length * 14);
      const added = graph.addNode({
        id: n.id,
        shape: 'rect',
        x: pos.x,
        y: pos.y + oy,
        width: pos.w,
        height: dynH,
        zIndex: 1,
        attrs: {
          body: { fill: theme.light, stroke: theme.stroke, strokeWidth: 1.6 + (n.importance || 0) * 0.9, rx: 6, ry: 6 },
          label: { text: labelText, fontSize: 15, fill: '#333', textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5 },
        },
        ports: portsDef(n.id, n.lane),
      });
      if (n.parent) {
        const parentCell = graph.getCellById(n.parent);
        if (parentCell) {
          parentCell.addChild(added);
          clampInside(added, parentCell);
        }
      }
      return;
    }
    if (n.parent) {
      const parentCell = graph.getCellById(n.parent);
      if (parentCell) {
        const base = graph.getCellById(n.id);
        if (base) parentCell.addChild(base);
      }
    }
  });

  // Detect feedback edges: source level >= target level (going backward or sideways in the DAG)
  const getTopLevel = (nodeId) => {
    const topId = childToParent[nodeId] || nodeId;
    return levelMap[topId] ?? 0;
  };
  const isFeedbackEdge = (e) => {
    if (e.dashed) return false; // internal_flow edges are not feedback
    const srcLvl = getTopLevel(e.source);
    const tgtLvl = getTopLevel(e.target);
    return srcLvl >= tgtLvl && e.source !== e.target;
  };

  // Edges — spread edges sharing the same source or same target to avoid overlap
  const pairCount = {};
  const pairIndex = {};
  const srcCount = {};
  const srcIndex = {};
  const tgtCount = {};
  const tgtIndex = {};
  parsedEdges.forEach(e => {
    const key = `${e.source}||${e.target}`;
    pairCount[key] = (pairCount[key] || 0) + 1;
    srcCount[e.source] = (srcCount[e.source] || 0) + 1;
    tgtCount[e.target] = (tgtCount[e.target] || 0) + 1;
  });
  parsedEdges.forEach((e) => {
    const key = `${e.source}||${e.target}`;
    const total = pairCount[key] || 1;
    const pi = pairIndex[key] || 0;
    pairIndex[key] = pi + 1;

    const si = srcIndex[e.source] || 0;
    srcIndex[e.source] = si + 1;
    const ti = tgtIndex[e.target] || 0;
    tgtIndex[e.target] = ti + 1;

    // Spread parallel edges (same src+tgt pair)
    const spreadStep = 24;
    const parallelOffset = total === 1 ? 0 : (pi - (total - 1) / 2) * spreadStep;

    // For edges sharing the same source, stagger exit point vertically
    const srcTotal = srcCount[e.source] || 1;
    const srcStep = 18;
    const srcOffset = srcTotal === 1 ? 0 : (si - (srcTotal - 1) / 2) * srcStep;

    // For edges sharing the same target, stagger entry point vertically
    const tgtTotal = tgtCount[e.target] || 1;
    const tgtStep = 18;
    const tgtOffset = tgtTotal === 1 ? 0 : (ti - (tgtTotal - 1) / 2) * tgtStep;

    const feedback = isFeedbackEdge(e);

    // Feedback edges: orange dashed arc with "↩ when: ..." label prefix
    // Forward edges: solid dark straight line
    const edgeColor = feedback ? '#f97316' : (e.dashed ? '#94a3b8' : '#64748b');
    const edgeStrokeWidth = feedback ? 1.4 : (e.dashed ? 1.3 : 1.6);
    const edgeDash = feedback ? '8 4' : (e.dashed ? '7 5' : '0');

    // Prefix feedback edge labels with a trigger indicator
    let labelText = e.label || '';
    if (feedback && labelText) {
      // If label doesn't already start with a trigger keyword, prefix it
      const triggerPrefixes = ['when', 'if', 'on', '↩', 'trigger', 'after', 'upon'];
      const lowerLabel = labelText.toLowerCase();
      const hasTrigger = triggerPrefixes.some(p => lowerLabel.startsWith(p));
      if (!hasTrigger) labelText = `↩ when: ${labelText}`;
    } else if (feedback && !labelText) {
      labelText = '↩ feedback';
    }

    const labels = labelText
      ? [{
        attrs: {
          label: {
            text: labelText,
            fontSize: feedback ? 10 : 11,
            fontWeight: feedback ? 500 : 600,
            fill: feedback ? '#c2410c' : '#334155',
            fontStyle: feedback ? 'italic' : 'normal',
          },
          rect: {
            fill: feedback ? '#fff7ed' : '#ffffff',
            stroke: feedback ? '#fed7aa' : '#cbd5e1',
            strokeWidth: 1,
            rx: 3,
            ry: 3,
            refWidth: 6,
            refHeight: 4,
            refX: -3,
            refY: -2,
          },
        },
        position: { distance: 0.5, offset: parallelOffset },
      }]
      : [];

    // Feedback edges use a smooth curved arc routed above/below the main flow
    // Forward edges use manhattan orthogonal routing
    const edgeRouter = feedback
      ? {
          name: 'manhattan',
          args: {
            padding: { top: 60, bottom: 60, left: 60, right: 60 },
            startDirections: ['bottom', 'top'],
            endDirections: ['bottom', 'top'],
          },
        }
      : {
          name: 'manhattan',
          args: {
            padding: { top: 40, bottom: 40, left: 48, right: 48 },
            startDirections: ['top', 'bottom', 'left', 'right'],
            endDirections: ['top', 'bottom', 'left', 'right'],
          },
        };

    const edgeConnector = feedback
      ? { name: 'smooth' }
      : { name: 'rounded', args: { radius: 10 } };

    graph.addEdge({
      id: e.id,
      source: { cell: e.source, anchor: { name: 'center', args: { dy: srcOffset } } },
      target: { cell: e.target, anchor: { name: 'center', args: { dy: tgtOffset } } },
      dashed: e.dashed,
      zIndex: feedback ? 1 : 2, // feedback edges render behind forward edges
      router: edgeRouter,
      connector: edgeConnector,
      attrs: {
        line: {
          stroke: edgeColor,
          strokeWidth: edgeStrokeWidth,
          strokeDasharray: edgeDash,
          targetMarker: { name: 'block', size: 7, fill: edgeColor },
        },
      },
      labels,
    });
  });

  // End suspended batch — triggers a single repaint
  try { graph.stopBatch('render'); } catch (e) { void e; }

  if (!skipZoom) {
    setTimeout(() => {
      try {
        graph.zoomToFit({ padding: 20, maxScale: 1.8 });
      } catch (e) { void e; }
    }, 50);
  }
}

// ─── Component ────────────────────────────────────────────────────────────────
const X6FlowViewer = forwardRef(function X6FlowViewer(
  { nodeDetails, onNodeClick, onPaneClick, onNodeDetailsChange },
  ref
) {
  const containerRef = useRef(null);
  const graphRef = useRef(null);
  const graphReadyRef = useRef(false);
  const isInternalUpdateRef = useRef(false);  // Flag to prevent render loops
  const isRenderingRef = useRef(false);
  const lastInternalSigRef = useRef('');
  const nodeDetailsRef = useRef(nodeDetails);
  const onNodeClickRef = useRef(onNodeClick);
  const onPaneClickRef = useRef(onPaneClick);
  const onNodeDetailsChangeRef = useRef(onNodeDetailsChange);
  // Persists node positions across re-renders; updated on drag and after each layout
  const savedPositionsRef = useRef({});
  const pendingRenderRef = useRef(null);  // Queues data that arrived before graph was ready
  const [collapsedModules, setCollapsedModules] = useState(new Map()); // Track per-module collapse state
  const collapsedModulesRef = useRef(collapsedModules); // Ref so closures always see latest value
  collapsedModulesRef.current = collapsedModules;
  const setCollapsedModulesRef = useRef(setCollapsedModules); // Ref to allow access inside closures
  setCollapsedModulesRef.current = setCollapsedModules;
  nodeDetailsRef.current = nodeDetails;
  onNodeClickRef.current = onNodeClick;
  onPaneClickRef.current = onPaneClick;
  onNodeDetailsChangeRef.current = onNodeDetailsChange;

  const createGraphInstance = useCallback((w, h) => {
    const container = containerRef.current;
    if (!container) return null;
    const graph = new Graph({
      container,
      width: w,
      height: h,
      autoResize: true,
      background: { color: '#f8f9fa' },
      grid: { visible: true, type: 'dot', args: { color: '#ccc', thickness: 1 } },
      panning: { enabled: true, eventTypes: ['leftMouseDown', 'mouseWheel'] },
      mousewheel: { enabled: true, modifiers: null },
      connecting: {
        router: {
          name: 'manhattan',
          args: {
            padding: { top: 40, bottom: 40, left: 48, right: 48 },
            startDirections: ['top', 'bottom', 'left', 'right'],
            endDirections: ['top', 'bottom', 'left', 'right'],
          },
        },
        connector: { name: 'rounded', args: { radius: 10 } },
        anchor: 'nodeCenter',
        connectionPoint: 'boundary',
        allowBlank: false,
        allowLoop: false,
        highlight: true,
        createEdge() {
          return new Shape.Edge({
            attrs: { line: { stroke: '#64748b', strokeWidth: 1.6, targetMarker: { name: 'block', size: 7, fill: '#64748b' } } },
            zIndex: 2,
          });
        },
      },
      highlighting: {
        magnetAdsorbed: { name: 'stroke', args: { attrs: { fill: '#fff', stroke: '#217cf3', strokeWidth: 3 } } },
      },
      interacting: { nodeMovable: true, edgeMovable: true, edgeLabelMovable: false, magnetConnectable: true },
    });
    graph.use(new Selection({ enabled: true, multiple: true, rubberband: true, movable: true, showNodeSelectionBox: true, modifiers: ['shift'] }));
    graph.use(new Export());
    const emitNodeDetailsChange = (g) => {
      if (!g || !onNodeDetailsChangeRef.current || isRenderingRef.current) return;
      const nextDetails = graphToNodeDetails(g, nodeDetailsRef.current, collapsedModulesRef.current);
      let sig = '';
      try {
        sig = JSON.stringify(nextDetails);
      } catch (e) { void e; }
      if (sig) lastInternalSigRef.current = sig;
      isInternalUpdateRef.current = true;
      onNodeDetailsChangeRef.current(nextDetails);
    };
    const syncOverlayLayout = (parentNode) => {
      if (!parentNode || !(parentNode.isNode && parentNode.isNode())) return;
      const id = String(parentNode.id || '');
      if (!id || id.endsWith('__hdr') || id.endsWith('__sub') || id.endsWith('__ico') || id.endsWith('__toggle')) return;
      const data = parentNode.getData ? parentNode.getData() : {};
      if (!data || typeof data.overlayTop !== 'number') return;
      const hdr = graph.getCellById(`${id}__hdr`);
      const sub = graph.getCellById(`${id}__sub`);
      const ico = graph.getCellById(`${id}__ico`);
      const toggle = graph.getCellById(`${id}__toggle`);
      if (!hdr || !sub) return;
      const pb = parentNode.getBBox();
      const innerW = Math.max(10, pb.width - 24);
      const titleRaw = String(hdr.attr('label/text') || hdr.getProp('label') || '').replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
      const descRaw = String(sub.attr('label/text') || sub.getProp('label') || '').replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
      const titleLines = wrapTextToLines(titleRaw, innerW, 14, 'bold');
      const descLines = wrapTextToLines(descRaw, innerW, 12, 'normal');
      const titleH = Math.max(20, titleLines.length * 16);
      const descH = descLines.length > 0 ? descLines.length * 14 : 0;
      const overlayTop = 10 + titleH + (descH ? 6 + descH : 0);
      hdr.position(pb.x + 12, pb.y + 10);
      hdr.resize(innerW, titleH);
      hdr.attr('label/text', titleLines.join('\n'));
      if (ico) ico.position(pb.x + pb.width - 38, pb.y + 8);
      if (toggle) toggle.position(pb.x + pb.width - (ico ? 68 : 38), pb.y + 8);
      sub.position(pb.x + 12, pb.y + 10 + titleH + 6);
      sub.resize(innerW, descH || 14);
      sub.attr('label/text', descLines.join('\n'));
      try { parentNode.setData && parentNode.setData({ ...data, overlayTop }); } catch (e) { void e; }
      const minH = overlayTop + 80;
      if (pb.height < minH) parentNode.resize(pb.width, minH);
    };
    graph.on('node:change:position', ({ node }) => {
      const id = String(node?.id || '');
      if (id.endsWith('__hdr') || id.endsWith('__sub') || id.endsWith('__ico') || id.endsWith('__toggle')) return;
      // Persist position so layout can restore it on next re-render
      const pos = node.getPosition();
      const size = node.getSize();
      if (pos) savedPositionsRef.current[id] = { x: pos.x, y: pos.y, w: size?.width, h: size?.height };
      const parent = node.getParent && node.getParent();
      if (parent && parent.isNode && parent.isNode()) {
        // Child node moved inside group — clamp to boundaries
        const pb = parent.getBBox();
        const data = parent.getData ? parent.getData() : {};
        const overlayTop = (data && data.overlayTop) || 0;
        const pad = 12;
        const b = node.getBBox();
        let nx = b.x;
        let ny = b.y;
        const minX = pb.x + pad;
        const maxX = pb.x + pb.width - b.width - pad;
        const minY = pb.y + overlayTop + pad;
        const maxY = pb.y + pb.height - b.height - pad;
        if (nx < minX) nx = minX;
        if (nx > maxX) nx = Math.max(minX, maxX);
        if (ny < minY) ny = minY;
        if (ny > maxY) ny = Math.max(minY, maxY);
        if (nx !== b.x || ny !== b.y) node.position(nx, ny);
      } else {
        // Top-level node moved — sync overlay nodes (hdr/sub/ico/toggle) positions
        syncOverlayLayout(node);
      }
    });
    graph.on('node:click', ({ node }) => {
      const nodeId = String(node.id || '');

      // Check if clicked on toggle indicator
      if (nodeId.endsWith('__toggle')) {
        const moduleId = nodeId.slice(0, -8); // Remove '__toggle'
        const moduleNode = graph.getCellById(moduleId);
        if (moduleNode) {
          const data = moduleNode.getData ? moduleNode.getData() : {};
          const isCurrentlyCollapsed = data.collapsed ?? true;

          // Toggle collapse state for this module via ref (closure-safe)
          setCollapsedModulesRef.current(prev => {
            const next = new Map(prev);
            next.set(moduleId, !isCurrentlyCollapsed);
            return next;
          });
        }
        return;
      }

      if (onNodeClickRef.current) onNodeClickRef.current(nodeId);
    });
    graph.on('blank:click', () => { if (onPaneClickRef.current) onPaneClickRef.current(); });
    graph.on('node:dblclick', ({ node, e }) => {
      e.stopPropagation();
      if (isRenderingRef.current) return;

      let targetId = String(node.id || '');
      if (targetId.endsWith('__hdr') || targetId.endsWith('__sub') || targetId.endsWith('__ico')) {
        targetId = targetId.slice(0, -5);
      }

      const baseNode = graph.getCellById(targetId) || node;
      const hdrCell = graph.getCellById(`${targetId}__hdr`);

      let cur = '';
      if (hdrCell) {
        cur = String(hdrCell.attr('label/text') || '').replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
      } else {
        cur = baseNode.attr('label/text') || baseNode.getProp('label') || baseNode.id;
        // strip out description if it's a sub-node
        const lines = cur.split('\n');
        if (lines.length > 0) cur = lines[0]; // simplistic heuristic
      }

      const next = window.prompt('重命名节点:', cur);
      if (next !== null && next.trim() && next.trim() !== cur) {
        const newName = next.trim();

        if (hdrCell) {
          hdrCell.attr('label/text', newName); // raw text
          try { syncOverlayLayout(baseNode); } catch (ex) { console.error(ex); }
        } else {
          // Standard node without hdrcell
          let dynText = newName;
          try {
            const innerW = Math.max(10, baseNode.getSize().width - 24);
            const details = nodeDetailsRef.current || {};
            let origNode = details[targetId];
            if (!origNode) {
              // maybe it's inside a group
              Object.values(details).forEach(m => {
                if (m.sub_nodes) {
                  const child = m.sub_nodes.find(sn => sn.id === targetId || sn.name === targetId);
                  if (child) origNode = child;
                }
              });
            }
            const rawDesc = origNode ? (origNode.brief_description || origNode.desc || '') : '';
            const titleLines = wrapTextToLines(newName, innerW, 14, 'bold');
            const descLines = wrapTextToLines(rawDesc.replace(/\s+/g, ' ').trim(), innerW, 12, 'normal');
            dynText = descLines.length > 0 ? [...titleLines, ...descLines].join('\n') : titleLines.join('\n');
          } catch (e) { void e; }
          baseNode.attr('label/text', dynText);
        }

        const g = graphRef.current;
        emitNodeDetailsChange(g);
      }
    });
    graph.on('edge:dblclick', ({ edge, e }) => {
      e.stopPropagation();
      if (isRenderingRef.current) return;
      const cur =
        edge.getLabels?.()[0]?.attrs?.label?.text ||
        edge.getLabels?.()[0]?.attrs?.text?.text ||
        edge.attr('label/text') ||
        edge.getProp('label') ||
        '';
      const next = window.prompt('编辑连线标签:', cur);
      if (next !== null) {
        const text = next.trim();
        if (text) {
          edge.setLabels([{ attrs: { label: { text, fontSize: 11, fontWeight: 600, fill: '#475569' } }, position: { distance: 0.5, offset: 12 } }]);
        } else {
          edge.setLabels([]);
        }
      }
      const g = graphRef.current;
      emitNodeDetailsChange(g);
    });
    graph.on('node:removed', ({ node }) => {
      if (isRenderingRef.current) return;
      const g = graphRef.current;
      if (!g) return;
      try {
        const id = String(node?.id || '');
        if (!id.endsWith('__hdr') && !id.endsWith('__sub') && !id.endsWith('__ico') && !id.endsWith('__toggle')) {
          const hdr = g.getCellById(`${id}__hdr`);
          const sub = g.getCellById(`${id}__sub`);
          const ico = g.getCellById(`${id}__ico`);
          const toggle = g.getCellById(`${id}__toggle`);
          const toRemove = [];
          if (hdr) toRemove.push(hdr);
          if (sub) toRemove.push(sub);
          if (ico) toRemove.push(ico);
          if (toggle) toRemove.push(toggle);
          if (toRemove.length) g.removeCells(toRemove);
        }
      } catch (e) { void e; }
      try {
        const all = g.getNodes?.() || [];
        all.forEach(p => {
          const data = p.getData ? p.getData() : {};
          if (data && typeof data.overlayTop === 'number') {
            const pb = p.getBBox();
            const pad = 12;
            const overlayTop = data.overlayTop || 0;
            const children = (p.getChildren?.() || []).filter(c => {
              const cid = String(c.id || '');
              return c.isNode && c.isNode() && !cid.endsWith('__hdr') && !cid.endsWith('__sub') && !cid.endsWith('__ico') && !cid.endsWith('__toggle');
            });
            let needW = 240, needH = overlayTop + 80;
            if (children.length > 0) {
              let maxX = 0, maxY = 0;
              children.forEach(c => {
                const cb = c.getBBox();
                maxX = Math.max(maxX, (cb.x - pb.x) + cb.width);
                maxY = Math.max(maxY, (cb.y - pb.y) + cb.height);
              });
              needW = Math.max(240, maxX + pad);
              needH = Math.max(overlayTop + 80, maxY + pad);
            }
            if (Math.abs(needW - pb.width) > 1 || Math.abs(needH - pb.height) > 1) {
              p.resize(needW, needH);
            }
            syncOverlayLayout(p);
          }
        });
      } catch (e) { void e; }
      emitNodeDetailsChange(g);
    });
    graph.on('edge:connected', ({ edge }) => {
      if (isRenderingRef.current) return;
      const has = edge.getLabels?.() && edge.getLabels().length > 0;
      if (!has) {
        const t = window.prompt('设置连线标签(可选):', '');
        if (t && t.trim()) {
          edge.setLabels([{ attrs: { label: { text: t.trim(), fontSize: 11, fontWeight: 600, fill: '#475569' } }, position: { distance: 0.5, offset: 12 } }]);
        }
      }
      const g = graphRef.current;
      emitNodeDetailsChange(g);
    });
    graph.on('edge:removed', () => {
      if (isRenderingRef.current) return;
      const g = graphRef.current;
      emitNodeDetailsChange(g);
    });
    graph.bindKey(['backspace', 'delete'], () => {
      if (isRenderingRef.current) return;
      const cells = graph.getSelectedCells();
      if (cells.length) {
        graph.removeCells(cells);
        const g = graphRef.current;
        emitNodeDetailsChange(g);
      }
    });
    return graph;
  }, []);

  useImperativeHandle(ref, () => ({
    exportImage: () =>
      new Promise((resolve, reject) => {
        const graph = graphRef.current;
        if (!graph) return reject(new Error('Graph not ready'));
        try {
          graph.zoomToFit({ padding: 60, maxScale: 1 });
          setTimeout(() => {
            graph.toPNG((dataUri) => resolve(dataUri), {
              padding: 60,
              backgroundColor: '#ffffff',
              ratio: 4,
              copyStyles: true
            });
          }, 300);
        } catch (e) { reject(e); }
      }),
  }));

  const syncToJson = useCallback(() => {
    const graph = graphRef.current;
    if (!graph || !onNodeDetailsChangeRef.current) return;
    if (isRenderingRef.current) return;
    const nextDetails = graphToNodeDetails(graph, nodeDetailsRef.current, collapsedModulesRef.current);
    let sig = '';
    try {
      sig = JSON.stringify(nextDetails);
    } catch (e) { void e; }
    if (sig) lastInternalSigRef.current = sig;
    isInternalUpdateRef.current = true;
    onNodeDetailsChangeRef.current(nextDetails);
  }, []);

  // ── Init graph once (deferred until container has real dimensions) ────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;

    const create = (w, h) => {
      if (graphRef.current || disposed) return;
      const g = createGraphInstance(w, h);
      if (!g) return;
      graphRef.current = g;
      graphReadyRef.current = true;

      // Render the most recent data: use pendingRenderRef if set (from useEffect that fired
      // before graph was ready), otherwise fall back to nodeDetailsRef.current.
      const detailsToRender = pendingRenderRef.current ?? nodeDetailsRef.current;
      pendingRenderRef.current = null;
      const hasDetails = Array.isArray(detailsToRender)
        ? detailsToRender.length > 0
        : !!(detailsToRender && Object.keys(detailsToRender).length > 0);
      if (hasDetails) {
        isRenderingRef.current = true;
        try {
          // Initialize collapsedModules with all modules expanded
          const initialCollapsed = new Map();
          const modules = Array.isArray(detailsToRender) ? detailsToRender : Object.values(detailsToRender);
          modules.forEach(m => {
            if (m && (m.id || m.name)) {
              const key = m.id || m.name;
              initialCollapsed.set(key, false);
            }
          });
          setCollapsedModules(initialCollapsed);
          savedPositionsRef.current = {}; // fresh plan — start with no preserved positions
          renderGraph(g, detailsToRender, { collapsedMap: initialCollapsed });
        } finally {
          isRenderingRef.current = false;
        }
      }
    };

    const w0 = container.offsetWidth;
    const h0 = container.offsetHeight;
    if (w0 > 0 && h0 > 0) {
      create(w0, h0);
    } else {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            ro.disconnect();
            create(width, height);
            break;
          }
        }
      });
      ro.observe(container);
      return () => { disposed = true; ro.disconnect(); if (graphRef.current) { graphRef.current.dispose(); graphRef.current = null; } };
    }

    return () => {
      disposed = true;
      if (graphRef.current) { graphRef.current.dispose(); graphRef.current = null; }
    };
  }, [createGraphInstance]);

  // ── Re-render when nodeDetails changes (new data → zoom to fit) ──────────
  useEffect(() => {
    let incomingSig = '';
    try {
      incomingSig = JSON.stringify(nodeDetails);
    } catch (e) { void e; }
    const isInternalEcho = !!incomingSig && incomingSig === lastInternalSigRef.current;
    if (isInternalUpdateRef.current) {
      isInternalUpdateRef.current = false;
      if (isInternalEcho) return;
    }
    if (!graphReadyRef.current || !graphRef.current) {
      pendingRenderRef.current = nodeDetails;
      return;
    }
    isRenderingRef.current = true;
    try {
      // If most node ids are new (plan switch), clear saved positions so layout runs fresh.
      const incomingIds = new Set(
        (Array.isArray(nodeDetails) ? nodeDetails : Object.values(nodeDetails || {}))
          .map(m => m?.id || m?.name).filter(Boolean)
      );
      const savedIds = Object.keys(savedPositionsRef.current);
      const overlap = savedIds.filter(id => incomingIds.has(id)).length;
      if (savedIds.length > 0 && overlap < incomingIds.size * 0.5) {
        savedPositionsRef.current = {}; // plan switch — start fresh
      }
      renderGraph(graphRef.current, nodeDetails, { collapsedMap: collapsedModules, skipZoom: false, preservedPositions: savedPositionsRef.current });
      // Save positions after layout so future re-renders preserve them
      graphRef.current.getNodes().forEach(cell => {
        const id = String(cell.id || '');
        if (id.endsWith('__hdr') || id.endsWith('__sub') || id.endsWith('__ico') || id.endsWith('__toggle')) return;
        const pos = cell.getPosition();
        const size = cell.getSize();
        if (pos) savedPositionsRef.current[id] = { x: pos.x, y: pos.y, w: size?.width, h: size?.height };
      });
    } finally {
      isRenderingRef.current = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeDetails, createGraphInstance]);

  // ── Re-render when collapsedModules changes (user toggle → skip zoom) ────
  useEffect(() => {
    if (!graphReadyRef.current || !graphRef.current) return;
    if (collapsedModules.size === 0) return; // skip initial empty map

    const g = graphRef.current;

    // Store current viewport matrix before re-render (captures both scale and translate)
    const savedMatrix = g.transform.getMatrix();

    isRenderingRef.current = true;
    try {
      renderGraph(g, nodeDetailsRef.current, { collapsedMap: collapsedModules, skipZoom: true, preservedPositions: savedPositionsRef.current });
    } finally {
      isRenderingRef.current = false;
    }

    // Restore viewport by setting the exact matrix back — avoids any scale/translate side-effects
    g.transform.setMatrix(savedMatrix);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collapsedModules]);

  // ── Keyboard: delete/backspace remove selected ────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onKeyDown = (e) => {
      if (!graphRef.current) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        const cells = graphRef.current.getSelectedCells?.() || [];
        if (cells.length) {
          e.preventDefault();
          graphRef.current.removeCells(cells);
          syncToJson();
        }
      }
    };
    const onMouseDown = () => { container.focus?.(); };
    container.addEventListener('keydown', onKeyDown);
    container.addEventListener('mousedown', onMouseDown);
    return () => {
      container.removeEventListener('keydown', onKeyDown);
      container.removeEventListener('mousedown', onMouseDown);
    };
  }, [syncToJson]);

  // ── Toolbar: add node helper ──────────────────────────────────────────────
  const handleAddNode = useCallback(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const name = window.prompt('模块名称:');
    if (!name?.trim()) return;
    const brief = window.prompt('模块简介(可选):') || '';
    const id = `mod-${Date.now()}`;
    const x = 120 + Math.random() * 180;
    const y = 120 + Math.random() * 120;
    const width = 240;
    const baseBodyH = 80;
    const innerW = Math.max(10, width - 24);
    const titleLines = wrapTextToLines(name.trim(), innerW, 17, 'bold');
    const descLines = wrapTextToLines((brief || '').trim(), innerW, 15, 'normal');
    const gap = 12;
    const titleH = Math.max(20, titleLines.length * 18);
    const descH = descLines.length > 0 ? Math.max(14, descLines.length * 16) : 0;
    const overlayTop = 12 + titleH + (descH ? gap + descH : 0);
    const height = baseBodyH + overlayTop;
    const group = graph.addNode({
      id,
      shape: 'rect',
      x, y,
      width, height,
      zIndex: 1,
      attrs: {
        body: { fill: 'rgba(33,124,243,0.05)', stroke: '#217cf3', strokeWidth: 2, strokeDasharray: '6 3', rx: 8, ry: 8 },
        label: { text: '', fontSize: 17, fill: '#217cf3', textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5 },
      },
      ports: {
        groups: {
          top: { position: 'top', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
          bottom: { position: 'bottom', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
          left: { position: 'left', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
          right: { position: 'right', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
        },
        items: [
          { id: `${id}-top`, group: 'top' }, { id: `${id}-bottom`, group: 'bottom' },
          { id: `${id}-left`, group: 'left' }, { id: `${id}-right`, group: 'right' },
        ],
      },
    });
    try { group.setData && group.setData({ overlayTop }); } catch (e) { void e; }
    const titleNode = graph.addNode({
      id: `${id}__hdr`,
      shape: 'rect',
      x: x + 12,
      y: y + 12,
      width: innerW,
      height: titleH,
      zIndex: 5,
      attrs: {
        body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
        label: { text: titleLines.join('\n'), fontSize: 17, fontWeight: 'bold', fill: '#217cf3', textAnchor: 'middle', textVerticalAnchor: 'top', refX: 0.5, refY: 0, pointerEvents: 'none' },
      },
      ports: { groups: {}, items: [] },
    });
    const descNode = graph.addNode({
      id: `${id}__sub`,
      shape: 'rect',
      x: x + 12,
      y: y + 12 + titleH + gap,
      width: innerW,
      height: descH || 14,
      zIndex: 5,
      attrs: {
        body: { fill: 'transparent', stroke: 'transparent', pointerEvents: 'none' },
        label: { text: descLines.join('\n'), fontSize: 15, fill: '#666', textAnchor: 'middle', textVerticalAnchor: 'top', refX: 0.5, refY: 0, pointerEvents: 'none' },
      },
      ports: { groups: {}, items: [] },
    });
    group.addChild(titleNode);
    group.addChild(descNode);
    syncToJson();
  }, [syncToJson]);

  const handleAddSubNode = useCallback(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const pad = 12;
    const getOverlayTop = (parent) => {
      const data = parent.getData ? parent.getData() : {};
      return (data && data.overlayTop) || 0;
    };
    const expandParentToFit = (parent, child) => {
      try {
        const pb = parent.getBBox();
        const overlayTop = getOverlayTop(parent);
        const cb = child.getBBox();
        const requiredW = (cb.x - pb.x) + cb.width + pad;
        const requiredH = (cb.y - pb.y) + cb.height + pad;
        const minW = Math.max(pb.width, requiredW);
        const minH = Math.max(pb.height, Math.max(requiredH, overlayTop + 80));
        if (minW > pb.width || minH > pb.height) {
          parent.resize(minW, minH);
        }
      } catch (e) { void e; }
    };
    const placeChildInGrid = (parent, childIndex) => {
      const pb = parent.getBBox();
      const overlayTop = getOverlayTop(parent);
      const contentW = Math.max(1, pb.width - 2 * pad);
      const cellW = NODE_W + pad;
      const cols = Math.max(1, Math.floor(contentW / cellW));
      const row = Math.floor(childIndex / cols);
      const col = childIndex % cols;
      const x = pb.x + pad + col * (NODE_W + pad);
      const y = pb.y + overlayTop + pad + row * (NODE_H + pad);
      return { x, y };
    };
    const recomputeGroupSize = (parent) => {
      try {
        const overlayTop = getOverlayTop(parent);
        const pb = parent.getBBox();
        const children = (parent.getChildren?.() || []).filter(c => {
          const cid = String(c.id || '');
          return c.isNode && c.isNode() && !cid.endsWith('__hdr') && !cid.endsWith('__sub');
        });
        if (children.length === 0) {
          const minW = 240;
          const minH = overlayTop + 80;
          if (pb.width !== minW || pb.height !== minH) parent.resize(minW, minH);
          return;
        }
        let maxX = 0, maxY = 0;
        children.forEach(c => {
          const cb = c.getBBox();
          maxX = Math.max(maxX, (cb.x - pb.x) + cb.width);
          maxY = Math.max(maxY, (cb.y - pb.y) + cb.height);
        });
        const needW = maxX + pad;
        const needH = Math.max(overlayTop + 80, maxY + pad);
        const newW = Math.max(needW, 240);
        const newH = Math.max(needH, overlayTop + 80);
        if (Math.abs(newW - pb.width) > 1 || Math.abs(newH - pb.height) > 1) {
          parent.resize(newW, newH);
        }
      } catch (e) { void e; }
    };
    const sel = (graph.getSelectedCells?.() || []).filter(c => c.isNode && c.isNode());
    let parentId = null;
    if (sel.length === 1) {
      const id = String(sel[0].id);
      const hdr = graph.getCellById(`${id}__hdr`);
      if (hdr) parentId = id;
    }
    if (!parentId) {
      const input = window.prompt('输入要加入的模块ID:');
      if (!input || !input.trim()) return;
      const exists = graph.getCellById(input.trim());
      const hdr = graph.getCellById(`${input.trim()}__hdr`);
      if (!exists || !hdr) return;
      parentId = input.trim();
    }
    const parent = graph.getCellById(parentId);
    if (!parent) return;
    const name = window.prompt('子节点名称:');
    if (!name || !name.trim()) return;
    const id = `sub-${Date.now()}`;
    const existingChildren = (parent.getChildren?.() || []).filter(c => {
      const cid = String(c.id || '');
      return c.isNode && c.isNode() && !cid.endsWith('__hdr') && !cid.endsWith('__sub');
    });
    const idx = existingChildren.length;
    const { x, y } = placeChildInGrid(parent, idx);
    const node = graph.addNode({
      id,
      shape: 'rect',
      x, y,
      width: NODE_W, height: NODE_H,
      attrs: {
        body: { fill: '#ffffff', stroke: '#666', strokeWidth: 1.5, rx: 4, ry: 4 },
        label: { text: name.trim(), fontSize: 16, fill: '#333', textAnchor: 'middle', textVerticalAnchor: 'middle', refX: 0.5, refY: 0.5 },
      },
      ports: {
        groups: {
          top: { position: 'top', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
          bottom: { position: 'bottom', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
          left: { position: 'left', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
          right: { position: 'right', attrs: { circle: { r: 4, magnet: true, stroke: '#217cf3', fill: '#fff', strokeWidth: 1.5 } } },
        },
        items: [
          { id: `${id}-top`, group: 'top' }, { id: `${id}-bottom`, group: 'bottom' },
          { id: `${id}-left`, group: 'left' }, { id: `${id}-right`, group: 'right' },
        ],
      },
    });
    parent.addChild(node);
    expandParentToFit(parent, node);
    recomputeGroupSize(parent);
    syncToJson();
  }, [syncToJson]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} tabIndex={0} />
      <div style={{ position: 'absolute', top: 8, right: 8, display: 'flex', gap: 6, zIndex: 10 }}>
        <button title="放大" onClick={() => graphRef.current?.zoom(0.1)} style={btnStyle}>+</button>
        <button title="缩小" onClick={() => graphRef.current?.zoom(-0.1)} style={btnStyle}>−</button>
        <button title="适应画布" onClick={() => graphRef.current?.zoomToFit({ padding: 40, maxScale: 1.2 })} style={btnStyle}>⊡</button>
        <button title="添加节点" onClick={handleAddNode} style={{ ...btnStyle, backgroundColor: '#217cf3', color: '#fff', borderColor: '#217cf3' }}>+ 模块</button>
        <button title="添加子节点" onClick={handleAddSubNode} style={{ ...btnStyle, backgroundColor: '#217cf3', color: '#fff', borderColor: '#217cf3' }}>+ 子节点</button>
      </div>
      <div style={{ position: 'absolute', bottom: 8, left: 8, fontSize: 11, color: '#aaa', zIndex: 10, pointerEvents: 'none' }}>
        双击节点重命名 · 双击连线编辑标签 · 拖拽端口连线 · Delete 删除选中 · 点击 ▶/▼ 展开/折叠模块
      </div>
    </div>
  );
});

const btnStyle = {
  padding: '4px 10px', fontSize: 13, background: '#fff',
  border: '1px solid #ccc', borderRadius: 4, cursor: 'pointer', lineHeight: 1.4,
};

export default X6FlowViewer;
