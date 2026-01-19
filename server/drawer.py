import json
from graphviz import Digraph

# ==========================================
# 1. 科学作图风格配置 (The "Style System")
# ==========================================
STYLE_CONFIG = {
    "PROCESSOR": {
        "shape": "box",
        "style": "filled,rounded", # 圆角矩形，看起来更柔和
        "fillcolor": "#E3F2FD",     # 浅蓝 (Material Blue 50)
        "color": "#1565C0",         # 深蓝边框
        "icon": "⚙️"
    },
    "INPUT": {
        "shape": "parallelogram",   # 平行四边形表示 I/O
        "style": "filled",
        "fillcolor": "#E8F5E9",     # 浅绿
        "color": "#2E7D32",
        "icon": "📥"
    },
    "OUTPUT": {
        "shape": "parallelogram",
        "style": "filled",
        "fillcolor": "#E8F5E9",
        "color": "#2E7D32",
        "icon": "📤"
    },
    "DATABASE": {
        "shape": "cylinder",        # 圆柱体
        "style": "filled",
        "fillcolor": "#FFF3E0",     # 浅橙
        "color": "#EF6C00",
        "icon": "🗄️"
    },
    "GATE": {
        "shape": "diamond",         # 菱形
        "style": "filled",
        "fillcolor": "#F3E5F5",     # 浅紫
        "color": "#7B1FA2",
        "icon": "🔀"
    },
    "DEFAULT": {
        "shape": "box",
        "style": "filled",
        "fillcolor": "#F5F5F5",     # 浅灰
        "color": "#616161",
        "icon": "🧩"
    }
}
def clean_and_infer_data(json_plan):
    """
    1. 去除重复的模块 (基于 ID)。
    2. 自动推断模块类型 (如果没有显式提供)。
    """
    clean_modules = {}
    seen_ids = set()
    
    raw_modules = json_plan.get('module_details', [])
    
    # --- 步骤 A: 去重与提取 ---
    for item in raw_modules:
        # 兼容两种格式：list of dicts 或 dict of dicts
        # 假设格式是 [{"ID": {...}}, {"ID2": {...}}]
        for mod_id, details in item.items():
            if mod_id in seen_ids:
                print(f"⚠️ Warning: Detected duplicate module ID '{mod_id}'. Skipping duplicate.")
                continue
            
            seen_ids.add(mod_id)
            
            # --- 步骤 B: 自动推断类型 (Heuristics) ---
            # 如果 json 里没有 'type' 字段，我们通过名称来猜测，以应用样式
            name_lower = details.get('name', '').lower()
            role_type = "DEFAULT"
            
            if "database" in name_lower or "knowledge" in name_lower or "store" in name_lower:
                role_type = "DATABASE"
            elif "input" in name_lower or "user" in name_lower or "speech to" in name_lower:
                role_type = "INPUT"
            elif "output" in name_lower or "synthesis" in name_lower or "response" in name_lower:
                role_type = "OUTPUT"
            elif "router" in name_lower or "decision" in name_lower or "gate" in name_lower:
                role_type = "GATE"
            elif "llm" in name_lower or "process" in name_lower:
                role_type = "PROCESSOR"
            
            details['_inferred_type'] = role_type # 存入临时字段
            clean_modules[mod_id] = details

    return clean_modules

# ==========================================
# 3. 渲染器 (The "Renderer")
# ==========================================
def render_scientific_chart(json_data, filename="scientific_arch"):
    # 1. 清洗数据
    modules = clean_and_infer_data(json_data)
    
    # 2. 初始化 Graphviz
    dot = Digraph(comment='Scientific Architecture', format='svg')
    
    # --- 全局属性 (符合科学排版) ---
    dot.attr(rankdir='LR')           # Left to Right 布局
    dot.attr(splines='ortho')        # 折线连接，工程感更强
    dot.attr(nodesep='0.6')          # 增加节点间距，避免拥挤
    dot.attr(ranksep='0.8')          # 增加层级间距
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9', color='#455A64', arrowsize='0.8')

    # 3. 绘制节点
    for mod_id, data in modules.items():
        role = data.get('_inferred_type', 'DEFAULT')
        style = STYLE_CONFIG.get(role, STYLE_CONFIG['DEFAULT'])
        
        # 构建 Label：包含 Icon + Title + Tech Detail
        # 使用 HTML Label 进行精细排版
        # 注意：对于 Diamond/Cylinder 形状，HTML label 可能会溢出，这里做简化处理
        
        icon = style['icon']
        title = data.get('name', mod_id)
        tech = data.get('technical_methods', '')
        # 截断过长的技术描述
        if len(tech) > 25: tech = tech[:22] + "..."

        # 核心技巧：Label 使用 HTML 子集
        label_html = f'''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><FONT POINT-SIZE="14">{icon}</FONT></TD></TR>
            <TR><TD><B>{title}</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="8" COLOR="#555555">{tech}</FONT></TD></TR>
        </TABLE>
        >'''

        dot.node(mod_id, 
                 label=label_html,
                 shape=style['shape'],
                 style=style['style'],
                 fillcolor=style['fillcolor'],
                 color=style['color'],
                 penwidth='1.5')

        # 处理子节点 (Sub-nodes) - 如果有的话
        # 这里为了保持科学图表整洁，暂时略过子节点内部逻辑，或者可以在此处递归

    # 4. 绘制连接
    for mod_id, data in modules.items():
        if 'ports' in data and 'outgoing' in data['ports']:
            for edge in data['ports']['outgoing']:
                target = edge['target_mod']
                label = edge.get('edge_label', '')
                
                # 校验：确保目标节点真的存在（防止 Dead Links）
                if target in modules:
                    dot.edge(mod_id, target, label=f"  {label}  ") # 加空格增加 padding
                else:
                    print(f"⚠️ Warning: Node '{mod_id}' tries to connect to missing node '{target}'. Edge dropped.")

    # 5. 输出
    print(f"Generating diagram: {filename}.svg")
    svg_source = dot.pipe(format='svg').decode('utf-8')
    
    return svg_source

# ==========================================
# 测试数据 (模拟包含重复错误的 Plan)
# ==========================================
test_plan_with_errors = {
    "module_details": [
        {"SpeechToText": {"name": "Speech Input", "technical_methods": "Whisper API", "ports": {"outgoing": [{"target_mod": "LLM_Core", "edge_label": "Text"}]}}},
        {"LLM_Core": {"name": "LLM Processor", "technical_methods": "GPT-4", "ports": {"outgoing": [{"target_mod": "Database_1", "edge_label": "Query"}]}}},
        {"Database_1": {"name": "Vector DB", "technical_methods": "Pinecone", "ports": {"outgoing": [{"target_mod": "LLM_Core", "edge_label": "Context"}]}}},
        {"Router": {"name": "Intent Gate", "technical_methods": "Classifier", "ports": {"outgoing": []}}} # 这是一个 Gate 类型的测试
    ]
}

# 运行
render_scientific_chart(test_plan_with_errors)