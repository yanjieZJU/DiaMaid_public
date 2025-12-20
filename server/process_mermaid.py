import json
import re
import os

def clean_mermaid_code(code):
    """
    清理Mermaid代码中的转义字符
    """
    # 替换转义的换行符
    code = code.replace('\\n', '\n')
    
    # 替换转义的双引号
    code = code.replace('\\"', '"')
    
    # 移除可能存在的首尾引号（如果有的话）
    if code.startswith('"') and code.endswith('"'):
        code = code[1:-1]
    
    # 移除多余的空白字符
    code = code.strip()
    
    return code

def process_json_to_mermaid(file_path):
    """
    处理Feed-o-meter.json文件中的Mermaid代码
    """
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取Mermaid代码
    mermaid_code = data['diagram']['overview']['full_code']
    # mermaid_code = data['mermaid_code']
    
    # 清理代码
    cleaned_code = clean_mermaid_code(mermaid_code)
    
    # 更新JSON数据
    # data['diagram']['overview']['code'] = cleaned_code
    
    # 保存回文件（可选）
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)
    
    return cleaned_code


if __name__ == "__main__":
    # 测试脚本功能
    code = process_json_to_mermaid(".\case_llm_education\json_files\ELF_Educational LLM framework of improving and evaluating AI-generated content for classroom teaching.json")
    print("处理后的Mermaid代码:")
    print(code)
    
    # 启动API服务（可选）
    # import uvicorn
    # app = create_api_endpoint()
    # uvicorn.run(app, host="0.0.0.0", port=8001)
