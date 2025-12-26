import json

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


def process_json_to_mermaid(json_path):
      
      with open(json_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
      
      mermaid_code = data['full_code']
      
      cleaned_code = clean_mermaid_code(mermaid_code)
      
      return cleaned_code

if __name__ == '__main__':
    json_path = 'case_llm_education\\json_files\\LLM-based interactive coding education via predictive query management and student-centered fine-tuning.json'
    mermaid_code = process_json_to_mermaid(json_path)
    print(mermaid_code)