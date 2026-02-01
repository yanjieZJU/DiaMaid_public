import openai
import time
import httpx
import os
import json
import sys
from PyPDF2 import PdfReader  # 添加PDF读取库
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
# ================= 配置区域 =================

# OpenAI API配置（中转API）
http_client = httpx.Client(proxy=None, trust_env=False)


# 配置 OpenAI SDK
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    http_client=http_client
)

def load_prompt_from_file():
    """
    从外部文件加载提示词
    """
    prompt_file_path = "./offline_processor/diagram2code_prompt.md"
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_output_template():
    """
    从外部文件加载输出模板
    """
    template_file_path = "./offline_processor/output_template.json"
    
    # 检查模板文件是否存在
    if not os.path.exists(template_file_path):
        print(f"警告: 输出模板文件 {template_file_path} 不存在")
        return None
    
    # 从文件读取模板
    with open(template_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # 移除代理对字符(surrogates)，防止API调用时的编码错误
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        return text
    except Exception as e:
        print(f"提取PDF文本时出错: {e}")
        return ""

def chat_with_pdf_openai(pdf_file_path, query):
    """
    使用OpenAI中转API发送文件内容和问题给模型
    """
    # 读取PDF文件内容
    pdf_content = extract_text_from_pdf(pdf_file_path)
    
    # 如果PDF内容为空，给出警告
    if not pdf_content:
        print("警告: 无法提取PDF内容或PDF为空")
    
    # 加载输出模板
    template = load_output_template()
    
    # 构建完整的提示词，包含模板信息和PDF内容
    full_prompt = f"{query}\n\nFinal Output Structure (in JSON format):{json.dumps(template, ensure_ascii=False, indent=2)}"
    
    # 使用OpenAI API生成回答
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash", 
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": pdf_content}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用OpenAI API时出错: {e}")
        return None

def save_json_response(json_output_dir, response_text, pdf_filename):
    """
    从LLM响应中提取JSON并保存到文件
    包含针对Mermaid代码中未转义引号的自动修复逻辑
    """
    import re
    
    def repair_mermaid_quotes(text):
        """修复Mermaid节点定义中常见的未转义引号问题: Node["Label"] -> Node[\"Label\"]"""
        # 排除已知的JSON数组字段名
        protected_keys = ["modules", "subnodes", 
                         "incoming", "outgoing", "text_ref", "connects_to", "originates_from",
                         "sources", "targets", "files", "paths", "includes", "excludes", "TargetLintErrorIds", "ImagePaths"]
        
        def replace_callback(match):
            key = match.group(1)
            content = match.group(2)
            if key in protected_keys: 
                return match.group(0) 
            return f'{key}[\\"{content}\\"]'

        pattern = r'(\w+)\s*\["(.*?)"\]'
        try:
            return re.sub(pattern, replace_callback, text)
        except Exception:
            return text

    try:
        os.makedirs(json_output_dir, exist_ok=True)
        json_data = None
        
        # 提取JSON文本候选
        candidate_text = response_text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            candidate_text = json_match.group(1)
        else:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                candidate_text = response_text[start:end]
        
        # 尝试解析
        try:
            json_data = json.loads(candidate_text, strict=False)
        except json.JSONDecodeError:
            print("JSON直接解析失败，尝试修复 Mermaid 引号问题...")
            repaired_text = repair_mermaid_quotes(candidate_text)
            try:
                json_data = json.loads(repaired_text, strict=False)
                print("修复后解析成功！")
            except json.JSONDecodeError as e:
                print(f"无法从响应中提取有效的JSON: {e}")
                # print("Debug - Repaired Text Fragment:", repaired_text[:200])

        if json_data:
            base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
            output_file = os.path.join(json_output_dir, f"{base_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"JSON结果已保存到: {output_file}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"保存JSON时出错: {e}")
        return False

# ================= 主程序 =================

if __name__ == "__main__":

    rnum = 30

    # 加载提示词
    image_extactor_prompt = load_prompt_from_file()
    
    # 替换成你本地的 PDF 路径
    pdf_folder = "./case_llm_artifact/original_articles"
    
    # 确保输出目录存在
    json_output_dir = "./case_llm_artifact/json_files"
    os.makedirs(json_output_dir, exist_ok=True)
    
    # 检查PDF目录是否存在
    if not os.path.exists(pdf_folder):
        print(f"PDF目录不存在: {pdf_folder}")
        exit(1)
    
    processed_count = 0
    error_count = 0
    
    # file_name = "3613904.3641915.pdf"
    # file_path = os.path.join(pdf_folder, file_name)
    # answer = chat_with_pdf_openai(file_path, image_extactor_prompt)
    # if answer:
    #     save_json_response(json_output_dir, answer, file_name)
    
    import random
    
    # 获取所有PDF文件
    all_pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    # 随机采样
    if len(all_pdf_files) > rnum:
        pdf_files_to_process = random.sample(all_pdf_files, rnum)
        print(f"Total PDFs found: {len(all_pdf_files)}. Sampled {rnum} for processing.")
    else:
        pdf_files_to_process = all_pdf_files
        print(f"Total PDFs found: {len(all_pdf_files)}. Processing all.")

    # for filename in os.listdir(pdf_folder):
    
    # 遍历选中的PDF文件
    for filename in pdf_files_to_process:

        if filename.endswith(".pdf"):
            # Check if JSON file already exists
            base_name = os.path.splitext(filename)[0]
            expected_json_path = os.path.join(json_output_dir, f"{base_name}.json")
            
            if os.path.exists(expected_json_path):
                print(f"跳过 {filename}: JSON 结果已存在于 {expected_json_path}")
                continue

            pdf_path = os.path.join(pdf_folder, filename)
            print(f"正在处理: {pdf_path}")
            
            try:
                # 直接使用OpenAI中转API处理PDF内容
                print("正在调用OpenAI中转API...")
                answer = chat_with_pdf_openai(pdf_path, image_extactor_prompt)
                
                if answer:
                    print("\n" + "="*20 + " LLM 回答 " + "="*20 + "\n")
                    print(answer[:500] + "..." if len(answer) > 500 else answer)  # 只显示前500字符
                    
                    # 保存JSON响应
                    if save_json_response(json_output_dir, answer, filename):
                        print(f"成功处理: {filename}")
                        processed_count += 1
                    else:
                        print(f"处理失败: {filename}")
                        error_count += 1
                else:
                    print(f"处理失败: {filename}")
                    error_count += 1
                
            except Exception as e:
                print(f"处理 {filename} 时发生错误: {e}")
                error_count += 1
                continue
    
    print(f"\n处理完成! 成功处理 {processed_count} 个文件，{error_count} 个文件出错。")
