import openai
import time
import os
import json
from PyPDF2 import PdfReader  # 添加PDF读取库

# ================= 配置区域 =================

# OpenAI API配置（中转API）
OPENAI_API_KEY = "sk-HOhPpbrbpl2pEa9UU6BGea6I2O93cLYX1eS342bLaMypwvlc"  # 请替换为你的OpenAI API密钥
OPENAI_BASE_URL = "https://poloai.top/v1"  # 中转API地址


# 配置 OpenAI SDK
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
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
            model="gemini-2.5-pro",  # 使用OpenAI支持的模型
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": pdf_content}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用OpenAI API时出错: {e}")
        return None

def save_json_response(response_text, pdf_filename):
    """
    从LLM响应中提取JSON并保存到文件
    """
    try:
        # 创建目标目录如果不存在
        json_output_dir = "./case_llm_education/json_files"
        os.makedirs(json_output_dir, exist_ok=True)
        
        # 提取JSON部分（假设响应是纯JSON或包含JSON）
        # 这里简化处理，实际可能需要更复杂的解析
        json_data = None
        
        # 尝试直接解析整个响应
        try:
            json_data = json.loads(response_text)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试从文本中提取JSON部分
            # 寻找第一个 '{' 和最后一个 '}' 之间的内容
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                try:
                    json_data = json.loads(json_str)
                except json.JSONDecodeError:
                    print("无法从响应中提取有效的JSON")
                    return False
        
        if json_data:
            # 生成输出文件名（去除.pdf扩展名，添加.json）
            base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
            output_file = os.path.join(json_output_dir, f"{base_name}.json")
            
            # 保存JSON到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"JSON结果已保存到: {output_file}")
            return True
        else:
            print("未能提取到有效的JSON数据")
            return False
            
    except Exception as e:
        print(f"保存JSON时出错: {e}")
        return False

# ================= 主程序 =================

if __name__ == "__main__":
    # 加载提示词
    image_extactor_prompt = load_prompt_from_file()
    
    # 替换成你本地的 PDF 路径
    pdf_folder = "./case_llm_education"
    
    # 确保输出目录存在
    json_output_dir = "./case_llm_education/json_files"
    os.makedirs(json_output_dir, exist_ok=True)
    
    # 检查PDF目录是否存在
    if not os.path.exists(pdf_folder):
        print(f"PDF目录不存在: {pdf_folder}")
        exit(1)
    
    processed_count = 0
    error_count = 0
    
    file_name = "Court to conversation_tactical badminton analysis via computer vision and RAG-enhanced LLM.pdf"
    file_path = os.path.join(pdf_folder, file_name)
    answer = chat_with_pdf_openai(file_path, image_extactor_prompt)
    if answer:
        save_json_response(answer, file_name)

    # # 遍历目录中的所有PDF文件
    # for filename in os.listdir(pdf_folder):
    #     if filename.endswith(".pdf"):
    #         pdf_path = os.path.join(pdf_folder, filename)
    #         print(f"正在处理: {pdf_path}")
            
    #         try:
    #             # 直接使用OpenAI中转API处理PDF内容
    #             print("正在调用OpenAI中转API...")
    #             answer = chat_with_pdf_openai(pdf_path, image_extactor_prompt)
                
    #             if answer:
    #                 print("\n" + "="*20 + " LLM 回答 " + "="*20 + "\n")
    #                 print(answer[:500] + "..." if len(answer) > 500 else answer)  # 只显示前500字符
                    
    #                 # 保存JSON响应
    #                 if save_json_response(answer, filename):
    #                     print(f"成功处理: {filename}")
    #                     processed_count += 1
    #                 else:
    #                     print(f"处理失败: {filename}")
    #                     error_count += 1
    #             else:
    #                 print(f"处理失败: {filename}")
    #                 error_count += 1
                
    #         except Exception as e:
    #             print(f"处理 {filename} 时发生错误: {e}")
    #             error_count += 1
    #             continue
    
    # print(f"\n处理完成! 成功处理 {processed_count} 个文件，{error_count} 个文件出错。")
