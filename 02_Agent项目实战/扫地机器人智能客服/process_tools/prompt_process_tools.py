import os

from config import PATH as pt
#加载rag提示词
def load_rag_prompt():
    rag_prompt_path=os.path.join(pt.BASE_DIR,"prompt","rag_summarize.txt")
    if not os.path.isfile(rag_prompt_path):
        print(f"【提示词加载】{rag_prompt_path}不是文件，加载失败")
        return
    if not os.path.exists(rag_prompt_path):
        print(f"【提示词加载{rag_prompt_path}文件不存在，加载失败")
        return
    with open(rag_prompt_path,"r") as f:
        return f.read()

#加载agent需要的提示词，分两种情况，如果需要生成报告，那么就调用report_prompts,如果不需要，就调用main_prompts
def load_system_prompts():
    system_prompt_path=os.path.join(pt.BASE_DIR,"prompt","main_prompt.txt")
    if not os.path.isfile(system_prompt_path):
        print(f"【提示词生成】{system_prompt_path}不是文件，生成失败")
        return ""
    try:
        with open(system_prompt_path,"r") as f:
            return f.read()
    except Exception as e:
        print(f"【提示词生成】{system_prompt_path}文件生成失败")
        return ""

def load_report_prompts():
    report_prompt_path=os.path.join(pt.BASE_DIR,"prompt","report_prompt.txt")
    if not os.path.isfile(report_prompt_path):
        print(f"【提示词生成】{report_prompt_path}不是文件，生成失败")
        return ""
    try:
        with open(report_prompt_path,"r") as f:
            return f.read()
    except Exception as e:
        print(f"【提示词生成】{report_prompt_path}文件生成失败")
        return ""

if __name__=="__main__":
    print(load_rag_prompt())
    print(load_system_prompts())
    print(load_report_prompts())