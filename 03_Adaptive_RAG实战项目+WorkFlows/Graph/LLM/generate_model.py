from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import config_data as config
import prompt as pt


#将生成模型进行封装成一个类
class Generate_Chain():
    def __init__(self):
        #通过openai的接口调用千问模型
        self.llm=config.llm
        #大模型根据用户问题和参考资料生成答案的系统提示词
        self.prompt=pt.generate_system_prompt

        #创建生成链对象
        self.chain=self.get_chain()
    def get_chain(self):
        #创建提词模板
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            ("user", "用户提问：{question}\n\n参考资料：{context}\n")
        ])
        # 创建发生链
        chain = prompt_template | self.llm
        return chain



