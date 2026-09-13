from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  PromptTemplate

from RAG.split_vector import VectorService
from process_tools.prompt_process_tools import load_rag_prompt
from config import model as md

#为了方便调试，将组装后的提示词打印出来
def print_prompt(prompt):
    print("-"*50)
    print(prompt.to_string())
    print("-"*50)
    return prompt


#将检索-》拼装-》提示词-》模型-》输出解释包装成rag服务
class RagService:
    def __init__(self):
        #创建向量库对象
        self.vector_service=VectorService()
        #创建检索器对象
        self.retriever=self.vector_service.get_retriever()
        #创建提示词对象
        self.prompt_text=load_rag_prompt()
        self.prompt_template=PromptTemplate.from_template(self.prompt_text)
        #创建模型对象
        self.model=md.LLM
        #获取链对象
        self.chain=self.get_chain()

    def get_chain(self):
        chain=self.prompt_template|print_prompt|self.model|StrOutputParser()
        return chain

    #检索文档
    def retriever_docs(self,query:str):
        return self.retriever.invoke(query)

    #拼接检索出来的文档成字符串
    def compact_docs(self,docs:list[Document]):
        compact_docs=""
        for idx ,doc in enumerate(docs):
            compact_docs+=f"\n【参考资料{idx+1}】:\n参考资料：{doc.page_content}\n参考资料元数据：{doc.metadata}\n\n"
        return compact_docs
    #将用户问题和参考资料全部送入链中
    def rag_service(self,query:str):
        context=self.compact_docs(self.retriever_docs(query))
        return self.chain.invoke({"input":query,"context":context})


if __name__=="__main__":
    rag=RagService()
    result=rag.rag_service("小户型适合的机器人")
    print(result)



