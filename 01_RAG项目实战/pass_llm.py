from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from history_store import get_history
from vector_retrieve import VectorRetrieve
import config_data as config

#把拼接好的提示词打印出来
def print_prompt(prompt):
    print("="*50)
    print(prompt.to_string())
    print("="*50)
    return prompt

# 从 RunnableWithMessageHistory 传入的 input 中分离出当前问题和历史消息
# 新版 langchain 1.x: input 是 [历史消息..., 当前HumanMessage]，没有单独的 history 键
def _split_current_and_history(val):
    if isinstance(val, list):
        if not val:
            return "", []
        current = val[-1]  # 最后一条是当前用户消息
        current_text = current.content if hasattr(current, "content") else str(current)
        history = val[:-1]  # 前面的都是历史对话
        return current_text, history
    if hasattr(val, "content"):
        return val.content, []
    return str(val), []

#创建一个从检索结果到传入llm中的类
class ragservice(object):
    #定义类的的对象
    def __init__(self):
        #实例化向量库和检索
        self.vector=VectorRetrieve(DashScopeEmbeddings(model=config.EMBEDDING_MODEL))

        #创建大语言模型对象
        self.llm=ChatOpenAI(
            model=config.CHAT_MODEL,
            api_key=config.ALIBL_API_KEY,
            base_url=config.ALIBL_BASE_URL,
            streaming=True,
        )

        #创建提示词模板
        self.prompt_template=ChatPromptTemplate.from_messages(
            [("system","以我提供的已知参考资料为主，简洁和专业的回答用户的问题，\n参考资料：{context}"),
             ("system","并且我提供了用户的对话记录以下："),
             MessagesPlaceholder("history"),
             ("user","请回答用户提问：{input}")
            ]
        )

        #获取最终的LCEL执行链
        self.chain=self.get_chain()

    def get_chain(self):
        #组装RAG执行链
        #获取检索器
        retriever=self.vector.get_retriever()
        #自定义格式化函数：将检索出来的文档列表拼接成长字符串
        def format_docment(docs:List[Document]):
            if not docs:
                return "未找到参考资料"
            formated_doc=""
            for doc in docs:
                formated_doc+=f"文档片段：{doc.page_content}\n文档的元数据：{doc.metadata}\n\n"
            return formated_doc

        #检索只需要当前用户问题文本（取消息列表最后一条）
        def format_text(input):
            current_text, _ = _split_current_and_history(input["input"])
            return current_text


        #将并行分支的输出整理为提示词模板所需的字段
        def format_prompt(value):
            result={}
            # value["input"] 是 RunnablePassthrough 的输出 = {"input": [消息列表]}
            # value["input"]["input"] 才是真正的消息列表
            current_text, history = _split_current_and_history(value["input"]["input"])
            result["input"] = current_text
            result["context"] = value["context"]
            result["history"] = history
            return result

        #构造LCET链
        chain=({
            "input":RunnablePassthrough(),
            "context":RunnableLambda(format_text)|retriever|RunnableLambda(format_docment)}
            |RunnableLambda(format_prompt)
            |self.prompt_template
            |print_prompt
            |self.llm
            |StrOutputParser()
        )
        #给链装上会话记忆
        coversion_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            message_history_key="history",
        )
        return coversion_chain


if __name__=="__main__":
    rag=ragservice()
    # # 第一轮：告知体重
    # result0=rag.chain.invoke(input={"input":"我体重180，尺码推荐"}, config={ "configurable": {"session_id": "user_001" }})
    # print("第一轮回答:", result0)
    # 第二轮：基于历史追问
    result1=rag.chain.invoke(input={"input":"你知道我的身高吗"},config={ "configurable": {"session_id": "user_001" }})
    print("第三轮回答:", result1)
