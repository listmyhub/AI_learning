
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import prompt
import config_data as config
#通过大模型来判断用户的需求是否是走网络搜索还是RAG检索
#封装成一个类
class JudgerRoute(BaseModel):
    """根据用户的问题来判断路线"""
    datasource:Literal["retriever","web_search"]=Field(...,description="根据用户的问题选择是进行RAG检索还是网络搜素")

llm=config.llm
judge_system_prompt=prompt.judge_system_prompt
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",judge_system_prompt),
        ("user","用户提问{question}"),
    ]
)
route_judge_structure=llm.with_structured_output(JudgerRoute)
route_chain=prompt_template|route_judge_structure


# print(route_chain.invoke({"question":"我想买一件红色的衣服"}))
