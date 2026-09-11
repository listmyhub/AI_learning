#@tool装饰器定义工具
#create_agent 创建智能体：模型+工具列表+系统提示词
#agent.invoke 运行智能体，StrOutPaser从消息中提取纯文本
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import config


#1.定义一个很简单的工具，查询天气工具
@tool("get_weather",description="查询天气")
def get_weather(query:str):
    """返回今天的天气情况"""
    return "晴天"

#2.创建智能体
agent=create_agent(
    model=ChatOpenAI(model=config.CHAT_MODEL,
                     api_key=config.ALIBL_API_KEY,
                     base_url=config.ALIBL_BSUL),
    tools=[get_weather],
    system_prompt="你是一个聊天助手，可以回答用户的问题！"
)

#3.运行智能体
#invoke返回的是一个字典
resp=agent.invoke({"messages":[HumanMessage(content="明天的南京的天气如何")]})

#输出结果
parser=StrOutputParser()
for msg in resp["messages"]:
    print(f"{type(msg)}:{parser.invoke(msg)}")
