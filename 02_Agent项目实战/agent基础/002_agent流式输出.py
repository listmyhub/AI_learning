from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import config

#定义一个简单的获取股价的工具
@tool
def get_price(name:str):
    """获取某只股票的价格

    Args:
        name:股票的名字

    Returns:
        返回当时股票的价格
        """
    return f"股票{name}的单价是20元"


@tool
def get_info(name:str):
    """
    description:
        获取某只股票的信息

    Args:
        name：股票的名字

    Returns:
        返回字符串类型，得到股票的信息

    """
    return f"股票{name}，是一家A股上市公司，专注于教育行业"

#创建智能体

agent=create_agent(
    model=config.llm,
    tools=[get_price,get_info],
    system_prompt="你是一个智能助手，可以回答股票相关问题，记住请告知我思考过后曾，让我知道你为什么调用某个工具"
)

#以流式方式运行智能体
for chunk in agent.stream(
    input={"messages":[HumanMessage(content="星火教育的股价是多少，并介绍一下这只股票")]},
    stream_mode="values"
):
    messages=chunk["messages"][-1]
    if messages:
        print(type(messages).__name__,messages.content)
    #如果模型要调用工具，则打印调用工具的信息
    try:
        if messages.tool_calls:
            print(f"工具调用：{[msg for msg in messages.tool_calls]}")
    except:
        pass

