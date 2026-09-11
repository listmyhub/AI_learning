#agent 的React框架：思考-》行动-》观察-》再思考
#每一轮只调用一个工具
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import config

#定义一个获取家庭电话号码的工具
@tool
def get_number(name:str)->str:
    """
    Description:能够获取某个人的电话号码

    Args:
        name:某个人的姓名

    Returns:
        返回某个人的电话号码，
    """
    return "11654878952"

#定义拨打电话的工具
@tool
def take_phone(number:str)->str:
    """
    Description:
        给这个电话号码进行拨号
    Args:
        number:某个人额电话号码
    Returns:
        返回拨号的状态
    """
    return "您拨打的电话正在通话中"

#创建智能体
agent=create_agent(
    model=config.llm,
    tools=[get_number, take_phone],
    system_prompt="""你是严格按照React框架的智能体，必须安装流程解决问题，
    且每轮智能调用一个工具，禁止单次调用多个工具。并告知我你思考的过程，以及调用工具的原因，按思考，行动，观察三个结构告知我
    """

)
#以流式输出的方式运行智能体
for chunk in agent.stream(
    input={"messages":[HumanMessage(content="我想给我女朋友打电话，请帮我打一下")]},
    stream_mode="values"
):
    messages=chunk["messages"][-1]
    if messages:
        print(f"{type(messages).__name__}:{messages.content}")
    try:
        if messages.tool_calls:
             print(f"工具调用:{[msg for msg in messages.tool_calls]}")
    except:
        pass

