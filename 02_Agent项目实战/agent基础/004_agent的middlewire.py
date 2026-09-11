from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, \
    wrap_tool_call
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
import config

@tool(description="查询天气")
def get_weather():
    return "晴天"

#生命周期钩子
@before_agent
def log_before_agent(state: AgentState,runtime: Runtime):
    #agent自行前自行调用函数，传入state和runtime两个对象
    print(f"[before_agent]agent启动，并附带{len(state["messages"])}")
    print(f"{runtime}")


@after_agent
def log_after_agent(state: AgentState,runtime: Runtime):
    print(f"[after_agent]agent结束，并附带了{len(state["messages"])}")

@before_model
def log_before_model(state: AgentState,runtime: Runtime):
    print(f"[before_model]模型即将调用，并附带{len(state["messages"])}消息")
    print(f"{runtime}")

@after_model
def log_after_model(state: AgentState,runtime: Runtime):
    print(f"[after_model]模型调用结束，并附带{len(state["messages"])}消息")
    print(f"{runtime}")

#包装调用钩子

@wrap_model_call
def model_call_hook(request,handler):
    print(f"模型调用啦，调用的模型是{config.CHAT_MODEL}")
    return handler(request)

@wrap_tool_call
def tool_call_hook(request,handler):
    print(f"工具调用啦，调用了{request.tool_call["name"]}")
    print(f"工具的参数是{request.tool_call["args"]}")
    return handler(request)


#创建agent,并注入中间件
agent=create_agent(
    model=config.llm,
    tools=[get_weather],
    middleware=[log_before_agent,log_after_agent,log_before_model,log_after_model,model_call_hook,tool_call_hook],

)
#运行测试
resp=agent.invoke({"messages":[HumanMessage(content="明天南京天气如何")]})
print("-"*50)
print(resp)