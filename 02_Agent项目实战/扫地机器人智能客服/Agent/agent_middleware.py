from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, before_model
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

from process_tools.prompt_process_tools import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(
        request:ToolCallRequest,
        handler
):
    #工具执行监控
    print(f"【工具监控】执行工具：{request.tool_call["name"]}")
    print(f"【工具监控】传入的参数：{request.tool_call["args"]}")

    try:
        result=handler(request)
        print(f"【工具监控】工具{request.tool_call["name"]}调用成功")

        #动态切换场景：如果调用了fill_context_for_report工具，则标记上下文
        if request.tool_call["name"]=="fill_context_for_report":
            request.runtime.context["report"]=True
        # 必须把工具结果返回给langgraph，否则会把None写进消息流导致报错
        return result
    except Exception as e:
        print(f"{request.tool_call["name"]}调用失败：{str(e)}")
        raise e
@before_model
def log_before_model(state:AgentState,runtime:Runtime):
    if state["messages"]:
        print(f"[log_before_model]{type(state["messages"][-1]).__name__}:|{state["messages"][-1].content}")
        print(f"[log_before_model]即将调用模型，带有{len(state["messages"])}条消息")
    return None


@dynamic_prompt #每一次生成提示词之前，调用此函数
def report_prompt(request):
    print(f"【提示词生成】提示词生成中")
    is_report=request.runtime.context.get("report",False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()


