from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from Agent.agent_middleware import log_before_model, monitor_tool, report_prompt
from Agent.agent_tools import rag_summary, get_weather, get_user_id, get_user_location, get_current_month, \
    fetch_external_data, fill_context_for_report
from config import model as md
from process_tools.prompt_process_tools import load_system_prompts


#组装智能体：7个工具+3个中间件+系统提示词
class ReactAgent:
    #创建agent的对象
    def __init__(self):
        self.agent=create_agent(
            model=md.LLM,
            tools=[rag_summary,get_weather,get_user_id,get_user_location,get_current_month,fetch_external_data,
                   fill_context_for_report],
            middleware=[log_before_model,monitor_tool,report_prompt],
            system_prompt=load_system_prompts()
        )
    def output_stream(self,query:str):
        # 注意：create_agent 的输入必须是 {"messages": [...]} 结构，
        # 直接传 {"role","content"} 会导致用户消息丢失、模型凭空作答
        input_dict={
            "messages":[HumanMessage(content=query)]
        }
        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            latest_message=chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip()+"\n"


if __name__=="__main__":
    agent=ReactAgent()
    for chunk in agent.output_stream("生成报告"):
        print(chunk,end="",flush=True)
