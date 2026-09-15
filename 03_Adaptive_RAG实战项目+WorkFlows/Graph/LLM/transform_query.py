from langchain_core.prompts import ChatPromptTemplate

import config_data as config
import prompt as pt


llm=config.llm
transform_system_prompt=pt.translate_system_prompt
prompt_template=ChatPromptTemplate.from_messages([
    ("system",transform_system_prompt),
    ("user","user：{question}")
])

transform_chain=prompt_template|llm


if __name__=="__main__":
    resp=transform_chain.invoke({"question":"我的姓名"})
    print(resp)