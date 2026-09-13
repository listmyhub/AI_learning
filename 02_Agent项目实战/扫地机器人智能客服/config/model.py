import os

from langchain_openai import ChatOpenAI



#定义全局模型
LLM=ChatOpenAI(
    model='qwen-turbo',
    api_key=os.getenv("ALIBL_API_KEY"),
    base_url=os.getenv("ALIBL_BSUL"),
    temperature=0.5
)