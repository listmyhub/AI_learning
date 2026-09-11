from dotenv import load_dotenv
import os

from langchain_community.utilities.tavily_search import TAVILY_API_URL
from langchain_openai import ChatOpenAI

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BSUL=os.getenv("DEEPSEEK_BSUL")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ALIBL_API_KEY = os.getenv("ALIBL_API_KEY")
ALIBL_BSUL=os.getenv("ALIBL_BSUL")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")

#模型
CHAT_MODEL="qwen-turbo"
llm=ChatOpenAI(model=CHAT_MODEL,
                     api_key=ALIBL_API_KEY,
                     base_url=ALIBL_BSUL)

