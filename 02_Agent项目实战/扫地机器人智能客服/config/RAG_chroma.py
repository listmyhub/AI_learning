import os

from dotenv import load_dotenv

from config.PATH import BASE_DIR

load_dotenv()
#数据库的全局配置
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL="text-embedding-v4"
COLLECTION_NAME="agent"
PERSIST_DIRECTORY=os.path.join(BASE_DIR,"output","chroma_bd")

CHUNK_SIZE=1000
CHUNK_OVERLAP=100
SEPARATOR=["\n\n","\n",".","。","!","?"," ",""]
TOP_N=3
