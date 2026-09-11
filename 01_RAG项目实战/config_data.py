import os
from dotenv import load_dotenv
load_dotenv()
#路径配置
#相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#已入库的MD5记录文件路径
md5_path=os.path.join(BASE_DIR,"md5.txt")

#Chroma 向量库配置
COLLECTION_NAME="RAG"   #向量库的表名
PERSIST_DIRECTORY=os.path.join(BASE_DIR,"chroma_db")    #向量库的存储目录

#文本分割的参数
CHUNK_SIZE=1000 #每个块的最大长度
CHUNK_OVERLAP=100   #块与块之间重叠的字符数
SEPARATOR=["/n/n","/n","。",".","!","?",""]  #递归的优先切割符号

#检索参数
TOP_K=3 #检索返回大模型的片段数量

#模型配置
EMBEDDING_MODEL="text-embedding-v4"     #文本嵌入模型
CHAT_MODEL="qwen-turbo" #对话模型
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")    #阿里百炼的key
ALIBL_API_KEY=os.getenv("ALIBL_API_KEY")
ALIBL_BASE_URL=os.getenv("ALIBL_BSUL")
#入库的元数据
OPERATOR="小范"   #入库操作人

#会话记忆配置
session_config={
    "configurable":{
        "session_id":"user_001"
    }
}





