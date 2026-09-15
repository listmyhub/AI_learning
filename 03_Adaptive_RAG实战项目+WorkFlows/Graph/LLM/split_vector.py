import datetime
import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config

DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")
#MD5获取
def get_md5(input_str,encoding='utf-8'):
    str_byte=input_str.encode(encoding)
    md5 = hashlib.md5()
    md5.update(str_byte)
    md5_str=md5.hexdigest()
    return md5_str

def check_md5(md5_str:str):
    """检查MD5_str是否出现在MD5。txt中"""
    if not os.path.exists(config.md5_path):
        #文件不存在，先创建空文件
        open(config.md5_path,'w',encoding="utf-8").close()
        return False
    else:
        #逐行搜索
        with open(config.md5_path,'r',encoding="utf-8") as f:
            for line in f.readlines():
                if md5_str==line.strip():
                    return True
        return False
#将未出现的MD5码记录在MD5_path
def md5_save(md5_str:str):
    if not check_md5(md5_str):
        with open(config.md5_path,'a',encoding="utf-8") as f:
            f.write(md5_str+"\n")
    return

class split_vector(object):
    def __init__(self):
        #先创建chroma向量库文件夹
        os.makedirs(config.PERSIST_DIRECTORY,exist_ok=True)
        #创建chroma对象
        self.chroma=Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=DashScopeEmbeddings(model=config.EMBEDDING_MODEL),
            persist_directory=config.PERSIST_DIRECTORY,
        )
        #创建递归分块器
        self.spliter=RecursiveCharacterTextSplitter(
            separators=config.SEPARATOR,
            chunk_overlap=config.CHUNK_OVERLAP,
            chunk_size=config.CHUNK_SIZE,
        )
    def update_chroma(self,text:str,filename:str):
        #先计算文本的Md5码，判断是否已经入库。
        md5=get_md5(text)
        if check_md5(md5):
            return "内容已经存在知识库中"
        #长文本切分成片段，短文本整段入库
        if len(text)>config.CHUNK_SIZE:
            chunks=self.spliter.split_text(text)
        else:
            chunks=[text]
        #每个片段附带相同的元数据
        metadata={
            "source":filename,#来源文件名
            "create_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), #入库时间
            "operator":config.OPERATOR #入库操作人
        }
        print(f"总块数为：{len(chunks)}")
        #批量向量化写入Chroma
        self.chroma.add_texts(chunks,metadatas=[metadata for i in chunks])

        #保存MD5码，防止下次重复用
        md5_save(md5)
        return "成功将内容载入向量库"

if __name__=="__main__":
    data_path=os.path.join(config.BASE_DIR, "../data", "颜色选择.txt")
    with open(data_path,"r",encoding="utf-8") as f:
        text=f.read()
    server=split_vector()
    r=server.update_chroma(text,"颜色选择.txt")
    print(r)





