from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config

#创建一个类，将向量库和检索过程封装起来，只暴露获取检索器的接口
class VectorRetrieve(object):
    #创建对象
    def __init__(self,embedding):
        #传入嵌入模型
        self.embedding = embedding
        #连接向量库
        self.db=Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=config.PERSIST_DIRECTORY,
        )
    def get_retriever(self):
        vector_retrieve =self.db.as_retriever(search_kwargs={"k":3})
        return vector_retrieve

if __name__ == '__main__':
    #实例化服务，传入通义千问的嵌入模型
    service=VectorRetrieve(DashScopeEmbeddings(model=config.EMBEDDING_MODEL))
    retriever=service.get_retriever()
    res=retriever.invoke("我的体重180斤，尺码推荐")
    print(res[0].page_content)

