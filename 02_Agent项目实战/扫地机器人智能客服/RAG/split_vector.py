import os
from config import RAG_chroma as rag
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from process_tools.files_process_tools import load_files_path, get_files_md5, load_txt, load_pdf
from config import load_requested_files as rf
from config import PATH as pt


class VectorService:
    def __init__(self):
        self.vector_db=Chroma(
            embedding_function=DashScopeEmbeddings(model=rag.EMBEDDING_MODEL,dashscope_api_key=rag.DASHSCOPE_API_KEY),
            collection_name=rag.COLLECTION_NAME,
            persist_directory=rag.PERSIST_DIRECTORY,
        )
        self.spliter=RecursiveCharacterTextSplitter(
            separators=rag.SEPARATOR,
            chunk_size=rag.CHUNK_SIZE,
            chunk_overlap=rag.CHUNK_OVERLAP,
            length_function=len
        )
    def get_retriever(self):
        return self.vector_db.as_retriever(search_kwargs={"k":rag.TOP_N})
    #检查某个文件是否已经入库了，观察MD5是否已经保存了
    def check_md5(self,md5:str):
        if not os.path.exists(pt.MD5_PATH):
            #创建文件
            open(pt.MD5_PATH,'w',encoding="utf-8").close()
            return False
        with open(pt.MD5_PATH,"r",encoding="utf-8") as f:
            for line in f.readlines():
                if md5 ==line.strip():
                    return True #说明MD5已经入库了
        return False

    def save_md5(self,md5:str):
        with open(pt.MD5_PATH,"a",encoding="utf-8") as f:
            f.write(md5+"\n")

    #读取文件的分开后的document
    def load_document(self,path:str):
        if path.endswith(".txt"):
            return load_txt(path)
        if path.endswith(".pdf"):
            return load_pdf(path)

    def load_files_to_vector(self):
        """
        从数据文件夹中读取pdf文件和txt文件，并加载进入向量库
        """
        file_folder_dir=os.path.join(rag.BASE_DIR,"data")
        files_path=load_files_path(file_folder_dir,rf.REQUESTED_FILES)
        for file in files_path:
            #获取每一个文件的MD5码
            md5=get_files_md5(file)
            if self.check_md5(md5):
                print(f"【知识库加载】文件{file}已经入库了，不需要导入了")
            else:
                #读取该文件
                try:
                    docs=self.load_document(file)
                    chunks=self.spliter.split_documents(docs)
                    if not chunks:
                        print(f"[加载知识库]{file}分块之后没有有效部分")
                        continue
                    self.vector_db.add_documents(chunks)    #将分块之后的document加载到向量库里
                    self.save_md5(md5) #w文件已入库，记录MD5
                    print(f"【知识库加载】文件{file}加载成功")
                except Exception as e:
                    print(f"【知识库加载】文件读取失败：{str(e)}")
                    continue



if __name__=="__main__":
    vs=VectorService()
    vs.load_files_to_vector()
    retriever=vs.get_retriever()
    res=retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*40)

