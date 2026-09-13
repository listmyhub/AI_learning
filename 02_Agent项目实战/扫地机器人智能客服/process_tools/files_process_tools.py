import hashlib
import os
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader


#读取文件夹里面所有的文件的绝对路径，包括pdf,txt文件
def load_files_path(path:str,requested_files:tuple[str])->List[str]:
    if not os.path.exists(path):
        print(f"{path}不是一个文件夹")
        return []
    files=[]
    for file in os.listdir(path):
        if file.endswith(requested_files):
            files.append(os.path.join(path,file))
    return files

#获取每个文件的md5码，以此为导入向量库的凭证
def get_files_md5(path:str):
    if not os.path.exists(path):
        print(f"[md5计算]路径{path}不存在")
        return
    if not os.path.isfile(path):
        print(f"[md5计算]路径{path}不是一个文件")
        return
    md5=hashlib.md5()
    chunk_size=4096  #分片，避免文件过大
    try:
        with open(path,'rb') as f:
            while chunk:= f.read(chunk_size):
                md5.update(chunk)
        md5_hex=md5.hexdigest()
        return md5_hex
    except Exception as e:
        print(f"计算文件{path}MD5失败，{str(e)}")
        return None

#加载两种类型的文件，分别时pdf,txt
def load_pdf(path:str):
    if not os.path.exists(path):
        print(f"[文件加载]文件{path}不存在")
        return
    return PyMuPDFLoader(path).load()   #加载pdf文件

def load_txt(path:str):
    if not os.path.exists(path):
        print(f"[文件加载]文件{path}不存在")
        return
    return TextLoader(path).load() #加载txt文件