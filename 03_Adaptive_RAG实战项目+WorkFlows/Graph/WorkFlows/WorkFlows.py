
from typing import List, Optional,TypedDict
from LLM.route_choice import route_chain
from LLM.filter_document import filter_chain
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_tavily import tavily_search
from LLM.vector_retrieve import VectorRetrieve
from LLM.generate_model import Generate_Chain
from LLM.grade_generation import grade_chain
import config_data as config
from LLM.transform_query import transform_chain
TAVILY_API_KEY=config.TAVILY_API_KEY


#定义一个继承TypedDict的类作为工作流传递的信息
class Graphstate(TypedDict):
    """
      Represents the state of our graph.

      Attributes:
          question: question
          generation: LLM generation
          documents: list of documents
      """
    documents:List[Document]
    question: str
    generation: str
    datasource:Optional[str]
    re_count:int


#网络搜索工具
def web_search(state):
    query=state["question"]
    web_search_tool=tavily_search.TavilySearch(max_results=3)
    rep=web_search_tool.invoke(query)
    return format_document(state,rep)

#将网络搜索得到的document转成列表
def format_document(state,rep):
    web_result=[]
    for result in rep['results']:
        if not result["content"]:
            continue
        else:
            web_result.append(Document(page_content=result["content"],metadata={"url":result["url"]}))
    return {"documents":web_result,"question":state["question"]}

#将document列表转成字符串导入大模型之中
def docment_to_str(doc:List[Document]):
    str_result=""
    for document in doc:
        str_result+=document.page_content+"\n"
    return str_result

#生成答案
def generate(state):
    """Generate answer
    Args:
        state:当前图节点的消息状态
    Returns:
        返回新的key,加入到state,生成的答案generation
        """
    str_result=docment_to_str(state["documents"])
    generator=Generate_Chain()
    result=generator.chain.invoke({"question":state["question"],"context":str_result})
    return {"documents":state["documents"],"question":state["question"],"generation":result.content}

#对用户问题进行数据库检索
def retriever(state):
    query=state["question"]
    re_count=state["re_count"]
    vectorize=VectorRetrieve(DashScopeEmbeddings(model=config.EMBEDDING_MODEL))
    ret=vectorize.get_retriever()
    result=ret.invoke(query)
    re_count+=1
    print(f"第{re_count}次检索：")
    return {"documents":result,"question":query,"re_count":re_count}


#判断是否是网络搜素还是进行检索
def web_search_or_retriever(state):
    query=state["question"]
    result=route_chain.invoke({"question":query})
    if result is None:
        print("default--retriever")
        return "retriever"
    datasource=getattr(result,"datasource",None)
    if datasource=="web_search":
        print("route the question to web_search")
        return "web_search"
    else:
        print("route the question to retriever")
        return "retriever"

def filter_documents(state):
    #对检索出来的片段与用户问题进行打分过滤，获取有用的信息
    documents=state["documents"]
    question=state["question"]
    filter_docs=[]
    re_count=state["re_count"]
    if not documents:
        print("no documents")
    for d in documents:
        resp=filter_chain.invoke({"documents":d,"question":question})
        score=getattr(resp,"relation",None)
        # print(f"{d}与{question}的相关性：{score}")
        if score=="YES":
            # print(f"{d} is relevant")
            filter_docs.append(d)
        else:
            # print(f"{d} is not related to question")
            continue
    return {"documents":filter_docs,"question":question,"re_count":re_count}

#判断是否要进行生成答案
def decide_to_generate(state):
    """decide to gennerate a answer or regenerate a new query"""
    documents=state["documents"]
    re_count=state["re_count"]
    if re_count>3:
        return "give_up"
    if not documents:
        print("filter all documents,no useful documents")
        #所有的documents全部过滤掉了，需要重新生成一个question
        return "transform_query"
    else:
        #documents没有全部过滤，直接生成
        return "generate"

#换一种问法
def transform_query(state):
    query=state["question"]
    documents=state["documents"]
    re_count=state["re_count"]
    result=transform_chain.invoke({"question":query})
    best_question=result.content
    print(f"新的问题是{best_question}")
    return {"documents":documents,"question":best_question,"re_count":re_count}


#判断生成的答案是否有用
def grade_generation(state):
    query=state["question"]
    generation=state["generation"]
    rep=grade_chain.invoke({"question":query,"generation":generation})
    result=getattr(rep,"assessment",None)
    if result=="Yes":
        return "useful"
    else:
        return "not useful"









