
from langgraph.graph import END,StateGraph,START
from WorkFlows import Graphstate, web_search, retriever, filter_documents, generate, transform_query, \
    web_search_or_retriever, decide_to_generate, grade_generation
#建立工作流
workflows=StateGraph(Graphstate)

#定义每一个节点
workflows.add_node("web_search",web_search)
workflows.add_node("retrieve",retriever)
workflows.add_node("filter_document",filter_documents)
workflows.add_node("generate",generate)
workflows.add_node("transform_query",transform_query)
def give_up_answer(state):
    return {"generation": "Sorry, the current knowledge is not enough to answer this question."}
workflows.add_node("give_up",give_up_answer)

#定义工作流的边
workflows.add_edge("give_up",END) #结束的边
#增加条件边，确定是走web_search还是RAG检索
workflows.add_conditional_edges(START,web_search_or_retriever,{
    "web_search":"web_search",
    "retriever":"retrieve",
})
#网络搜索后，直接进行生成答案
workflows.add_edge("web_search","generate")
#将检索完的内容，传入下个节点进行过滤参考资料，减少没用的检索片段
workflows.add_edge("retrieve","filter_document")
#如果没有参考资料的话，换一种问法继续检索，有资料则直接生成答案，如果检索超时直接结束
workflows.add_conditional_edges("filter_document",decide_to_generate,{
    "transform_query":"transform_query",
    "generate":"generate",
    "give_up":"give_up",
})
#重新生成语句后在进行检索
workflows.add_edge("transform_query","retrieve")
#根据生成的答案，判断答案是否有用
workflows.add_conditional_edges("generate",grade_generation,{
    "useful":END,
    "not useful":"transform_query",
})

#编译工作流
app=workflows.compile()


#测试
if __name__=="__main__":
    input_state={
        "question":"我想穿什么颜色的衣服比较合适",
        "documents":[],
        "generation":"",
        "datasource":"",
        "re_count":0
    }
    res=app.invoke(input_state)
    print(res["generation"])