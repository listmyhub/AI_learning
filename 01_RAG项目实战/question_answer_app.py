import streamlit as st
from pass_llm import ragservice
#制作一个sreamlit页面：模拟智能客服

#页面标题和分隔线
st.title("智能客服")
st.divider()

#初始化聊天记录（session_state)
#如果用户刷新页面，保证之前聊天内容不会丢失

if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你吗？"}]

#初始化RAG服务
if "rag" not in st.session_state:
    st.session_state["rag"]=ragservice()

#渲染之前的聊天记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

#获取用户的输入
prompt=st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})
    ai_list=[]
    #显示AI思考的加载动画
    with st.spinner("AI思考中...",show_time=True):
        res=st.session_state["rag"].chain.stream({"input":prompt},{"configurable":{"session_id":"user_002"}})
        #定义一个捕捉器，一边把生成的内容渲染出去，一边吧内容放入缓存中
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        #将流式内容渲染到前端页面
        st.chat_message("assistant").write_stream(capture(res,ai_list))

    #把完整的流式输出完整的拼起来,存入历史记录
    st.session_state["message"].append({"role":"assistant","content":"".join(ai_list)})



