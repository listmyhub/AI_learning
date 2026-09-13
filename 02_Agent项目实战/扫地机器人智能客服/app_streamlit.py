import time

import streamlit as st

from Agent.react_agent import ReactAgent

st.title("扫地机器人客服")
st.divider()

#初始化聊天消息
if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好，我是扫地机器人客服，有什么可以帮助你的码？"}]

#初始化智能体
if "agent" not in st.session_state:
    st.session_state["agent"] =ReactAgent()

#遍历聊天消息中的每一条消息，加入到页面中
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

#获取用户输入的问题
query= st.chat_input()
if query:
    st.session_state["message"].append({"role":"user","content":query})
    st.chat_message("user").write(query)

#智能体思考回答问题
res=[]
with st.spinner("智能体思考中...",show_time=True):
    if query:
        result=st.session_state["agent"].output_stream(query)
        def capture(generator,res):
            for chunk in generator:
                res.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char
        st.chat_message("assistant").write_stream(capture(result,res))
        st.session_state["message"].append({"role":"assistant","content":"".join(res).strip()})



