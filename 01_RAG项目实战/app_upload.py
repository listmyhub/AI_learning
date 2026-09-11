import streamlit as st
from split_vector import split_vector
#上传文件更新知识库：streamlit页面

st.title("知识库更新服务")

uploader=st.file_uploader("请上传TXT文件",
                          type=["txt"],
                          accept_multiple_files=False)
#初始化更新服务
if "service" not in st.session_state:
    st.session_state["service"]=split_vector()

if uploader is not None:
    #文件的基本信息
    file_name=uploader.name
    file_type=uploader.type
    file_size=uploader.size/1024
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type}|大小：{file_size:.2f}KB")

    #将上传的字节解码成字符串
    text=uploader.getvalue().decode("utf-8-sig")

    #调用向量库写入
    with st.spinner("载入知识库中..."):
        result=st.session_state["service"].update_chroma(text,file_name)
        st.write(result)





