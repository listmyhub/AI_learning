from importlib.metadata import metadata
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import prompt as pt
import config_data as config

#定义一种结构化输出的类
class FilterDocument(BaseModel):
    """将与用户问题无关的检索片段过滤掉"""

    relation:Literal["YES","NO"]=Field(...,description="根据用户检索出来的片段，判断是否与用户的问题相关，YES or NO,过滤掉不相关的信息")

#调用的模型
llm=config.llm
#结构化输出绑定
llm_structure_filter=llm.with_structured_output(FilterDocument)
relative_prompts=pt.relative_prompt
#创建提示词模板
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",relative_prompts),
        ("user","retrieve_documents:{documents}\nuser_question:{question}"),
    ]
)
#创建链
filter_chain=prompt_template|llm_structure_filter


#测试
if __name__=="__main__":
    res=filter_chain.invoke({"documents":Document(
        page_content="""身高：155-165cm，体重：75-95斤，建议尺码S。
身高：160-170cm，体重：90-115斤，建议尺码M。
身高：165-175cm，体重：115-135斤，建议尺码L。
身高：170-178cm，体重：130-150斤，建议尺码XL。
身高：175-182cm，体重：145-165斤，建议尺码2XL。
身高：178-185cm，体重：160-180斤，建议尺码3XL。
身高：180-190cm，体重：180-210斤，建议尺码4XL。
身高：190cm+，体重：210斤+，建议尺码5XL。""",metadata={"url":"1234"}),"question":"我的尺码推荐"})
    print(res)