
from typing import Literal


from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import config_data as config
import prompt as pt

#定义一种输出包含assessment:List["Yes","No"]的输出格式
class GradeGeneration(BaseModel):
    """评估生成的解决用户问题的答案的好坏"""
    assessment:Literal["Yes","No"]=Field(...,description="对生成的答案进行评估好坏，‘Yes' or 'No'")

#模型
llm=config.llm
#结构化绑定
llm_structure=llm.with_structured_output(GradeGeneration)
#提示词
assess_system_prompt=pt.assess_system_prompt
#提示词模板
prompt_template=ChatPromptTemplate.from_messages([
    ("system",assess_system_prompt),
    ("user","用户的问题：{question}\n生成的答案：{generation}")
])
#创建链
grade_chain=prompt_template|llm_structure

#测试
if __name__=="__main__":
    res=grade_chain.invoke({"question":"我的尺码推荐","generation":"推荐你穿衣服"})
    print(res)