#创建智能体工具集
#ragservice检索工具
#分别是get_weather,get_user_id,get_user_location,get_current_month;模拟外部数据
#最主要的工具是fetch_external_data,从外部数据中获取用户信息
#fill_context_for_report,用于标记“报告生成”工具
import csv
import os
import random

from langchain_core.tools import tool

from RAG.rag_service import RagService
from config import PATH as pt
#全局变量
rag=RagService()
user_id=[str(idx) for idx in range(1001,1011)]  # 与 data/external/records.csv 的用户范围对齐
months=[f"2025-{idx:02d}" for idx in range(1,13)]  # 与 records.csv 的时间格式(YYYY-MM)对齐
external_data={}



@tool
def rag_summary(query:str):
    """
    Description:
        通过rag向量库检索，送入大模型中进行总结工具
    Args:
        query:用户的问题
    Returns:
        返回大模型总结后的回答，为字符串
    """
    return rag.rag_service(query)


@tool
def get_weather(city:str):
    """
    Description:
        获取指定城市的天气
    Args:
        city:指定的城市名
    Returns:
        返回该城市的天气，是一个字符串

    """
    return f"{city}城市的天气味清甜，气温30摄氏度"

@tool
def get_user_id(id:str|None):
    """
    Description:
        获取用户的id
    Args:
        id:用户提供的id或者不提供
    Returns:
        返回用户的id，是字符串格式

    """
    if not id:
        return random.choice(user_id)
    return id

@tool
def get_current_month(month:str|None):
    """
    Description:
        获取当前的月份
    Args：
        month:用户提供的月份或着不提供
    Returns:
        返回当前的的月份，以字符串的形式

    """
    if not month:
        return random.choice(months)
    return month

@tool
def get_user_location():
    """
    Description:
        获取用户所处在的位置
    Returns:
        返回用户的位置，以字符串形式
    """
    return random.choice(["南京","北京","南昌"])


def get_external_data():
    """从外部读取数据，并建立结构化字典"""
    global external_data

    external_data_path=os.path.join(pt.BASE_DIR,"data","external","records.csv")
    if not os.path.exists(external_data_path):
        print(f"【外部数据加载】{external_data_path}文件不存在，加载失败")
        return
    with open(external_data_path,"r",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:
            user_id=row["用户ID"]
            feature=row["特征"]
            efficiency=row["清洁效率"]
            consumables=row["耗材"]
            comparison=row["对比"]
            time=row["时间"]
            if user_id not in external_data:
                external_data[user_id]={}
            external_data[user_id][time]={
                "特征":feature,
                "效率":efficiency,
                "耗材":consumables,
                "对比":comparison
            }
@tool
def fetch_external_data(user_id:str,month:str):
    """
    Descriptions:
        从外部数据中获取指定用户指定时间的使用记录
    Args:
         user_id:用户的id
         month:指定的月份
    Returns:
        返回用户的使用情况，以字符串的形式
    """
    get_external_data()
    try:
        return external_data[user_id][month]
    except KeyError as e:
        print(f"【获取信息工具】获取{user_id}{month}的使用情况失败")
        raise e


@tool
def fill_context_for_report():
    """
    Description:
        无参数输入，无返回值，调用后出发中间件自动为报告生成的场景注入上下文信息，为后续的提示词切换提供上下文信息
    """
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    print(fetch_external_data.func("1001","2025-01"))
    print(get_user_id.func())
    print(get_user_location.func())
    print(get_weather.func("南昌"))
    print(get_current_month.func())




