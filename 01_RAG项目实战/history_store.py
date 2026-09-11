import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import config_data as config
#会话记忆的存储目录
Memory_path=os.path.join(config.BASE_DIR,"memory")

#获取对用id 的记忆
def get_history(session_id):
    return ChatMemoryHistory(session_id,Memory_path)

#将对话历史封装成一个类，保存为json文件进行保存
class ChatMemoryHistory(BaseChatMessageHistory):
    #定义对象
    def __init__(self,session_id,path):
        #一个session_id对应一个文件
        self.session_id=session_id
        self.storage_path=path
        #完整的文件路径
        self.filepath=os.path.join(self.storage_path,session_id)
        #确保文件夹存在
        os.makedirs(os.path.dirname(self.filepath),exist_ok=True)

    def add_messages(self,messages:Sequence[BaseMessage]):
        """追加新消息写入文件"""
        #获取当前的消息列表
        all_messages=list(self.messages)
        all_messages.extend(messages)

        #使用官方工具message
        new_messages=[message_to_dict(message) for message in all_messages]

        #将数据同步写入本地文件中
        with open(self.filepath,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)
    @property
    def messages(self):
        #读取文档中的历史消息
        try:
            with open(self.filepath,"r",encoding="utf-8") as f:
                messages=json.load(f)
            return messages_from_dict(messages)
        except FileNotFoundError:
            return []

    def clear(self):
        #写入空列表
        with open(self.filepath,"w",encoding="utf-8") as f:
            json.dump([],f)
