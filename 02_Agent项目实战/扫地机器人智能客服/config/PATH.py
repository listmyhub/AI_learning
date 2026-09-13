import os

#目录相对路径
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

#md5存储路径
MD5_PATH=os.path.join(BASE_DIR,"output","md5.txt")