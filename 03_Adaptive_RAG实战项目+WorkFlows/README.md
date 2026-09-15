# Adaptive RAG 实战项目 + WorkFlows

基于 **LangGraph** 实现的自适应检索增强生成（Adaptive RAG）实战项目。系统会根据用户问题**自动判断**走「向量库检索（RAG）」还是「Tavily 联网搜索」，并对检索结果进行**相关性过滤**、**查询改写**、**答案生成**与**答案质量评估**，形成一条可自我修正的自适应检索生成工作流。

知识库内容聚焦**服装领域**，覆盖三个主题：尺码推荐、洗涤养护、颜色选择。

## 核心特性

- **智能路由**：LLM 判断问题适合 RAG 检索还是联网搜索
- **向量检索**：Chroma 向量库 + 通义千问 `text-embedding-v4` 嵌入模型
- **相关性过滤**：对每条检索片段与用户问题做相关性打分，剔除无关文档
- **查询改写**：检索结果不可用时，自动改写问题重新检索（最多重试 3 次）
- **答案评估**：生成答案后进行质量打分，不满意则改写问题重新检索
- **增量入库去重**：基于 MD5 记录已入库内容，避免重复写入
- **失败兜底**：多次检索无果时给出友好提示并结束流程

## 项目结构

```
03_Adaptive_RAG实战项目+WorkFlows/
└── Graph/
    ├── config_data.py              # 全局配置：路径、向量库、模型、检索与分块参数
    ├── prompt.py                   # 各节点使用的提示词（生成/路由/过滤/改写/评估）
    ├── md5.txt                     # 已入库内容的 MD5 记录（增量去重）
    ├── chroma_db/                  # Chroma 向量库持久化目录
    ├── data/                       # 知识库原始文本（入库数据源）
    │   ├── 尺码推荐.txt
    │   ├── 洗涤养护.txt
    │   └── 颜色选择.txt
    ├── LLM/                        # 单点 LLM 能力模块
    │   ├── route_choice.py         # 路由判断：retriever / web_search
    │   ├── vector_retrieve.py      # 连接 Chroma，构建向量检索器（top-k=3）
    │   ├── filter_document.py      # 检索片段相关性过滤（YES/NO）
    │   ├── transform_query.py      # 查询改写（换一种问法）
    │   ├── generate_model.py       # 基于参考资料生成答案
    │   └── grade_generation.py     # 答案质量评估（Yes/No）
    └── WorkFlows/                  # LangGraph 工作流
        ├── WorkFlows.py            # 图状态定义 + 各节点函数实现
        └── complieflows.py         # 节点/边编排、条件分支、编译入口
```

## 工作流说明

整体流程如下：

```
                          ┌─────────────────┐
    START ──────────────► │ 路由判断          │
                          │ (web_search_or_  │
                          │  retriever)      │
                          └────────┬────────┘
                  ┌────────────────┴────────────────┐
           web_search                           retriever
                  │                                  │
                  ▼                                  ▼
             web_search                         retrieve（向量检索）
                  │                                  │
                  │                                  ▼
                  │                          filter_document（相关性过滤）
                  │                                  │
                  │                                  ▼
                  │                          decide_to_generate
                  │                    ┌───────────────┼────────────────┐
                  │              有可用文档          全部被过滤          re_count > 3
                  │                    │               │                  │
                  │                    ▼               ▼                  ▼
                  └──────────────► generate    transform_query      give_up
                                      │               │                  │
                                      ▼               ▼                  ▼
                              grade_generation     retrieve（重新检索）   END
                                │        │
                           useful     not useful
                                │        │
                                ▼        └─────────► transform_query
                               END
```

**核心分支逻辑：**

1. **路由**：LLM 判断问题应走 `retriever`（知识库检索）还是 `web_search`（Tavily 联网搜索）。联网搜索结果直接进入生成节点。
2. **检索 → 过滤**：向量检索结果先逐条判断与问题的相关性，过滤掉无关片段。
3. **过滤后决策**：
   - 存在可用文档 → 直接生成答案；
   - 文档被全部过滤 → 改写问题，重新检索；
   - 累计检索次数 > 3 → 放弃并结束，返回兜底提示。
4. **生成 → 评估**：对生成的答案打分，若不合格则改写问题重新检索，形成闭环。

**图状态（GraphState）字段：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `question` | str | 用户问题（可能被改写） |
| `documents` | List[Document] | 检索/搜索得到的文档片段 |
| `generation` | str | 最终生成的答案 |
| `datasource` | Optional[str] | 数据来源（retriever / web_search） |
| `re_count` | int | 累计检索次数（重试上限 3 次） |

## 环境准备

- Python 3.13（项目以 3.13 运行，3.10+ 应可兼容）
- 需要以下第三方依赖（`pip install`）：

```
langchain-core
langchain-openai
langchain-chroma
langchain-community
langchain-text-splitters
langchain-tavily
langgraph
python-dotenv
pydantic
```

> 注：`generate_model.py` 顶部有一条 `langchain_classic` 的导入语句，若环境中未安装该包且不影响使用，可删除此行（详见「已知事项」）。

### 环境变量

在项目根目录（或 `Graph/` 目录）创建 `.env` 文件，配置以下变量（`config_data.py` 通过 `python-dotenv` 加载）：

| 变量名 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（用于文本嵌入） |
| `ALIBL_API_KEY` | 通义千问对话模型的 API Key |
| `ALIBL_BSUL` | 通义千问 OpenAI 兼容接口的 Base URL |
| `TAVILY_API_KEY` | Tavily 联网搜索 API Key |

## 使用说明

> 所有命令需在 `Graph/` 目录下执行（代码中的模块导入均以 `Graph/` 为根）。

### 1. 知识入库

将知识文本放入 `Graph/data/`，然后使用 `LLM/split_vector.py` 中的 `split_vector` 类入库：

```python
from LLM.split_vector import split_vector

server = split_vector()
# text 为待入库文本，filename 为来源文件名（会写入元数据）
result = server.update_chroma(text, "洗涤养护.txt")
print(result)   # 成功入库 / 内容已经存在知识库中
```

入库逻辑：
- 长文本（超过 `CHUNK_SIZE=400`）按递归字符切分，短文本整段入库；
- 每条片段附带 `source`（来源）、`create_time`（入库时间）、`operator`（操作人）元数据；
- 内容 MD5 已存在于 `md5.txt` 时跳过，避免重复入库。

### 2. 运行完整工作流

```bash
cd Graph
python WorkFlows/complieflows.py
```

默认的测试用例为「我想穿什么颜色的衣服比较合适」，执行结束后输出最终答案 `res["generation"]`。

也可以在代码中自定义输入状态：

```python
input_state = {
    "question": "我的体重 180 斤，请推荐尺码",
    "documents": [],
    "generation": "",
    "datasource": "",
    "re_count": 0,
}
res = app.invoke(input_state)
print(res["generation"])
```

## 关键配置项（`config_data.py`）

| 配置 | 值 | 说明 |
| --- | --- | --- |
| `COLLECTION_NAME` | `RAG` | Chroma 集合名 |
| `PERSIST_DIRECTORY` | `./chroma_db` | 向量库持久化目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 400 / 50 | 文本分块大小与重叠 |
| `TOP_K` | 3 | 检索返回片段数（`vector_retrieve.py` 中固定为 3） |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 通义千问文本嵌入模型 |
| `CHAT_MODEL` | `qwen-turbo` | 对话模型（temperature=0.5） |
| `OPERATOR` | `小范` | 入库元数据中的操作人 |

## 已知事项

- `config_data.py` 中 `SEPARATOR` 的取值为 `"/n/n"`、`"/n"`（正斜杠），若递归切分未按预期换行，可改为标准换行符 `"\n\n"`、`"\n"`。
- `config_data.py` 中环境变量名为 `ALIBL_BSUL`（疑似 `BSUL` 为 Base URL 的笔误），请确保 `.env` 中变量名与之一致。
- `generate_model.py` 顶部存在一条未使用的 `langchain_classic` 导入，若该包不可用导致报错，删除此行即可。
- 该项目的代码仅供学习参考，生产使用前建议补充异常处理与日志。

## 技术栈

LangGraph · LangChain · Chroma · 通义千问（qwen-turbo / text-embedding-v4） · Tavily · Pydantic
