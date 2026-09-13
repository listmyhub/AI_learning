# Agent 基础 —— LangChain Agent 入门示例

一套循序渐进的 **LangChain Agent** 学习示例，从「最小可运行的 Agent」逐步深入到「流式输出 → ReAct 框架 → 中间件（Middleware）」，覆盖 Agent 开发的四大核心知识点。适合作为学习 `create_agent`、工具定义、中间件机制的第一份上手代码。

> 本项目位于 `AI_leaning/02_Agent项目实战/agent基础`，对应实战项目为同级目录的「扫地机器人智能客服」。

---

## 学习路径

| 编号 | 文件 | 知识点 |
| --- | --- | --- |
| 001 | `001_Agent初体验.py` | 最小 Agent：`@tool` 定义工具、`create_agent` 创建智能体、`invoke` 运行、`StrOutputParser` 提取文本 |
| 002 | `002_agent流式输出.py` | 多工具 Agent：`agent.stream` 流式输出、观察工具调用信息 |
| 003 | `003_agent的React框架.py` | ReAct 框架：思考 → 行动 → 观察 → 再思考，单轮单工具调用 |
| 004 | `004_agent的middlewire.py` | 中间件机制：Agent/模型/工具的全生命周期钩子 |
| 公共 | `config.py` | 环境变量与模型（qwen-turbo）统一配置 |

建议按编号顺序阅读：**001 学会「跑起来」→ 002 学会「看过程」→ 003 理解「决策框架」→ 004 理解「可观测与拦截」**。

---

## 技术栈

- **框架**：LangChain `create_agent` + LangGraph（Agent 运行时、中间件钩子）
- **模型**：通义千问 `qwen-turbo`（OpenAI 兼容接口，`ChatOpenAI`，统一在 `config.py` 配置）
- **工具**：`langchain_core.tools.tool` 装饰器定义

---

## 各示例详解

### 001_Agent初体验.py —— 最小可运行 Agent

```python
@tool("get_weather", description="查询天气")
def get_weather(query: str):
    """返回今天的天气情况"""
    return "晴天"

agent = create_agent(
    model=ChatOpenAI(...),
    tools=[get_weather],
    system_prompt="你是一个聊天助手，可以回答用户的问题！"
)

resp = agent.invoke({"messages": [HumanMessage(content="明天的南京的天气如何")]})
```

**要点**：`invoke` 返回的消息字典；遍历 `resp["messages"]` 可以看到 AI 思考、工具调用、最终回答的完整消息链；`StrOutputParser` 从消息对象中提取纯文本。

### 002_agent流式输出.py —— 多工具 + 流式输出

```python
@tool
def get_price(name: str): ...   # 获取股票价格

@tool
def get_info(name: str): ...    # 获取股票信息

agent = create_agent(model=config.llm, tools=[get_price, get_info], system_prompt=...)

for chunk in agent.stream(input={"messages": [...]}, stream_mode="values"):
    messages = chunk["messages"][-1]
    ...
    if messages.tool_calls:   # 观察模型发起的工具调用
        print(f"工具调用：{messages.tool_calls}")
```

**要点**：`stream_mode="values"` 下每一块都是完整消息状态，取 `[-1]` 即最新消息；`messages.tool_calls` 能直接看到模型决定调用哪些工具及参数——这是理解 Agent 决策过程的关键入口。

### 003_agent的React框架.py —— ReAct 决策框架

```python
@tool
def get_number(name: str) -> str: ...   # 查电话号码

@tool
def take_phone(number: str) -> str: ... # 拨打电话

agent = create_agent(
    model=config.llm,
    tools=[get_number, take_phone],
    system_prompt="你是严格按照React框架的智能体，每轮只能调用一个工具……按思考、行动、观察三个结构告知我"
)
```

**要点**：演示「思考 → 行动 → 观察 → 再思考」的闭环——模型先调用 `get_number` 拿到号码，观察到结果后再调用 `take_phone` 拨号，最终给出结论。提示词中约束「每轮只调用一个工具」以展示多轮推理过程；实际项目中可通过提示词或中间件控制并行调用策略。

### 004_agent的middlewire.py —— 中间件（生命周期钩子）

```python
from langchain.agents.middleware import (before_agent, after_agent,
    before_model, after_model, wrap_model_call, wrap_tool_call)

@before_agent     # Agent 启动前
def log_before_agent(state, runtime): ...

@after_agent      # Agent 结束后
def log_after_agent(state, runtime): ...

@before_model     # 模型调用前
def log_before_model(state, runtime): ...

@after_model      # 模型调用后
def log_after_model(state, runtime): ...

@wrap_model_call  # 包装模型调用（可拦截/改写）
def model_call_hook(request, handler):
    return handler(request)

@wrap_tool_call   # 包装工具调用（可监控/拦截）
def tool_call_hook(request, handler):
    return handler(request)

agent = create_agent(..., middleware=[...六个钩子...])
```

**要点**：中间件是 Agent 的可观测性与控制面——`before/after` 系列做生命周期日志，`wrap` 系列可以包装调用（记录参数、拦截、改写返回值）。项目「扫地机器人智能客服」中的工具监控与动态提示词切换即基于此机制实现。

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- 至少一个可用的大模型 API Key（默认使用阿里云百炼的 `qwen-turbo`）

### 2. 安装依赖

```bash
pip install langchain langchain-core langchain-openai langgraph python-dotenv
```

### 3. 配置环境变量

在 `agent基础` 目录下创建 `.env` 文件（`config.py` 通过 `load_dotenv()` 加载）：

```ini
# qwen-turbo（OpenAI 兼容接口）
ALIBL_API_KEY=你的_API_Key
# 例如 https://dashscope.aliyuncs.com/compatible-mode/v1
ALIBL_BSUL=你的_base_url

# 以下为可选，按需配置（config.py 已预留，示例未全部使用）
# DEEPSEEK_API_KEY=...
# DEEPSEEK_BSUL=...
# ZHIPU_API_KEY=...
# TAVILY_API_KEY=...
# DASHSCOPE_API_KEY=...
```

### 4. 运行示例

```bash
python 001_Agent初体验.py
python 002_agent流式输出.py
python 003_agent的React框架.py
python 004_agent的middlewire.py
```

每个脚本都是自包含的独立示例，直接运行即可在终端看到 Agent 的思考过程、工具调用与最终回答。

---

## 常见问题

- **为什么运行 002/003 时看不到「工具调用」打印？** 只有模型决定调用工具的那一轮才会打印；若问题简单到无需工具，模型会直接回答。
- **如何换模型？** 修改 `config.py` 中的 `CHAT_MODEL`、`ALIBL_API_KEY`、`ALIBL_BSUL`，或改用 DeepSeek / 智谱等兼容接口的 Key。
- **中间件可以做什么？** 日志记录、调用统计、参数校验、结果缓存、动态切换提示词等（参见 004 及实战项目「扫地机器人智能客服」）。

---

## 已知限制与注意事项

- 示例中的工具均为**模拟实现**（固定返回值），用于演示调用链路，不代表真实数据。
- `002/003` 中通过 `try/except` 包裹 `tool_calls` 访问，不同版本 LangChain 的消息对象结构可能略有差异，属兼容性写法。
- `004` 中 `AgentState` 使用 `state["messages"]` 访问消息列表，不同版本字段命名可能变化，请以实际版本为准。

---

## License

本项目为学习示例代码，仅用于教学演示。
