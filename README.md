# LangGraph Playground

这是一个用来学习 [LangGraph](https://langchain-ai.github.io/langgraph/) 的项目。

我会在这里使用 LangGraph 编写一些 AI Agent 示例和项目，用于探索和实践 Agentic Workflow。

## 📚 Tutorials

这里是一系列循序渐进的教程，帮助你理解 LangGraph 的核心概念：

- **[01_state_and_nodes.py](tutorials/01_state_and_nodes.py)**
  - 基础入门：介绍 `StateGraph` 的构建。
  - 核心概念：`State` 定义 (TypedDict)、简单节点 (Nodes) 的编写、线性图结构。
  
- **[02_edges_and_routing.py](tutorials/02_edges_and_routing.py)**
  - 路由控制：介绍 `Conditional Edges` (条件边)。
  - 核心概念：Router 逻辑编写、根据 State 动态决定下一步走向 (分支逻辑)。

- **[03_tool_calling.py](tutorials/03_tool_calling.py)**
  - 工具调用：结合 LLM 进行 Tool Calling。
  - 核心概念：`bind_tools`、`ToolNode`、`tools_condition` 以及如何流式输出 (Streaming) 运行状态。

- **[04_persistence.py](tutorials/04_persistence.py)**
  - 记忆持久化：让 Agent 拥有"记忆"。
  - 核心概念：Checkpointer (`SqliteSaver`)、`thread_id` 会话管理、跨请求的状态恢复与隔离。

- **[05_human_in_the_loop.py](tutorials/05_human_in_the_loop.py)**
  - 人机交互 (HITL)：在 Agent 执行过程中加入人工干预。
  - 核心概念：`interrupt_before` 断点机制、人工审批/拒绝/修改工具调用、图的暂停与恢复 (Resuming)。
