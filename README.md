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

- **[06_parallelism_map_reduce.py](tutorials/06_parallelism_map_reduce.py)**
  - 并行处理：Map-Reduce 模式。
  - 核心概念：`Send` API 实现动态并行分支 (Map)、`operator.add` 聚合器 (Reduce)、并发状态管理。

- **[07_hybrid_subgraphs.py](tutorials/07_hybrid_subgraphs.py)**
  - 混合架构：父子图 (Subgraphs) 嵌套。
  - 核心概念：将不同架构（如 ReAct 和 Map-Reduce）封装为独立子图、层级状态管理、复杂工作流的模块化复用。

- **[08_multi_agent_supervisor](tutorials/)**
  - 多智能体协作：Supervisor (主管) 模式。
  - 核心概念：中心化路由控制、结构化输出做决策。
  - 变体：
    - **[Chat Mode](tutorials/08_multi_agent_supervisor_chat.py)**: 基于对话历史的协作。
    - **[Artifact Mode](tutorials/08_multi_agent_supervisor_artifact.py)**: 围绕特定工件 (Artifact) 的迭代优化。

- **[09_multi_agent_handoff.py](tutorials/09_multi_agent_handoff.py)**
  - 多智能体接力：Handoff (Swarm) 模式。
  - 核心概念：`Command` API 实现命令式跳转、去中心化控制、Agent 之间显式交接棒 (Context Passing)。

- **[10_plan_and_execute.py](tutorials/10_plan_and_execute.py)**
  - 规划与执行：Plan-and-Execute 模式。
  - 核心概念：Planner (规划)、Executor (执行)、Re-Planner (反思与动态调整) 的闭环循环，处理长链路复杂任务。
