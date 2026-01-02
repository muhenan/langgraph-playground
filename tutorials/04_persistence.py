import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# 使用你自定义的可视化工具
from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# 1. State & Tools
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a specific city."""
    return f"The weather in {city} is sunny and 25°C."

tools = [multiply, get_weather]

# ==========================================
# 2. Nodes & Model
# ==========================================
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ==========================================
# 3. Main Logic
# ==========================================
def main():
    # --- A. Setup Database ---
    # check_same_thread=False 允许在多线程环境下使用 SQLite
    conn = sqlite3.connect("tutorials/checkpoints/checkpoints.sqlite", check_same_thread=False)

    # --- B. Build Graph & Inject Checkpointer ---
    # 使用 Context Manager 自动管理连接生命周期
    # with SqliteSaver(conn) as checkpointer:
    checkpointer = SqliteSaver(conn)

    # ==========================================
    # LangGraph Persistence 核心机制笔记:
    # 
    # 1. 设计哲学:
    #    LangGraph 将 Graph 视为 "无状态计算引擎"，将 State 视为 "外部依赖"。
    #    Graph 定义了逻辑结构，而 Checkpointer 负责管理时间轴上的状态快照。
    #
    # 2. 状态维护 (Memory vs Persistence):
    #    - 在内存中 (Runtime): 在一次 invoke 执行期间，内存中必须维护完整的 State (如 messages 列表)，
    #      以便 LLM 理解上下文。
    #    - 跨请求时 (Between Invokes): Graph 对象本身不持久化状态。每次新的 invoke 调用 (带 thread_id)，
    #      系统会从数据库中 "重新水合 (Rehydrate)" 恢复之前的状态，然后继续执行。
    #
    # 3. 写入时机 (Checkpointing):
    #    - 并不是每产生一条消息就写一次 IO，而是在每一个 "Superstep (超步)" 结束时写入。
    #    - 流程: Node 执行 -> 返回 Update -> Reducer 合并状态 (如 add_messages) -> Checkpointer.put 写入数据库。
    #    - 这确保了每一步执行后的状态都是可恢复的 (Time Travel 基础)。
    # 
    # ------------------------------------------
    # 生产环境高并发架构思考 (10k+ QPS):
    # 
    # 简单的同步 DB 读写在高并发下会导致延迟增加和连接池耗尽。生产方案通常采用多级缓存架构：
    # 
    # A. 方案一：纯 Redis (如果数据量可控)
    #    - 仅使用 Redis 存储 Checkpoint，放弃传统关系型数据库。速度极快，适合短期记忆。
    #
    # B. 方案二：冷热分离 + 异步落库 (Write-Behind Pattern)
    #    - Hot Layer (Redis): Agent 运行时只读写 Redis，保证低延迟。
    #    - Buffer Layer (MQ): 更新 Redis 同时发送消息到 Kafka/RabbitMQ。
    #    - Cold Layer (DB/S3): Worker 异步消费 MQ，批量写入 Postgres 或 S3 永久存储。
    #    - 优点: 用户侧无延迟感知，数据库压力被平滑削峰。
    #
    # C. 方案三：Sticky Sessions + 本地内存 (Stateful Actor)
    #    - 终极高并发方案 (如游戏服/高频交易)。
    #    - 路由: LB 根据 thread_id 哈希，确保同用户请求总是打到同一个 Pod。
    #    - 内存: Agent 直接在进程内存中维护 State (0ms 读写)。
    #    - 持久化: 仅在对话闲置或定时间隔时，异步 Snapshot 到数据库。
    #    - 结合技术: Ray (Python 的分布式计算框架) 或 Orleans (Actor 模型)。
    # ==========================================

    builder = StateGraph(AgentState)
    
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")

    # [关键]: 编译时传入 checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    # --- C. Visualization ---
    visualize_graph(graph, "04_graph_structure.png")

    # --- D. Simulation: Thread Isolation ---
    
    # 场景 1: 用户 Neo (建立记忆)
    config_neo = {"configurable": {"thread_id": f"user_neo_{uuid.uuid4()}"}}
    
    # 第一轮对话
    print(f"\n=== Session: user_neo ===")
    print("User: Hi, I'm Neo.")
    result_neo = graph.invoke(
        {"messages": [HumanMessage(content="Hi, I'm Neo.")]}, 
        config=config_neo
    )
    print(f"Agent: {result_neo['messages'][-1].content}")

    # 第二轮对话
    print("\nUser: What is my name?")
    result_neo = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]}, 
        config=config_neo
    )
    print(f"Agent: {result_neo['messages'][-1].content}")

if __name__ == "__main__":
    main()
