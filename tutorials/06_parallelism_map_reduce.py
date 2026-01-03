# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv"
# ]
# ///

import operator
import uuid
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# 1. State Definitions
# ==========================================

# 全局状态
class OverallState(TypedDict):
    topic: str
    subjects: List[str]
    jokes: Annotated[List[str], operator.add]
    final_report: str
    # [新增]: 为了支持 Audit Log，我们加入 messages 列表
    # 使用 add_messages reducer，LangGraph 会自动处理并发写入时的合并
    messages: Annotated[List[BaseMessage], add_messages]

# Graph 在建立之初并不知道 WorkerState，这只是一个局部使用的状态
class WorkerState(TypedDict):
    section_subject: str

# ==========================================
# 2. Models & Nodes
# ==========================================

# 这里的 llms 都只看到当前的输入，不会看到历史记录。
llm_planner = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

class Subjects(BaseModel):
    subjects: List[str] = Field(description="A list of subjects to generate jokes about")

def planner_node(state: OverallState):
    print(f"--- [Planner] Analyzing topic: {state['topic']} ---")
    
    structured_llm = llm_planner.with_structured_output(Subjects)
    structured_result = structured_llm.invoke(f"Extract subjects from: {state['topic']}")
    
    # [Action]: 返回 subjects 数据，同时记录一条 AIMessage 到历史记录
    return {
        "subjects": structured_result.subjects,
        "messages": [AIMessage(content=f"PLANNER: I have split the task into: {structured_result.subjects}")]
    }

llm_joke = ChatOpenAI(model="gpt-4.1-nano", temperature=0.9)

def generation_node(state: WorkerState):
    subject = state["section_subject"]
    print(f"  -> [Worker] Processing: {subject}")
    
    response = llm_joke.invoke(f"Tell me a one-sentence joke about {subject}.")
    joke = f"{subject.upper()}: {response.content}"
    
    # [Action]: 返回 joke 数据，同时记录一条 AIMessage
    # 注意：在并发执行时，多个 Worker 会同时往 messages 里 append
    return {
        "jokes": [joke],
        "messages": [AIMessage(content=f"WORKER({subject}): Generated joke -> {response.content}")]
    }

def reducer_node(state: OverallState):
    jokes = state.get("jokes", [])
    subjects = state.get("subjects", [])
    
    # [Check]: Wait for all subjects to be processed
    if len(jokes) < len(subjects):
        return {}
    
    # [Check]: Avoid duplicate execution (Idempotency)
    if state.get("final_report"):
        return {}

    print("--- [Reducer] Combining results ---")
    summary = "\n".join(jokes)
    
    final_msg = f"Here is the collected humor report:\n\n{summary}"
    
    return {
        "final_report": final_msg,
        "messages": [AIMessage(content="REDUCER: All tasks finished. Report compiled.")]
    }

# ==========================================
# 3. Logic & Helper
# ==========================================
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"section_subject": s}) for s in state["subjects"]]

# --- 你的审计函数 ---
def print_audit_log(graph, config):
    """打印完整的对话历史审计日志"""
    print("\n" + "="*60)
    print("📜  FULL CONVERSATION HISTORY (AUDIT LOG)")
    print("="*60)

    # 从 MemorySaver 中读取最终状态
    final_snapshot = graph.get_state(config)
    all_messages = final_snapshot.values.get("messages", [])

    for i, msg in enumerate(all_messages):
        role = "UNKNOWN"
        icon = "❓"
        if isinstance(msg, HumanMessage):
            role, icon = "HUMAN", "👤"
        elif isinstance(msg, AIMessage):
            # 这里我们通过 content 前缀来区分是 Planner 还是 Worker
            role, icon = "AI   ", "🤖"
        elif isinstance(msg, ToolMessage):
            role, icon = "TOOL ", "🛠️"

        content = msg.content
        print(f"{icon}  [{role}]: {content}")
        print("-" * 60)

# ==========================================
# 4. Graph Construction
# ==========================================
def main():
    # [Setup]: 使用 MemorySaver
    checkpointer = MemorySaver()
    
    # 这里定义了 OverallState 作为 StateGraph 的 State 类型，类似于一个全局状态。
    builder = StateGraph(OverallState)
    builder.add_node("planner", planner_node)
    builder.add_node("generate_joke", generation_node)
    builder.add_node("reduce", reducer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", continue_to_jokes, ["generate_joke"])
    # [Changed]: Directly connect to reducer, let reducer handle synchronization
    builder.add_edge("generate_joke", "reduce")
    builder.add_edge("reduce", END)

    # 全图共享内存 checkpointer，用于在多个节点之间共享状态。
    graph = builder.compile(checkpointer=checkpointer)
    
    visualize_graph(graph, "06_map_reduce.png")

    # ==========================================
    # 5. Execution
    # ==========================================
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_input = "Tell me jokes about basketball, dogs, and python."
    
    print(f"User Request: {user_input}\n")

    # 初始输入也要包含 messages，以便 Audit Log 能显示 User Request
    initial_state = {
        "topic": user_input,
        "messages": [HumanMessage(content=user_input)]
    }

    # 运行流
    # 注意：Map-Reduce 可能会产生很多 steps，stream_mode="updates" 可以看到每一步谁完成了
    for event in graph.stream(initial_state, config=config):
        # 这里我们只打印简单的进度点，详细的看 Audit Log
        for key, value in event.items():
            if key == "generate_joke":
                # 尝试从 message 里提取 info，或者直接打印
                print(f" ✅ Worker finished a task.")

    # 打印最终报告
    final_state = graph.get_state(config).values
    if "final_report" in final_state:
        print("\n" + "="*40)
        print(f"FINAL REPORT:\n{final_state['final_report']}")
    else:
        print("\n" + "="*40)
        print("FINAL REPORT: Not generated.")

    # 打印审计日志
    print_audit_log(graph, config)

if __name__ == "__main__":
    main()