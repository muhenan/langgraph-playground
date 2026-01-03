# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv"
# ]
# ///

import operator
from typing import TypedDict, Annotated, List, Literal, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# 1. State Definitions
# ==========================================

# 我们需要一个字段来存储"下一个是谁"
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # 一长串聊天记录 Chat 模式
    next: str

# ==========================================
# 2. The Supervisor (The Brain)
# ==========================================
llm_supervisor = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# 定义 Supervisor 的可选输出
# 这就是"路由表"，LLM 必须从中选一个
members = ["Coder", "Reviewer"]
options = ["FINISH"] + members

# 使用结构化输出强制 LLM 做选择
class RouteResponse(BaseModel):
    next: Literal["FINISH", "Coder", "Reviewer"] = Field(
        description="Who should act next? Select 'FINISH' if the task is complete."
    )

def supervisor_node(state: AgentState):
    """
    主管节点：只负责观察历史，决定下一步去哪里。
    """
    print("--- [Supervisor] Thinking... ---")
    
    system_prompt = (
        "You are a supervisor managing a conversation between the following workers: "
        f"{members}. Given the following user request, respond with the worker to act next. "
        "Each worker will perform a task and respond with their results and status. "
        "When finished, respond with FINISH."
    )
    
    # 构造 Prompt，包含完整的历史记录
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 使用 with_structured_output 强制输出 JSON
    structured_llm = llm_supervisor.with_structured_output(RouteResponse)
    response = structured_llm.invoke(messages)
    
    next_agent = response.next
    print(f"--- [Supervisor] Route -> {next_agent} ---")
    
    # 我们不往 messages 里写 Supervisor 的决策过程，只更新 'next' 字段
    # 这样对话历史看起来就像是 Coder 和 Reviewer 在直接对话
    return {"next": next_agent}

# ==========================================
# 3. The Workers (Coder & Reviewer)
# ==========================================
llm_worker = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def coder_node(state: AgentState):
    """写代码的工人"""
    print("  -> [Coder] Working...")
    
    messages = [
        SystemMessage(content=(
            "You are a Python Coder. "
            "Write code to solve the user's problem. "
            "Return the code in a markdown block (```python ... ```). "
            # [核心修改]: 明确禁止它做 Review
            "IMPORTANT: Do NOT review the code yourself. Do NOT explain the code. "
            "Just output the code block. Another agent will review it."
        )),
    ] + state["messages"]
    
    response = llm_worker.invoke(messages)
    
    # 加上前缀方便审计
    return {
        "messages": [AIMessage(content=f"[Coder]: {response.content}")]
    }

def reviewer_node(state: AgentState):
    """审查代码的工人"""
    print("  -> [Reviewer] Reviewing...")
    
    messages = [
        SystemMessage(content="You are a Code Reviewer. Check the Coder's code. If it looks good, say 'LGTM'. If not, ask for changes."),
    ] + state["messages"]
    
    response = llm_worker.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"[Reviewer]: {response.content}")]
    }

# ==========================================
# 4. Graph Construction
# ==========================================
def main():
    builder = StateGraph(AgentState)
    
    # 添加节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("Coder", coder_node)
    builder.add_node("Reviewer", reviewer_node)
    
    # [关键边]: 所有工人在完成工作后，必须汇报给 Supervisor
    builder.add_edge("Coder", "supervisor")
    builder.add_edge("Reviewer", "supervisor")
    
    # [入口]: 也是先找 Supervisor
    builder.add_edge(START, "supervisor")
    
    # [条件边]: Supervisor 决定去哪里
    # 这是一个动态路由逻辑
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"], # 读取 state["next"]
        {
            "Coder": "Coder",
            "Reviewer": "Reviewer",
            "FINISH": END
        }
    )
    
    graph = builder.compile(checkpointer=MemorySaver())
    visualize_graph(graph, "08_supervisor.png")

    # ==========================================
    # 5. Execution & Audit
    # ==========================================
    config = {"configurable": {"thread_id": "team_coding"}}
    
    # 这里我们故意给一个稍微复杂的任务，让它们互动起来
    user_query = "Write a Python script to calculate Fibonacci sequence, then review it."
    print(f"User Request: {user_query}")
    
    # 运行图
    # event 的格式通常是: {'node_name': {'key': 'value'}}
    for event in graph.stream(
        {"messages": [HumanMessage(content=user_query)]}, 
        config=config, 
        recursion_limit=20
    ):
        for node_name, state_update in event.items():
            # 1. 捕获 Supervisor 的决策
            if node_name == "supervisor":
                next_step = state_update.get("next")
                print(f"👀 [Supervisor] Routing to -> {next_step}")
            
            # 2. 捕获 Coder 或 Reviewer 的输出
            elif node_name in ["Coder", "Reviewer"]:
                # 获取该节点生成的最新一条消息
                messages = state_update.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    # 为了控制台整洁，我们将换行符替换为空格，并截断过长的内容
                    content_preview = last_msg.content.replace("\n", " ")
                    if len(content_preview) > 100:
                        content_preview = content_preview[:100] + "..."
                    
                    icon = "💻" if node_name == "Coder" else "🔍"
                    print(f"{icon}  [{node_name}] Output: {content_preview}")
        
    print_audit_log(graph, config)

def print_audit_log(graph, config):
    print("\n" + "="*60)
    print("📜  FULL CONVERSATION HISTORY (AUDIT LOG)")
    print("="*60)
    final_snapshot = graph.get_state(config)
    for msg in final_snapshot.values.get("messages", []):
        icon = "👤"
        role = "User"
        if isinstance(msg, AIMessage):
            if "[Coder]" in msg.content:
                icon = "💻"
                role = "Coder"
                msg.content = msg.content.replace("[Coder]: ", "")
            elif "[Reviewer]" in msg.content:
                icon = "🔍"
                role = "Reviewer"
                msg.content = msg.content.replace("[Reviewer]: ", "")
            else:
                icon = "🤖"
                role = "AI"
        
        print(f"{icon} [{role}]: {msg.content[:500]}...") # 只打印前500字避免刷屏
        print("-" * 60)

if __name__ == "__main__":
    main()