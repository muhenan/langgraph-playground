# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv"
# ]
# ///

import operator
from typing import TypedDict, Annotated, List, Literal, Union, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# 1. State Definitions (Artifact-Centric)
# ==========================================

class AgentState(TypedDict):
    # --- 核心工单字段 (Single Source of Truth) ---
    request: str                # 原始需求
    code: str                   # 当前版本的代码
    review: str                 # 当前版本的评审意见
    revision_number: int        # 迭代次数 (防止死循环)
    
    # --- 控制流字段 ---
    next: str                   # 下一步是谁
    
    # --- 审计日志字段 (仅供人类阅读，Worker 不依赖此字段干活) ---
    messages: Annotated[List[BaseMessage], add_messages]

# ==========================================
# 2. Worker Nodes (Stateless & Focused)
# ==========================================
llm_worker = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

def coder_node(state: AgentState):
    """
    Coder 专注于写代码。
    输入: request, review, code (历史)
    输出: code (新), revision_number (+1)
    """
    request = state["request"]
    review = state.get("review", "No feedback yet.")
    current_code = state.get("code", "")
    revision = state.get("revision_number", 0)
    
    print(f"  -> [Coder] Coding (Revision {revision + 1})...")
    
    # [Prompt Engineering]: 构建完形填空式的 Prompt
    prompt = f"""
    ROLE: You are an expert Python programmer.
    
    GOAL: Write Python code to satisfy this request: "{request}"
    
    PREVIOUS CODE (If any):
    {current_code}
    
    REVIEWER FEEDBACK (If any):
    {review}
    
    INSTRUCTIONS:
    1. If this is the first run, write the code from scratch.
    2. If there is feedback, FIX the issues in the previous code.
    3. Output ONLY the code inside a markdown block (```python ... ```).
    4. Do NOT include any explanations or review. Just the code.
    """
    
    response = llm_worker.invoke([HumanMessage(content=prompt)])
    new_code = response.content
    
    return {
        "code": new_code,
        "revision_number": revision + 1,
        # 记录到审计日志
        "messages": [AIMessage(content=f"[Coder]: Code generated (Rev {revision + 1})")]
    }

def reviewer_node(state: AgentState):
    """
    Reviewer 专注于找茬。
    输入: request, code
    输出: review
    """
    request = state["request"]
    code = state.get("code", "")
    
    print(f"  -> [Reviewer] Reviewing code...")
    
    prompt = f"""
    ROLE: You are a strict Code Reviewer.
    
    GOAL: Verify if the code satisfies the request: "{request}"
    
    CODE TO REVIEW:
    {code}
    
    INSTRUCTIONS:
    1. Check for syntax errors, logical bugs, and security issues.
    2. Check if it meets the user request.
    3. If the code is perfect and ready for production, output EXACTLY: "LGTM" (Looks Good To Me).
    4. If there are issues, list them clearly and concisely.
    """
    
    response = llm_worker.invoke([HumanMessage(content=prompt)])
    review_content = response.content
    
    return {
        "review": review_content,
        # 记录到审计日志
        "messages": [AIMessage(content=f"[Reviewer]: {review_content}")]
    }

# ==========================================
# 3. Supervisor Node (Logic-Based Router)
# ==========================================
def supervisor_node(state: AgentState):
    """
    主管节点：基于工单状态(State)进行路由
    """
    review = state.get("review", "")
    revision = state.get("revision_number", 0)
    
    # 1. 判断是否完成 (Termination Condition)
    if "LGTM" in review:
        decision = "FINISH"
        log_msg = "✅ [Supervisor]: Code approved (LGTM). Task completed."
    
    # 2. 判断是否超限 (Safety Guard)
    elif revision >= 6:
        decision = "FINISH"
        log_msg = "⚠️ [Supervisor]: Max revisions reached. Stopping to prevent infinite loop."
    
    # 3. 路由逻辑 (Routing Logic)
    else:
        # 如果最近一次是由 Reviewer 发言（且不是 LGTM），那肯定得让 Coder 改
        # 如果是刚开始（revision=0）或者刚写完没 review，这里我们需要一个状态机逻辑
        # 我们可以简单地通过"谁刚更新了状态"来推断，或者由图的结构决定。
        # 在这个图中，Coder -> Supervisor -> Reviewer -> Supervisor -> Coder 是一个固定循环
        # 但为了灵活性，我们在这里显式判断：
        
        # 这里的 trick 是：我们需要知道"上一跳"是谁。
        # 简单判定：如果 code 存在，且 review 是空的(或者是上一轮的旧 review)，去 Reviewer。
        # 但因为我们每次都覆盖 update review，比较难判断是"旧"的还是"新"的。
        
        # 更简单的做法：查看 messages 里的最后一条消息是谁发的
        last_msg = state["messages"][-1] if state["messages"] else None
        
        if last_msg and "[Coder]" in last_msg.content:
            decision = "Reviewer"
            log_msg = "👉 [Supervisor]: New code detected. Assigning to Reviewer."
        elif last_msg and "[Reviewer]" in last_msg.content:
            decision = "Coder"
            log_msg = "👉 [Supervisor]: Issues found. Assigning back to Coder."
        else:
            # 默认情况 (比如刚开始)
            decision = "Coder"
            log_msg = "👉 [Supervisor]: Starting task. Assigning to Coder."

    print(f"--- [Supervisor] Decision: {decision} ---")
    
    return {
        "next": decision,
        "messages": [AIMessage(content=log_msg)]
    }

# ==========================================
# 4. Graph Construction
# ==========================================
def main():
    builder = StateGraph(AgentState)
    
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("Coder", coder_node)
    builder.add_node("Reviewer", reviewer_node)
    
    # 连接边：所有工人做完后都回汇报给主管
    builder.add_edge("Coder", "supervisor")
    builder.add_edge("Reviewer", "supervisor")
    
    # 入口
    builder.add_edge(START, "supervisor")
    
    # 动态路由
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "Coder": "Coder",
            "Reviewer": "Reviewer",
            "FINISH": END
        }
    )
    
    graph = builder.compile(checkpointer=MemorySaver())
    visualize_graph(graph, "08_supervisor_optimized_2.png")

    # ==========================================
    # 5. Execution & Real-time Logging
    # ==========================================
    config = {"configurable": {"thread_id": "optimized_dev_team"}}
    
    user_query = "Write a Python script to verify if a number is prime."
    print(f"User Request: {user_query}\n")
    
    initial_state = {
        "request": user_query,
        "revision_number": 0,
        "messages": [HumanMessage(content=user_query)]
    }
    
    print("--- Execution Started ---")
    
    for event in graph.stream(initial_state, config=config, recursion_limit=15):
        for node_name, state_update in event.items():
            
            # 打印 Supervisor 的决策
            if node_name == "supervisor":
                next_step = state_update.get("next")
                print(f"👀 [Supervisor] -> {next_step}")
            
            # 打印 Coder 的产出 (只显示 code 预览)
            elif node_name == "Coder":
                code_snippet = state_update.get("code", "")[:1000].replace("\n", " ")
                print(f"💻 [Coder] generated code: {code_snippet}...")
            
            # 打印 Reviewer 的产出
            elif node_name == "Reviewer":
                review_snippet = state_update.get("review", "")[:1000].replace("\n", " ")
                print(f"🔍 [Reviewer] feedback: {review_snippet}...")

    print("--- Execution Finished ---\n")
    
    # 打印最终结果
    final_state = graph.get_state(config).values
    if "code" in final_state:
        print("Final Code Artifact:")
        print(final_state["code"])
    
    print_audit_log(graph, config)

def print_audit_log(graph, config):
    print("\n" + "="*60)
    print("📜  FULL CONVERSATION HISTORY (AUDIT LOG)")
    print("="*60)
    final_snapshot = graph.get_state(config)
    
    for msg in final_snapshot.values.get("messages", []):
        icon = "❓"
        role = "Unknown"
        content = msg.content
        
        if isinstance(msg, HumanMessage):
            icon = "👤"
            role = "User"
        elif isinstance(msg, AIMessage):
            if "[Coder]" in content:
                icon = "💻"
                role = "Coder"
                # Coder 的消息我们只记录了一个占位符，如果想看代码，可以打印 state['code']
                # 但为了日志整洁，这里保持原样
            elif "[Reviewer]" in content:
                icon = "🔍"
                role = "Reviewer"
                # 如果内容太长，截断显示
                if len(content) > 1000: content = content[:1000] + "..."
            elif "[Supervisor]" in content:
                icon = "👮"
                role = "Supervisor"
            else:
                icon = "🤖"
                role = "AI"
        
        print(f"{icon}  [{role}]: {content}")
        print("-" * 60)

if __name__ == "__main__":
    main()