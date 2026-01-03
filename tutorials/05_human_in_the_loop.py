# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv"
# ]
# ///

import uuid
import sys
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# 1. State & Tools
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def buy_stock(ticker: str, amount: int) -> str:
    """Executes a stock purchase trade."""
    return f"✅ SUCCESS: Bought {amount} shares of {ticker}."

tools = [buy_stock]

# ==========================================
# 2. Nodes
# ==========================================
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 这里 llm 会看到整个对话历史，包括工具调用和人类干预。
def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ==========================================
# 3. Helper Functions
# ==========================================
def resume_graph(graph, config, input_payload=None):
    """恢复图运行并打印流式日志"""
    print("\n[System] Resuming Graph Execution...")
    for event in graph.stream(input_payload, config=config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"🤖 Agent Thought: {last_msg.content}")
            elif isinstance(last_msg, ToolMessage):
                print(f"🛠️  Tool Output: {last_msg.content}")

def print_audit_log(graph, config):
    """打印完整的对话历史审计日志"""
    print("\n" + "="*60)
    print("📜  FULL CONVERSATION HISTORY (AUDIT LOG)")
    print("="*60)

    final_snapshot = graph.get_state(config)
    all_messages = final_snapshot.values.get("messages", [])

    for i, msg in enumerate(all_messages):
        role = "UNKNOWN"
        icon = "❓"
        if isinstance(msg, HumanMessage):
            role, icon = "HUMAN", "👤"
        elif isinstance(msg, AIMessage):
            role, icon = "AI   ", "🤖"
        elif isinstance(msg, ToolMessage):
            role, icon = "TOOL ", "🛠️"

        content = msg.content
        extra_info = ""

        if isinstance(msg, AIMessage) and msg.tool_calls:
            call_details = [f"{c['name']}{c['args']}" for c in msg.tool_calls]
            extra_info = f"\n   >>> [Tool Request]: {', '.join(call_details)}"
        
        print(f"{icon}  [{role}]: {content}{extra_info}")
        print("-" * 60)

# ==========================================
# 4. Graph Setup
# ==========================================
def build_graph():
    checkpointer = MemorySaver()
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("action", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "action", END: END})
    builder.add_edge("action", "agent")
    
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

# ==========================================
# 5. Main Logic
# ==========================================
def main():
    graph = build_graph()
    visualize_graph(graph, "05_hitl_structure.png")

    config = {"configurable": {"thread_id": "demo_5_strategies"}}
    
    print("\n=== User Request ===")
    user_query = "Please buy 100 shares of Apple (AAPL)."
    print(f"User: {user_query}")

    # --- 初始运行 ---
    resume_graph(graph, config, {"messages": [HumanMessage(content=user_query)]})

    # --- 交互循环 ---
    while True:
        snapshot = graph.get_state(config)
        
        # [退出条件]
        if not snapshot.next:
            print("\n✅ [System] Process Completed.")
            break

        # [断点检测]
        if snapshot.next[0] == "action":
            last_message = snapshot.values["messages"][-1]
            tool_call = last_message.tool_calls[0]
            ticker = tool_call["args"].get("ticker")
            amount = tool_call["args"].get("amount")

            print(f"\n" + "!"*60)
            print(f"⚠️  INTERRUPT: Agent wants to buy {amount} shares of {ticker}")
            print("!"*60)
            print("1. Approve (Execute directly)")
            print("2. Reject (Inject failure message)")
            print("3. Natural Feedback (Append 'Admin Instruction', let AI reason)")
            print("-" * 60)

            choice = input("Select strategy (1-3): ").strip()
            
            if choice == "1":
                print("\n✅ [Strategy 1: Approve]")
                resume_graph(graph, config, None)

            elif choice == "2":
                print("\n🚫 [Strategy 2: Reject]")
                rejection_msg = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="❌ Transaction rejected by user.",
                    name=tool_call["name"]
                )
                graph.update_state(config, {"messages": [rejection_msg]}, as_node="action")
                resume_graph(graph, config, None)

            # 很多应用中都没有这种情况，例如 cursor 提供代码后，用户只能接受或拒绝，不能修改代码。
            elif choice == "3":
                print("\n🗣️ [Strategy 3: Natural Feedback]")
                
                new_amount = amount // 2
                
                # 1. 第一条：闭合 API 环 (ToolMessage)
                # 告诉 LLM：这个工具调用在技术上被拦截/取消了
                technical_msg = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"Transaction cancelled by admin intervention.", # 中性的技术反馈
                    name=tool_call["name"]
                )
                
                # 2. 第二条：真正的指令 (HumanMessage)
                # 这才是带有强烈权重的"人类命令"
                human_instruction = HumanMessage(
                    content=f"SYSTEM ADMIN ALERT: The amount {amount} is too high. I want you to buy exactly {new_amount} shares instead."
                )
                
                # 3. 同时注入两条消息
                # as_node="action": 假装这些是在 action 阶段发生的
                graph.update_state(
                    config, 
                    {"messages": [technical_msg, human_instruction]}, 
                    as_node="action"
                )
                
                resume_graph(graph, config, None)

            else:
                print("Invalid choice.")

    # --- 最终审计 ---
    print_audit_log(graph, config)

if __name__ == "__main__":
    main()