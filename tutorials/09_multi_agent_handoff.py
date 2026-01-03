# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv"
# ]
# ///

import uuid
from typing import TypedDict, Annotated, List, Literal, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# [核心]: 只需要 Command
from langgraph.types import Command

from utils.visualizer import visualize_graph

load_dotenv()

"""
Handoff (接力)
"""

# ==========================================
# 1. State Definitions
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    active_agent: str # 记录当前拿着"接力棒"的人

# ==========================================
# 2. Decision Schemas (替代 Tools)
# ==========================================
# 让 LLM 直接做选择题，而不是调用工具

class Response(BaseModel):
    """每个 Agent 的标准输出结构"""
    response_text: str = Field(description="The response to show to the user.")
    next_step: Literal["reply_to_user", "transfer"] = Field(
        description="Choose 'transfer' if you need to handoff to another agent, otherwise 'reply_to_user'."
    )
    transfer_target: Optional[Literal["triage", "tech_support"]] = Field(
        description="Only required if next_step is 'transfer'. The name of the agent to transfer to.",
        default=None
    )

# ==========================================
# 3. Agents (Nodes)
# ==========================================
# 这里 4.1 nano 和 4o mini 都有幻觉，最后回不到前台
# 4o 最后会回到前台
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

def triage_node(state: AgentState):
    """前台接待：处理账单，转接技术"""
    print("  -> [Triage] Processing...")
    
    # 1. 构造 Prompt
    system_msg = SystemMessage(content=(
        "You are 'Triage', the front desk support. You handle billing questions. "
        "If the user has a technical issue (code, bugs), you MUST transfer to 'tech_support'. "
        "Otherwise, reply to the user yourself."
    ))
    messages = [system_msg] + state["messages"]
    
    # 2. 获取结构化决策 (不涉及工具调用，纯逻辑判断)
    structured_llm = llm.with_structured_output(Response)
    decision = structured_llm.invoke(messages)
    
    # 3. 生成回复消息
    ai_msg = AIMessage(content=f"[Triage]: {decision.response_text}")
    
    # ============================================================
    # 4. Command Logic (核心跳转)
    # ============================================================
    if decision.next_step == "transfer" and decision.transfer_target:
        target = decision.transfer_target
        print(f"  🔄 Handoff: Triage -> {target}")
        
        # 注入一条系统消息，告诉下一棒发生了什么 (Context Passing)
        system_notice = SystemMessage(content=f"SYSTEM: User transferred from Triage.")
        
        return Command(
            goto=target, # 立即跳到目标节点
            update={
                "messages": [ai_msg, system_notice],
                "active_agent": target # 关键：交出接力棒
            }
        )
    
    # 5. 常规回复 -> 结束这一轮，等待用户
    return Command(
        goto=END,
        update={"messages": [ai_msg]}
    )


def tech_support_node(state: AgentState):
    """技术支持：修 Bug，修好转回前台"""
    print("  -> [Tech Support] Debugging...")
    
    system_msg = SystemMessage(content=(
        "You are 'Tech Support'. You solve coding issues. "
        "When the issue is resolved, or if the user asks about billing, transfer back to 'triage'. "
        "Otherwise, reply to the user."
    ))
    messages = [system_msg] + state["messages"]
    
    structured_llm = llm.with_structured_output(Response)
    decision = structured_llm.invoke(messages)
    
    ai_msg = AIMessage(content=f"[Tech]: {decision.response_text}")
    
    # Command Logic
    if decision.next_step == "transfer" and decision.transfer_target:
        target = decision.transfer_target
        print(f"  🔄 Handoff: Tech Support -> {target}")
        
        system_notice = SystemMessage(content=f"SYSTEM: User transferred from Tech Support.")
        
        return Command(
            goto=target,
            update={
                "messages": [ai_msg, system_notice],
                "active_agent": target
            }
        )

    return Command(
        goto=END,
        update={"messages": [ai_msg]}
    )

# ==========================================
# 4. Entry Router (不变)
# ==========================================
def entry_router(state: AgentState):
    """
    根据谁拿着接力棒，决定把用户的消息送给谁
    
    法则 1：入口路由 (The Entry Router)
    问： "问每个问题的时候都必须走 entry point 对吗？"
    答案：是的。
    这就是 "Persistent State" (持久化状态) 的作用。 每次用户输入新消息，对于程序来说都是一次全新的启动。
    程序启动时，必须从 START 节点开始。
    
    机制：Router 醒来后的第一件事，就是去读取内存（Memory）里的 active_agent 字段。
    - 如果 active_agent="tech_support"，Router 说：“哦，上次是 Tech 在服务，那我把这通电话直接转给 Tech。”
    - 日志证据：--- Router: Dispatching to tech_support ---，此时 Triage 根本不会被唤醒。
    """
    active = state.get("active_agent", "triage") # 默认给前台
    print(f"--- Router: Dispatching to {active} ---")
    return Command(goto=active)

# ==========================================
# 5. Graph Construction
# ==========================================
def main():
    """
    总结：这种架构叫 "Star Graph" (星型图)
    在这个架构下，图结构非常简单：
    1. 中心是 Entry Router。
    2. 周围是 Agents (Triage, Tech, Sales...)。
    3. Edge: 只有 START -> Router 这一条显式的边。
    4. Jumps: 所有的 Agent 都可以通过 Command 任意跳到其他 Agent（网状跳转），而不需要在图里画几十条线。

    这就是构建 OpenAI Swarm 风格多智能体系统的标准范式！
    
    法则 2：命令式跳转 (Command-based Handoff)
    问： "节点之间不用条件边，而是直接用 command goto 跳转？"
    答案：是的。这正是 Command API 取代 Conditional Edges 的地方。
    
    - 旧模式 (Conditional Edge): Node A 跑完 -> Graph 运行路由函数 -> 决定去 Node B。逻辑分散。
    - 新模式 (Command goto): Node A 跑完 -> 直接扔出一张 "传送卡" (Command(goto="B")) -> 立即传送。逻辑内聚，Node A 拥有完全自主权。
    """
    builder = StateGraph(AgentState)
    
    builder.add_node("entry", entry_router)
    builder.add_node("triage", triage_node)
    builder.add_node("tech_support", tech_support_node)
    
    # 只有这一个显式的 Edge
    builder.add_edge(START, "entry")
    
    # 注意：我们完全删除了所有的 add_edge(node, node)
    # 所有的连接都是隐式的，由 Command 动态生成
    
    graph = builder.compile(checkpointer=MemorySaver())
    visualize_graph(graph, "09_handoff_simple.png")

    # ==========================================
    # 6. Execution
    # ==========================================
    config = {"configurable": {"thread_id": "handoff_simple_01"}}
    
    user_inputs = [
        "Hi, I have a billing question about my invoice.", # Triage 接
        "Wait, actually my code is throwing a SegFault.",  # Triage -> Tech
        "Can you fix it?",                               # Tech 接
        "Thanks! Can you check my bill now?",            # Tech -> Triage
    ]
    
    for i, txt in enumerate(user_inputs):
        print(f"\n🗣️  User ({i+1}): {txt}")
        
        # 运行图
        for event in graph.stream({"messages": [HumanMessage(content=txt)]}, config=config):
            
            for node_name, update in event.items():
                
                # [关键修复]: 防御性编程
                # entry 节点返回 Command(goto=...) 但没有 update，所以 update 是 None
                if update is None:
                    if node_name == "entry":
                        print(f"  🚦 [Entry Router]: Switching line...")
                    else:
                        print(f"  ⏩ [{node_name}]: Pure jump (No state update)")
                    continue

                # --- 1. 捕捉 Handoff (控制权转移) ---
                if "active_agent" in update:
                    new_agent = update["active_agent"]
                    print(f"  🔄 [System]: Handoff triggered! Control moving to -> {new_agent.upper()}")

                # --- 2. 捕捉 Agent 回复 ---
                if "messages" in update:
                    last_msg = update["messages"][-1]
                    
                    if isinstance(last_msg, AIMessage):
                        icon = "🤖"
                        if node_name == "triage":
                            icon = "🛎️ "
                        elif node_name == "tech_support":
                            icon = "🛠️ "
                            
                        print(f"  {icon} [{node_name}]: {last_msg.content}")
                    
                    elif isinstance(last_msg, SystemMessage) and "SYSTEM:" in last_msg.content:
                        print(f"     [Context Injection]: {last_msg.content}")
             
    # 打印最终历史
    print_audit_log(graph, config)

def print_audit_log(graph, config):
    print("\n" + "="*60)
    print("📜  FULL HISTORY")
    print("="*60)
    state = graph.get_state(config)
    for msg in state.values.get("messages", []):
        if isinstance(msg, AIMessage):
            print(f"🤖 {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"👤 {msg.content}")
        elif isinstance(msg, SystemMessage):
            print(f"⚙️ {msg.content}")
    print("-" * 60)
    print(f"Final Active Agent: {state.values.get('active_agent')}")

if __name__ == "__main__":
    main()