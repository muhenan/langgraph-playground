# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv",
#     "langchain"
# ]
# ///

import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.types import Send

from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# PART 1: Child Agent A (Researcher)
# Pattern: Prebuilt ReAct (Standard Blackbox)
# ==========================================

@tool
def search_web(query: str) -> str:
    """Useful for searching tech trends."""
    print(f"    [Child A: Tool] Searching web for: {query}")
    return (
        "Recent trends in AI Agentic Workflows: "
        "1. Shift from single LLM to Multi-Agent Systems. "
        "2. Rise of 'Flow Engineering' over Prompt Engineering. "
        "3. Integration of Map-Reduce patterns for complex tasks."
    )

llm_researcher = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 直接创建一个标准的 ReAct 图
research_graph = create_agent(llm_researcher, tools=[search_web])


# ==========================================
# PART 2: Child Agent B (Writer)
# Pattern: Custom Map-Reduce (Parallelism)
# ==========================================

# --- 2.1 State Definitions for Writer ---
class WriterState(TypedDict):
    # 输入: 写作要求和参考资料
    request: str
    context: str
    # 内部: 拆分的章节列表
    sections: List[str]
    # 输出: 并行生成的草稿 (使用 add 算子合并)
    drafts: Annotated[List[str], operator.add]
    final_doc: str
    # 审计: 记录内部思考
    messages: Annotated[List[BaseMessage], add_messages]

class SectionState(TypedDict):
    section_title: str
    context: str

# --- 2.2 Nodes for Writer ---
llm_writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

class SectionSchema(BaseModel):
    sections: List[str] = Field(description="List of section titles, e.g. ['Intro', 'Deep Dive', 'Conclusion']")

def planner_node(state: WriterState):
    """拆解写作任务"""
    print(f"    [Child B: Planner] Splitting task: {state['request']}")
    structured_llm = llm_writer.with_structured_output(SectionSchema)
    result = structured_llm.invoke(f"Plan 3 short section titles for a post about: {state['request']}")
    
    return {
        "sections": result.sections,
        "messages": [AIMessage(content=f"[Writer Planner]: Split into {result.sections}")]
    }

def write_section_node(state: SectionState):
    """并行工作的写手节点"""
    title = state["section_title"]
    print(f"    [Child B: Worker] Writing section: {title}")
    
    prompt = f"Write a very short paragraph for section '{title}' based on this context: {state['context']}"
    response = llm_writer.invoke(prompt)
    
    content = f"## {title}\n{response.content}"
    # 注意：Worker 无法直接写入 WriterState 的 messages，只能返回 drafts
    # 如果想记录 worker log，需要返回 {"messages": [...]} 并在 WriterState 里处理合并
    return {"drafts": [content]}

def reducer_node(state: WriterState):
    """汇总节点"""
    print(f"    [Child B: Reducer] Compiling document...")
    full_text = "\n\n".join(state["drafts"])
    return {
        "final_doc": full_text,
        "messages": [AIMessage(content="[Writer Reducer]: Document compiled.")]
    }

# --- 2.3 Edges for Writer ---
def route_to_workers(state: WriterState):
    return [
        Send("write_section", {"section_title": s, "context": state["context"]})
        for s in state["sections"]
    ]

# 构建写手子图
writer_builder = StateGraph(WriterState)
writer_builder.add_node("planner", planner_node)
writer_builder.add_node("write_section", write_section_node)
writer_builder.add_node("reducer", reducer_node)

writer_builder.add_edge(START, "planner")
writer_builder.add_conditional_edges("planner", route_to_workers, ["write_section"])
writer_builder.add_edge("write_section", "reducer") # 所有 Worker 完工后去 Reducer
writer_builder.add_edge("reducer", END)

writing_graph = writer_builder.compile()


# ==========================================
# PART 3: Parent Agent (Editor)
# Pattern: Orchestrator
# ==========================================

class SuperGraphState(TypedDict):
    user_topic: str
    research_memo: str
    final_article: str
    messages: Annotated[List[BaseMessage], add_messages]

llm_parent = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def research_node(state: SuperGraphState):
    """调用 Child A (ReAct)"""
    print("--- [Parent] Step 1: Delegating to Researcher ---")
    
    # 1. 适配输入: ReAct Agent 需要 messages 列表
    child_input = {
        "messages": [HumanMessage(content=f"Research this topic: {state['user_topic']}")]
    }
    
    # 2. 调用子图
    result = research_graph.invoke(child_input)
    
    # 3. 提取结果
    # 这里的 result['messages'][-1] 通常是 Agent 的最终回答
    research_summary = result["messages"][-1].content
    
    # 4. 审计日志处理
    annotated_msgs = [
        AIMessage(content=f"[Subgraph Researcher]: {m.content}") 
        for m in result["messages"] if isinstance(m, AIMessage)
    ]
    
    return {
        "research_memo": research_summary,
        "messages": annotated_msgs
    }

def writing_node(state: SuperGraphState):
    """调用 Child B (Map-Reduce)"""
    print("--- [Parent] Step 2: Delegating to Writers ---")
    
    # 1. 适配输入: WriterState 需要 request 和 context
    child_input = {
        "request": state["user_topic"],
        "context": state["research_memo"]
    }
    
    # 2. 调用子图
    result = writing_graph.invoke(child_input)
    
    # 3. 提取结果
    article = result["final_doc"]
    
    # 4. 审计日志处理
    # Writer 子图的 messages 字段记录了 Planner 和 Reducer 的发言
    annotated_msgs = [
        AIMessage(content=f"[Subgraph Writer]: {m.content}") 
        for m in result["messages"] if isinstance(m, AIMessage)
    ]
    
    return {
        "final_article": article,
        "messages": annotated_msgs
    }

def publisher_node(state: SuperGraphState):
    """父节点最后的润色"""
    print("--- [Parent] Step 3: Final Polish ---")
    return {
        "messages": [AIMessage(content=f"EDITOR: Process complete. Article generated.")]
    }

# ==========================================
# PART 4: Build & Run
# ==========================================

def print_audit_log(graph, config):
    print("\n" + "="*60)
    print("📜  FULL CONVERSATION HISTORY (AUDIT LOG)")
    print("="*60)
    final_snapshot = graph.get_state(config)
    for msg in final_snapshot.values.get("messages", []):
        if "[Subgraph Researcher]" in msg.content:
            print(f"🕵️  {msg.content}")
        elif "[Subgraph Writer]" in msg.content:
            print(f"✍️  {msg.content}")
        elif "EDITOR" in msg.content:
            print(f"👔 {msg.content}")
        else:
            print(f"👤 {msg.content}")
        print("-" * 60)

def main():
    # 构建父图
    builder = StateGraph(SuperGraphState)
    builder.add_node("researcher", research_node)
    builder.add_node("writer", writing_node)
    builder.add_node("publisher", publisher_node)
    
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "publisher")
    builder.add_edge("publisher", END)
    
    super_graph = builder.compile(checkpointer=MemorySaver())
    
    # 可视化
    visualize_graph(super_graph, "07_hybrid_parent.png")
    visualize_graph(writing_graph, "07_hybrid_child_writer.png")
    visualize_graph(research_graph, "07_hybrid_child_researcher.png")
    
    # 运行
    config = {"configurable": {"thread_id": "hybrid_demo"}}
    user_input = "The future of AI Agents"
    
    print(f"User Request: {user_input}")
    
    super_graph.invoke(
        {"user_topic": user_input, "messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    # 打印最终结果
    state = super_graph.get_state(config).values
    print("\n" + "="*40)
    print("📰 FINAL ARTICLE PREVIEW:")
    print(state["final_article"])
    
    # 打印审计
    print_audit_log(super_graph, config)

if __name__ == "__main__":
    main()