import sqlite3
import os
import argparse
import sys
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# 1. 定义状态：只包含消息列表
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 定义节点逻辑：最简单的 LLM 调用，没有工具
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

def chatbot_node(state: State):
    # 直接调用 LLM，返回结果
    return {"messages": [llm.invoke(state["messages"])]}

# 3. 构建最简 Graph：START -> chatbot -> END
def build_simple_chat_graph(conn):
    checkpointer = SqliteSaver(conn)
    
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    
    # 编译时注入 checkpointer 以启用持久化
    return builder.compile(checkpointer=checkpointer)

# 4. 运行对话的主函数
def run_chat(thread_id: str, db_path: str):
    print(f"--- 尝试恢复聊天 (Thread ID: {thread_id}) ---")
    print(f"数据库路径: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # 注意：如果这里的 Graph 结构（节点名称）和之前保存该 Thread ID 时使用的结构不一样
    # LangGraph 可能无法正确加载所有状态。但对于 messages 列表通常是兼容的。
    graph = build_simple_chat_graph(conn)
    config = {"configurable": {"thread_id": thread_id}}

    # 检查历史记录
    current_state = graph.get_state(config)
    
    # 判断是否存在有效的历史消息
    if not current_state.values or "messages" not in current_state.values or not current_state.values["messages"]:
        print(f"❌ 未找到 Thread ID '{thread_id}' 的历史记录。程序退出。")
        return

    history = current_state.values["messages"]
    print(f"✅ 成功加载历史记录: {len(history)} 条消息")
    
    # 打印最近几条对话上下文
    print("-" * 30)
    for msg in history[-4:]:
        role = "AI" if msg.type == "ai" else "User"
        # 截断过长的消息以便显示
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"[{role}]: {content}")
    print("-" * 30)

    print("\n(输入 'q' 或 'quit' 退出)")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["q", "quit"]:
                break
                
            if not user_input.strip():
                continue
                
            # 调用 Graph
            # LangGraph 会自动：加载历史 -> 追加新消息 -> 执行 LLM -> 保存新状态
            # stream_mode="values" 会返回每一步的状态值
            events = graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            )
            
            # 打印最后一条消息（AI 的回复）
            # 注意：stream values 会先返回输入的消息，最后返回包含 AI 回复的完整列表
            for event in events:
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    # 只打印 AI 的回复
                    if last_msg.type == "ai":
                        print(f"AI: {last_msg.content}")
                        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LangGraph Chatbot Resumer")
    parser.add_argument("thread_id", help="The thread ID to resume")
    # 默认使用主 checkpoints.sqlite
    parser.add_argument("--db", default="tutorials/checkpoints/checkpoints.sqlite", help="Path to sqlite database")
    
    args = parser.parse_args()
    
    run_chat(args.thread_id, args.db)
