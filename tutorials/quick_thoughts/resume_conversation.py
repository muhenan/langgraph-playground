import sqlite3
import argparse
import sys
import os

# Ensure we can import from the parent/root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ==========================================
# Re-define Graph Structure (Must match the original graph)
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
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def build_graph(checkpointer):
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
    return builder.compile(checkpointer=checkpointer)

# ==========================================
# Main Resume Logic
# ==========================================
def resume_conversation(thread_id: str, new_query: str, db_path: str):
    print(f"--- Resuming Conversation ---")
    print(f"Database:  {db_path}")
    print(f"Thread ID: {thread_id}")
    
    # Connect to existing database
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # Using checkpointer to load state
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer)
    
    # Configuration with the same thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if there is existing state
    # graph.get_state(config) returns a StateSnapshot object
    current_state = graph.get_state(config)
    
    if not current_state.values:
        print(f"Warning: No existing state found for thread_id '{thread_id}'. A new conversation will be started.")
    else:
        print(f"Found existing conversation history ({len(current_state.values.get('messages', []))} messages).")
    
    print(f"\nUser: {new_query}")
    
    # Invoke with the new message. 
    # Because we passed the checkpointer and the same thread_id, 
    # LangGraph will:
    # 1. Load the latest state from SQLite
    # 2. Append our new message
    # 3. Run the graph
    # 4. Save the new state back to SQLite
    result = graph.invoke(
        {"messages": [HumanMessage(content=new_query)]},
        config=config
    )
    
    print(f"Agent: {result['messages'][-1].content}")
    print("\n--- Conversation Saved ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume a LangGraph conversation from SQLite checkpoints")
    parser.add_argument("thread_id", nargs="?", default="user_neo", help="The thread ID to resume")
    parser.add_argument("--query", default="What did we talk about previously?", help="The new question to ask")
    parser.add_argument("--db", default="tutorials/checkpoints/checkpoints.sqlite", help="Path to sqlite database")
    
    args = parser.parse_args()
    
    resume_conversation(args.thread_id, args.query, args.db)

