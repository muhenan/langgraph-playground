
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from utils.visualizer import visualize_graph

load_dotenv()

# 1. State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a specific city."""
    return f"The weather in {city} is sunny and 25°C."

tools = [multiply, get_weather]

# 3. Nodes
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    print("--- [Node: Agent] Thinking... ---")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. Graph Construction
def main():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")

    # [高级工程师写法]: 
    # 使用 tools_condition (标准) + 显式 Path Map (清晰)
    # 这样既不用自己写路由函数，又能清楚地看到 END 出口
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    builder.add_edge("tools", "agent")

    graph = builder.compile()

    # Visualization
    visualize_graph(graph, "03_graph_structure.png")

    # Run
    print("\n=== Test: Weather Query ===")
    initial_input = {"messages": [HumanMessage(content="What is the weather in Boston?")]}
    print(f"Initial Input: {initial_input}")
    
    final_state = graph.invoke(initial_input)
    print(f"Final State: {final_state}")
    print(f"Final Answer: {final_state['messages'][-1].content}")

if __name__ == "__main__":
    main()