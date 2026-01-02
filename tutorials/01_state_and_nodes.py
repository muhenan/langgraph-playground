# /// script
# dependencies = ["langgraph"]
# ///

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from utils.visualizer import visualize_graph

# ==========================================
# 1. State (数据定义)
# ==========================================
class AgentState(TypedDict):
    sentence: str
    processing_steps: List[str]

# ==========================================
# 2. Nodes (逻辑处理)
# ==========================================
def uppercase_node(state: AgentState) -> AgentState:
    """将句子转换为大写"""
    original = state["sentence"]
    # 打印简易日志
    print(f"--- [Node: Uppercase] Processing: '{original}'")
    
    return {
        "sentence": original.upper(),
        "processing_steps": state["processing_steps"] + ["Uppercase"]
    }

def reverse_node(state: AgentState) -> AgentState:
    """将句子反转"""
    original = state["sentence"]
    print(f"--- [Node: Reverse]   Processing: '{original}'")
    
    return {
        "sentence": original[::-1],
        "processing_steps": state["processing_steps"] + ["Reverse"]
    }

# ==========================================
# 3. Main (编排与执行)
# ==========================================
def main():
    # --- A. 构建图 ---
    builder = StateGraph(AgentState)

    builder.add_node("upper_caser", uppercase_node)
    builder.add_node("reverser", reverse_node)

    # 定义线性流: START -> upper_caser -> reverser -> END
    builder.add_edge(START, "upper_caser")
    builder.add_edge("upper_caser", "reverser")
    builder.add_edge("reverser", END)

    graph = builder.compile()

    # --- B. 可视化 (可选) ---
    visualize_graph(graph, "01_graph_structure.png")

    # --- C. 运行 ---
    initial_input = AgentState( {
        "sentence": "Hello LangGraph", 
        "processing_steps": []
    })
    
    print(f"\n[Start] Input: {initial_input}")
    
    # 触发执行
    result = graph.invoke(initial_input)
    
    print(f"[End]   Result: {result}\n")

if __name__ == "__main__":
    main()