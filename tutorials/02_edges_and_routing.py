# /// script
# dependencies = ["langgraph"]
# ///

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from utils.visualizer import visualize_graph

# ==========================================
# 1. State (数据定义)
# ==========================================
class AgentState(TypedDict):
    value: int          # 原始数值
    hex_repr: str       # [新增] 预处理后的十六进制表示
    action_taken: str   # 记录最终执行了什么动作

# ==========================================
# 2. Nodes (逻辑处理)
# ==========================================
def classify_input_node(state: AgentState) -> AgentState:
    """
    预处理节点：
    1. 数据清洗：确保数值为正数 (取绝对值)
    2. 特征提取：计算数值的十六进制表示
    """
    raw_val = state["value"]
    
    # [模拟数据清洗]: 假设下游任务只处理正数
    clean_val = abs(raw_val)
    
    # [模拟特征提取]: 转为十六进制字符串
    hex_val = hex(clean_val)
    
    print(f"--- [Node: Classify] Preprocessing: {raw_val} -> {clean_val} (Hex: {hex_val})")
    
    # 更新 State：更新清洗后的 value，并写入 hex_repr
    return {
        "value": clean_val, 
        "hex_repr": hex_val,
        "action_taken": "Preprocessed"
    }

def handle_big_number_node(state: AgentState) -> AgentState:
    """分支 A：处理大数逻辑"""
    val = state["value"]
    new_val = val // 2  # 整除
    print(f"--- [Node: Handle Big] {val} (>50). Halving it -> {new_val}")
    return {"value": new_val, "action_taken": "Halved (Big Logic)"}

def handle_small_number_node(state: AgentState) -> AgentState:
    """分支 B：处理小数逻辑"""
    val = state["value"]
    new_val = val * 2
    print(f"--- [Node: Handle Small] {val} (<=50). Doubling it -> {new_val}")
    return {"value": new_val, "action_taken": "Doubled (Small Logic)"}

# ==========================================
# 3. Router (决策逻辑)
# ==========================================
def decide_next_step(state: AgentState) -> Literal["handle_big", "handle_small"]:
    """
    根据预处理后的数据决定走向
    """
    clean_value = state["value"]
    
    if clean_value > 50:
        return "handle_big"
    else:
        return "handle_small"

# ==========================================
# 4. Main (编排与执行)
# ==========================================
def main():
    # --- A. 构建图 ---
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_input_node)
    builder.add_node("handle_big", handle_big_number_node)
    builder.add_node("handle_small", handle_small_number_node)

    builder.add_edge(START, "classify")

    """
    add_conditional_edges 的第二个参数（即 decide_next_step）只是一个纯粹的 Python 函数，用于决策。
    它不保存状态，也不修改状态（虽然技术上可以，但不建议），它的唯一任务是返回下一个节点的名称。
    因此，它不被视为图中的一个"驻留"节点，而是依附于前一个节点（classify）的出边逻辑。
    """
    builder.add_conditional_edges("classify", decide_next_step)
    
    builder.add_edge("handle_big", END)
    builder.add_edge("handle_small", END)

    graph = builder.compile()

    # --- B. 可视化 ---
    visualize_graph(graph, "02_graph_structure.png")

    # --- C. 运行测试 ---
    
    # Case 1: 负的小数 (测试清洗功能)
    # 输入 -10 -> 清洗为 10 -> 路由到 Small -> 乘2 -> 结果 20
    input_1: AgentState = {"value": -10, "hex_repr": "", "action_taken": ""}
    print(f"\n[Start Case 1] Input: {input_1}")
    result_1 = graph.invoke(input_1)
    print(f"[End Case 1]   Result: {result_1}")

    # Case 2: 大数
    input_2: AgentState = {"value": 100, "hex_repr": "", "action_taken": ""}
    print(f"\n[Start Case 2] Input: {input_2}")
    result_2 = graph.invoke(input_2)
    print(f"[End Case 2]   Result: {result_2}")

if __name__ == "__main__":
    main()