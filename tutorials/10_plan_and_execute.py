# /// script
# dependencies = [
#     "langgraph",
#     "langchain-openai",
#     "langchain-core",
#     "python-dotenv"
# ]
# ///


"""
Plan-and-Execute (规划-执行) 架构
核心隐喻：建筑工程队

Planner (设计师): 不干脏活累活。只负责看客户需求，画出图纸（Step 1, 2, 3）。
Executor (施工员): 不看整张图纸，只看“今天干什么”。但他需要知道昨天干了什么（Context）。
Re-Planner (工头/监理): 每天干完活来检查。根据进度调整计划，或者签字验收。

关键组件解析：
1. Planner (The Brain): 将模糊目标转化为结构化任务列表。强迫模型在动手前先 CoT。
2. Executor (The Hands): 专注执行当前任务。State Continuity (状态连续性) 至关重要，必须看到历史记忆。
3. Re-Planner (The Reflector): 闭环反馈系统。让系统有了“纠错”和“适应”的能力。

优缺点分析 (Trade-offs):
✅ Pros: 解决长难任务（如写代码）、鲁棒性（允许中途出错）、可观测性。
❌ Cons: 慢 & 贵（步骤多，LLM调用次数多）、上下文堆积（Prompt 越来越长）。
"""

import operator
from typing import Annotated, List, Tuple, TypedDict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from utils.visualizer import visualize_graph

load_dotenv()

# ==========================================
# 1. State Definitions
# ==========================================

class PlanExecuteState(TypedDict):
    input: str                          # 原始大目标
    plan: List[str]                     # 待执行的任务栈
    # 核心：记录历史步骤 [(Step Name, Result Content), ...]
    past_steps: Annotated[List[Tuple[str, str]], operator.add] 
    response: Optional[str]             # 最终答案

# ==========================================
# 2. Schema (Structured Output)
# ==========================================

class Plan(BaseModel):
    """Planner 的产出"""
    steps: List[str] = Field(description="List of steps to follow, in order.")

class Response(BaseModel):
    """Re-Planner 的产出"""
    response: Optional[str] = Field(description="Final answer to the user, if done.")
    plan: Optional[List[str]] = Field(description="Updated plan (remaining steps), if not done.")

# ==========================================
# 3. Nodes (Pure Logic)
# ==========================================

# 用更小的模型有幻觉，会丢上下文，丢步骤
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

def planner_node(state: PlanExecuteState):
    """
    Node 1: Planner (大脑)
    作用: 将模糊的自然语言目标（input）转化为结构化的任务列表（plan）。
    为什么重要: 处理长链路任务时，强迫模型先理顺逻辑。
    """
    print(f"--- [Planner] Strategizing for: {state['input']} ---")
    
    planner_llm = llm.with_structured_output(Plan)
    prompt = (
        "For the given objective, come up with a simple step-by-step plan. "
        "The result of the final step should be the final answer."
    )
    
    plan = planner_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["input"])
    ])
    
    print(f"📋 Initial Plan: {plan.steps}")
    # 初始化 Plan
    return {"plan": plan.steps}

def executor_node(state: PlanExecuteState):
    """
    Node 2: Executor (执行者)
    作用: 从任务栈顶取出一个 task 执行。
    State Continuity: 必须把“历史记忆” (past_steps) 传给当前操作者，否则它是“瞎子”。
    """
    plan = state["plan"]
    task = plan[0]
    
    print(f"--- [Executor] Working on: '{task}' ---")
    
    # [关键修复]: 构建上下文
    # 把之前做过的步骤和结果拼起来
    context = ""
    if state["past_steps"]:
        context = "Here is the context of what has been done so far:\n"
        for step, result in state["past_steps"]:
            context += f"Step: {step}\nResult: {result}\n---\n"
    
    # 将上下文 + 当前任务一起发给 LLM
    executor_prompt = (
        "You are a helpful worker. "
        "Execute the following task to the best of your ability."
        "Use the provided context if necessary to complete the task."
        "Provide a concise result."
    )
    
    result = llm.invoke([
        SystemMessage(content=executor_prompt),
        HumanMessage(content=f"{context}\n\nCurrent Task: {task}")
    ])
    
    output = result.content
    print(f"✅ Result: {output}")
    
    return {
        "past_steps": [(task, output)]
    }

def replanner_node(state: PlanExecuteState):
    """
    Node 3: Re-Planner (反思者)
    作用: 动态调整计划。
    输入: Goal + Reality (已完成的事实)
    输出: Gap (剩下的计划) 或 Response (最终答案)
    这是架构中最性感的部分，提供了“纠错”能力。
    """
    print("--- [Re-Planner] Updating Plan... ---")
    
    replanner_llm = llm.with_structured_output(Response)
    
    # 构造上下文：目标 + 原计划 + 已完成
    # 这里的 prompt 决定了 Agent 有多"聪明"
    past_steps_format = "\n".join([f"Step: {s}\nResult: {r}" for s, r in state["past_steps"]])
    
    # 核心修改：加强了 Instructions 部分的逻辑约束
    prompt = f"""
    Your objective: {state['input']}
    
    Original Plan: {state['plan']}
    
    Completed Steps:
    {past_steps_format}
    
    Instructions:
    1. Analyze the "Completed Steps". Did the last step successfully produce the final answer for the "Objective"?
    2. IF YES (Objective is Done):
       - You MUST output the final answer in the 'response' field.
       - The 'response' should be a synthesis of the execution results.
       - Set 'plan' to [].
       - **CRITICAL**: You cannot return an empty plan without a response. If the plan is empty, 'response' MUST contain the answer.
       
    3. IF NO (Objective is NOT Done):
       - Return a new list of *remaining* steps in 'plan'.
       - Remove the step that was just completed.
       - Do NOT set 'response'.
    """
    
    result = replanner_llm.invoke(prompt)
    
    # 这里的 Python 逻辑保持简单即可，因为我们相信 LLM 会遵循上面的 CRITICAL 指令
    if result.response:
        print("🎉 [Re-Planner] Finished via Response!")
        print(f"🎉 [Re-Planner] Response: {result.response}")
        return {"response": result.response, "plan": []}
    else:
        print(f"🔄 [Re-Planner] New Plan: {result.plan}")
        return {"plan": result.plan}

# ==========================================
# 4. Graph Logic
# ==========================================

def router(state: PlanExecuteState):
    # 1. 如果有最终回复，结束
    if state.get("response"):
        return END
    
    # 2. [修复] 如果计划表空了，也没活干了，强制结束
    if not state.get("plan"): # 空列表在 Python 中为 False
        return END
        
    # 3. 还有活，继续干
    return "executor"

def main():
    builder = StateGraph(PlanExecuteState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("re_planner", replanner_node)
    
    # 线性流：Start -> Plan -> Exec -> RePlan
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "re_planner")
    
    # 循环点：RePlan -> Exec (或者 END)
    builder.add_conditional_edges(
        "re_planner",     # Source
        router,           # Function (决定去哪)
        ["executor", END] # Path Map (告诉画图工具：只有这两条路)
    )
    
    graph = builder.compile()
    visualize_graph(graph, "10_plan_exec_pure.png")
    
    # Run
    user_query = "Write a haiku about recursion, then explain it, then translate the explanation to French."
    print(f"User Query: {user_query}\n")
    
    config = {"recursion_limit": 20}
    
    # 只需要简单的 stream 即可
    for event in graph.stream({"input": user_query}, config=config):
        pass # 日志已经在节点内部打印了

    # 获取最终结果
    # 注意：在 stream 结束后，我们通常无法直接获得最后的状态对象，除非使用 checkpointer
    # 但我们可以打印最后一次 event 或者上面的日志来验证

if __name__ == "__main__":
    main()