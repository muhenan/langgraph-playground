from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

def main():
    load_dotenv()

    # 初始化 LLM
    print("🤖 初始化 ChatBot...")
    try:
        # 尝试使用配置的模型，如果失败可能需要检查 .env 或 key
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 历史记录列表
    # 可以添加一个 SystemMessage 来设定人设
    messages = [
        SystemMessage(content="你是一个幽默风趣的 AI 助手。")
    ]

    print("\n=== 终端聊天机器人 (输入 'quit', 'exit' 或 'q' 退出) ===\n")

    while True:
        try:
            # 1. 获取用户输入
            user_input = input("👤 User: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break
            
            if not user_input:
                continue

            # 2. 将用户问题加入历史
            messages.append(HumanMessage(content=user_input))

            # 3. 调用 LLM
            # print("   (Thinking...)", end="\r") # 简单的加载提示
            response = llm.invoke(messages)

            # 4. 打印回答
            print(f"🤖 AI:   {response.content}\n")

            # 5. 将 AI 回答加入历史
            messages.append(response)

        except KeyboardInterrupt:
            # 捕获 Ctrl+C
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")

if __name__ == "__main__":
    main()

