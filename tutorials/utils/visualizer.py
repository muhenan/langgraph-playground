import os

def visualize_graph(graph, filename="graph_structure.png"):
    """
    将 LangGraph 的图结构保存为 PNG 图片。
    
    Args:
        graph: 编译后的 LangGraph 对象 (CompiledGraph)
        filename: 保存的文件名 (包含扩展名)，将自动保存在 images/ 目录下
    """
    try:
        # 确保 images 目录存在
        os.makedirs("images", exist_ok=True)
        
        # 拼接完整路径
        output_path = os.path.join("images", filename)
        
        # 生成并保存图片
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
            
        print(f"Graph image saved to '{output_path}'")
        
    except Exception as e:
        # 捕获异常并打印，避免因绘图失败影响主流程
        print(f"Skipping graph visualization: {e}")

