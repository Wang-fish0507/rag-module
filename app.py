
import os

# Import necessary functions and config
from config import Config
from rag_law import init_models, build_and_persist_faiss_index

from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
# 导入 StorageContext 用于加载
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

QA_TEMPLATE = (
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题。\n"
    "如果法律条文中没有相关信息，请回答“根据提供的法律条文，无法找到相关答案。”\n\n"
    "相关法律条文：\n{context_str}\n\n"
    "问题：{query_str}\n\n"
    "回答："
)


def main():
    print("=== 应用程序启动初始化 ===")

    embed_model, llm = init_models()

    # 检查并构建/加载索引
    if not os.path.exists(Config.PERSIST_DIR) or not os.listdir(Config.PERSIST_DIR):
        print("未找到现有索引，开始构建新的索引...")
        build_and_persist_faiss_index()
        print("新索引构建完成。")
    else:
        print("发现现有索引，准备加载...")

    # 加载索引 - 正确的方式：先加载 StorageContext
    print(f"尝试从 {Config.PERSIST_DIR} 加载现有索引...")
    # 从持久化目录加载存储上下文，这会加载所有存储的组件
    storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR)

    # 从加载的存储上下文中加载索引
    # 这个调用会利用 storage_context 中的 vector_store, docstore 等信息
    index = load_index_from_storage(storage_context, show_progress=True)

    print("✅ 索引加载成功")
    print("=== 初始化完成 ===")

    # --- 交互式查询循环 ---
    print("\n--- 法律问答系统已就绪 ---")
    print("输入 'quit' 或 'exit' 退出程序。\n")

    while True:
        query_str = input("请输入您的问题: ").strip()
        if query_str.lower() in ['quit', 'exit']:
            print("再见！")
            break

        if not query_str:
            print("输入不能为空，请重新输入。\n")
            continue

        try:
            # 使用加载的索引进行检索
            retriever = index.as_retriever(similarity_top_k=Config.TOP_K)
            nodes = retriever.retrieve(query_str)

            # --- 再次检查节点内容 ---
            print(f"\n--- 检索到 {len(nodes)} 个原始节点内容 ---")
            if not nodes:
                print("❌ 检索器未返回任何节点，可能存在索引加载或构建问题。")
                print("最终回答将来自模型的自由发挥（可能导致幻觉）:")
            else:
                for i, node in enumerate(nodes):
                    print(f"[节点 {i + 1}]")
                    print(f"ID: {node.node_id}")
                    print(f"得分: {node.score:.4f}")
                    print(f"内容: {node.get_text()}")  # 打印原始文本
                    print("-" * 20)

            # 提取检索到的上下文文本
            context_str = "\n".join([node.get_text() for node in nodes])

            # 构建提示模板并获取 LLM 回答
            prompt_template = PromptTemplate(QA_TEMPLATE)
            prompt = prompt_template.template.format(context_str=context_str, query_str=query_str)

            # 调用 LLM 获取响应
            response_obj = Settings.llm.complete(prompt)
            response_text = response_obj.text

            print("\n--- 最终回答 ---")
            print(response_text)
            print("-" * 20 + "\n")

        except KeyboardInterrupt:
            print("\n程序被用户中断。再见！")
            break
        except Exception as e:
            import traceback
            print(f"\n处理查询时发生错误: {e}")
            print(traceback.format_exc())  # 打印详细堆栈跟踪
            print("请重试。\n")


if __name__ == '__main__':

    main()
