import os
import json
import re
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# llama-index imports
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core import load_index_from_storage  # Add import for loading

# Embedding and LLM imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Import config
from config import Config


def init_models() -> Tuple[HuggingFaceEmbedding, HuggingFaceLLM]:
    """
    初始化嵌入模型和 LLM，并设置全局 Settings。
    Returns:
        tuple: (embed_model, llm)
    """
    print("正在初始化模型...")

    # 正确的 HuggingFaceEmbedding 初始化方式
    embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_PATH)

    # HuggingFaceLLM 初始化
    # 示例：更换为一个较小的模型
    llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        tokenizer_name="Qwen/Qwen2-0.5B-Instruct",
        query_wrapper_prompt="{query_str}",
        context_window=2048,
        max_new_tokens=256,  # 可以减小生成长度
        device_map="auto",  # 自动分配
        # generate_kwargs={"temperature": 0.3}, # 如果需要，可以设置
    )

    # 设置全局默认值，这样在创建索引和查询时无需再次指定
    Settings.embed_model = embed_model
    Settings.llm = llm

    print("模型初始化完成。")
    # 验证 embedding
    try:
        test_emb = embed_model.get_text_embedding("测试文本")
        print(f"Embedding维度: {len(test_emb)}")
    except Exception as e:
        print(f"Embedding 模型验证失败: {e}")
        raise
    return embed_model, llm


def load_and_validate_dataset(data_dir: str) -> List[Dict[str, Any]]:
    """加载 JSON 数据集并生成字典列表"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_dataset = []
    for file_path in data_path.glob("*.json"):
        print(f"正在加载文件: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{file_path.name} 根元素必须是 list")

        # --- 核心修改点 ---
        # 遍历列表中的每一个字典 (item)
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{file_path.name} 第 {idx} 项不是 dict")

            # 遍历字典内部的每一个 "法条标题": "法条内容" 键值对
            for law_title, law_content in item.items():
                if not isinstance(law_content, str):
                    raise ValueError(f"{file_path.name} 第 {idx} 项键 {law_title} 的值不是 str")

                # 将每个法条标题和内容组合成一个独立的文本块
                text_block = f"{law_title}\n{law_content}"

                all_dataset.append({
                    "text": text_block,
                    "metadata": {
                        "source": file_path.name,
                        "law_title": law_title  # 可以保留原始标题作为元数据
                    }
                })
    print(f"成功加载并处理 {len(all_dataset)} 条独立的法条数据")
    return all_dataset


import re
from typing import List, Dict, Any
from llama_index.core.schema import TextNode

def create_nodes_with_custom_id(data_entries: List[Dict[str, Any]]) -> List[TextNode]:
    nodes = []

    for entry in data_entries:
        text = entry["text"]
        metadata = entry["metadata"]

        source_file = metadata["source"]
        law_title = metadata["law_title"]

        # 清洗字符串，避免非法字符
        clean_source = re.sub(r"[^\w\-_.\u4e00-\u9fff]", "_", source_file)
        clean_title = re.sub(r"[^\w\-_.\u4e00-\u9fff]", "_", law_title)

        # 使用 source + law_title 作为稳定 ID
        node_id = f"{clean_source}::{clean_title}"

        # 将法律标题拼接进文本，增强语义完整性
        full_text = f"{law_title}\n{text}"

        node = TextNode(
            text=full_text,
            id_=node_id,
            metadata={
                **metadata,
                "source_file": source_file,
                "law_title": law_title,
                "content_type": "legal_article"
            }
        )

        nodes.append(node)

    print(f"生成 {len(nodes)} 个文本节点（示例ID：{nodes[0].id_}）")
    return nodes


def build_and_persist_faiss_index():
    """
    构建 FAISS 索引并持久化到 Config.PERSIST_DIR。
    """
    print("开始构建 FAISS 索引...")
    os.makedirs(Config.PERSIST_DIR, exist_ok=True)

    # 1. 读取数据
    data_entries = load_and_validate_dataset(Config.DATA_DIR)
    nodes = create_nodes_with_custom_id(data_entries)

    # 2. 分句 (可选，但通常有助于提高检索精度)
    # splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    # chunks = splitter.get_nodes_from_documents(nodes)
    # print(f"分句后得到 {len(chunks)} 个文本块。")
    chunks = nodes  # 直接使用 nodes，因为我们已经将每个法条作为一个独立块处理
    print(f"使用 {len(chunks)} 个独立的法条块构建索引。")

    # --- 修改点：直接创建 VectorStoreIndex ---
    # 这样 llama-index 会自动处理内部的 vector store (SimpleVectorStore) 和 node store
    # 并且会使用 Settings.embed_model 生成 embeddings

    index = VectorStoreIndex(
        nodes=chunks,
        show_progress=True  # 显示构建进度
    )

    # 3. 保存
    # 存储上下文会同时保存所有必要的组件
    index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    print(f"FAISS 索引已保存至 {Config.PERSIST_DIR}")


def load_existing_index() -> VectorStoreIndex:
    """
    从 Config.PERSIST_DIR 加载已存在的索引。
    """
    print(f"尝试从 {Config.PERSIST_DIR} 加载现有索引...")
    # 加载前必须确保模型已初始化，因为加载过程需要知道如何处理 embeddings
    storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR)
    index = load_index_from_storage(storage_context, show_progress=True)
    print("✅ 索引加载成功")
    return index