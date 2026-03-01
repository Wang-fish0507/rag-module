import os

class Config:
    # 嵌入模型路径
    EMBED_MODEL_PATH = "./model/bge-small-zh-v1.5"
    # 原始数据存放目录
    DATA_DIR = "./data"
    # 向量数据库持久化目录
    PERSIST_DIR = "./storage"
    # 索引中查找的最相似条目数
    TOP_K = 3
    # Ollama 使用的模型名称
    OLLAMA_MODEL_NAME = "qwen:1.8b"