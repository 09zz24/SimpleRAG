# SimpleRAG



本项目源于阿里云天池2023全球智能汽车AI挑战赛赛道一：AI大模型检索问答。项目实现了简单的RAG流程



```python
.
├── README.md
├── data # 数据文件
│   ├── 测试问题.json # 官方给出的测试问题
│   ├── 少年中国说.pdf # 文档切块的测试数据
│   └── 初赛训练数据集.pdf # 检索文档
├── src # 主程序部分
│   ├── data_loader.py # 文档切块
│   ├── infer.py
│   ├── reranker.py
│   └── retriever.py # embedding和检索器
```

