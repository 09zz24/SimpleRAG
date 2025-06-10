from abc import ABC
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TextEmbedding(Embeddings, ABC):
    def __init__(self, model_name, batch_size=64, max_len=512, device="cuda", **kwargs):
        super.__init__(**kwargs)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if "bge" in model_name:
            self.instruction = "为这个句子生成表示以检索相关文章："
        else:
            self.instruction = ""
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device
        print("Load embedding model successfully.")

    
