from abc import ABC
import torch
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TextEmbedding(Embeddings, ABC):
    def __init__(self, model_name, batch_size=64, max_len=512, **kwargs):
        super.__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if "bge" in model_name:
            self.instruction = "为这个句子生成表示以检索相关文章："
        else:
            self.instruction = ""
        self.batch_size = batch_size
        self.max_len = max_len
        print("Load embedding model successfully.")

    def embed_documents(self, texts: list[str], pooling="mean", whitening=False) -> list[list[float]]:
        num_texts = len(texts)
        texts = [t.replace('\n', ' ') for t in texts]
        all_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, max_length=self.max_len, padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                if pooling == "mean":
                    batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                elif pooling == "cls":
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    raise ValueError

                all_embeddings.extend(batch_embeddings.cpu().numpy())

        all_embeddings = np.array(all_embeddings)

        if whitening:
            if not hasattr(self, 'W') or not hasattr(self, "mu"):
                self.W, self.bias = self._compute_kernel_bias(all_embeddings)
            all_embeddings = (all_embeddings + self.bias) @ self.W

        all_embeddings = torch.nn.functional.normalize(torch.tensor(all_embeddings), p=2, dim=1)
        return all_embeddings.tolist()

    def _compute_kernel_bias(self, vecs, n_components=384):
        """
        白化操作，让向量之间的结构更均匀
        只适用于通用预训练的模型输出
        :param vecs:[num_samples, hidden_dim]
        :return: W, -mu
        """
        mu = vecs.mean(axis=0, keepdim=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def _mean_pooling(self, model_output, attention_mask):
        hidden_state = model_output.last_hidden_state
