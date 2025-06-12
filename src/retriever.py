from abc import ABC
import torch
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TextEmbedding(Embeddings, ABC):
    def __init__(self, model_name, batch_size=64, max_len=512, pooling="mean", whitening=False, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if "bge" in model_name:
            self.instruction = "为这个句子生成表示以检索相关文章："
        else:
            self.instruction = ""
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.pooling = pooling
        self.whitening = whitening
        print("Load embedding model successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

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
                if self.pooling == "mean":
                    batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                elif self.pooling == "cls":
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    raise ValueError

                all_embeddings.extend(batch_embeddings.cpu().numpy())

        all_embeddings = np.array(all_embeddings)

        if self.whitening:
            if not hasattr(self, 'W') or not hasattr(self, "mu"):
                self.w, self.mu = self._compute_kernel_bias(all_embeddings)
            all_embeddings = (all_embeddings + self.mu) @ self.w

        all_embeddings = torch.nn.functional.normalize(torch.tensor(all_embeddings), p=2, dim=1)
        return all_embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        text = text.replace('\n', ' ')

        if "bge" in self.model_name:
            encoded_input = self.tokenizer([text + self.instruction], max_length=self.max_len, padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        else:
            encoded_input = self.tokenizer(text, max_length=self.max_len, padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            if self.pooling == "mean":
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            elif self.pooling == "cls":
                sentence_embeddings = model_output.last_hidden_state[:, 0]
            else:
                raise ValueError

        sentence_embeddings = sentence_embeddings.cpu().numpy()
        if self.whitening:
            sentence_embeddings = (sentence_embeddings + self.mu) @ self.w

        sentence_embeddings = torch.nn.functional.normalize(torch.tensor(sentence_embeddings), p=2, dim=1)
        return sentence_embeddings.tolist()

    def _compute_kernel_bias(self, vecs, n_components=384):
        """
        白化操作，让向量之间的结构更均匀
        只适用于通用预训练的模型输出
        :param vecs:[num_samples, hidden_dim]
        :return: w, -mu
        """
        mu = vecs.mean(axis=0, keepdim=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        w = np.dot(u, np.diag(1 / np.sqrt(s)))
        return w[:, :n_components], -mu

    def _mean_pooling(self, model_output, attention_mask):
        hidden_state = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        state_sum = (hidden_state * mask_expanded).sum(dim=1)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)
        state_mean = state_sum / mask_sum
        return state_mean
