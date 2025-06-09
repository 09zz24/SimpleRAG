import fitz
from transformers import AutoTokenizer
import re


class Reader:
    def __init__(self, path: str, model_name='bert-base-Chinese', max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        if path.endswith("pdf"):
            self.corpus = self._read_pdf(path)
        else:
            self.corpus = self._read_file(path)


    def _read_pdf(self, path):
        doc = fitz.open(path)
        chunks = []

        # 按页分块，token过长则切割
        for page_number, page in enumerate(doc):
            current_text = page.get_text("text")
            if current_text:
                cleaned = current_text.replace('\n', ' ').strip()
                token_count = self._count_tokens(cleaned)

                if token_count <= self.max_tokens:
                    chunks.append({'page': page_number+1, 'text': cleaned})
                else:
                    sub_chunks = self._split_pages(cleaned)
                    for chunk in sub_chunks:
                        chunks.append({'page': page_number+1, 'text': chunk})
        return chunks

    def _read_file(self, path):
        raise NotImplementedError

    def _count_tokens(self, text):
        return self.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.shape[1]

    def _split_pages(self, text):
        sentences = re.split(r'(?<=[。！？])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        curr_chunk = ''

        for sent in sentences:
            temp_chunk = (curr_chunk+sent).strip() if curr_chunk else sent
            token_count = self._count_tokens(temp_chunk)

            if token_count <= self.max_tokens:
                curr_chunk = temp_chunk
            else:
                if curr_chunk:
                    chunks.append(curr_chunk)
                    curr_chunk = sent

        if curr_chunk:
            chunks.append(curr_chunk)

        return chunks

    def get_corpus(self):
        return self.corpus


if __name__ == "__main__":
    test = Reader("../data/少年中国说.pdf")
    corpus = test.get_corpus()
    print(corpus)