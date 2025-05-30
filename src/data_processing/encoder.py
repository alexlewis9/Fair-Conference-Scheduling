import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def mean_pool(chunks):
    return np.mean(np.array(chunks), axis=0)


def process_reconstruct(reconstruct):
    if reconstruct == 'mean_pool':
        return mean_pool
    return None


class Encoder:
    def __init__(self, name, st_compat = False, stride=0, max_tokens=0, reconstruct = 'mean_pool'):
        self.name = name
        if (name == "text-embedding-3-small" or
         name == "text-embedding-ada-002" or
         name == "text-embedding-3-large"):
            load_dotenv()
            self.client = OpenAI()
            self.max_tokens = 8192 if max_tokens == 0 else max_tokens
            self.tokenizer = tiktoken.encoding_for_model(name) # Used to splitting texts exclusively
        elif st_compat:
            self.st_compat = st_compat
            self.client = SentenceTransformer(name,
                                              trust_remote_code=True)
            self.max_tokens = self.client.get_max_seq_length() if max_tokens == 0 else max_tokens
            self.tokenizer = self.client.tokenizer  # Used to splitting texts exclusively
        else:
            raise ValueError(
                "Invalid model name. Please choose from 'text-embedding-3-small',"
                " 'text-embedding-ada-002', 'text-embedding-3-large', or 'gte-Qwen2-7B-instruct'.")
        self.stride = stride if stride != 0 else self.max_tokens
        self.reconstruct = process_reconstruct(reconstruct)

    def chunk_text_to_tokens(self, text: str, max_tokens: int = 0, stride= 0) -> list[str]:
        """
        Splits a long text into overlapping chunks of tokenized size with stride.
        Returns a list of token chunks.
        """
        max_tokens = self.max_tokens if max_tokens == 0 else max_tokens
        stride = self.stride if stride == 0 else stride
        enc = self.tokenizer
        input_ids = enc.encode(text)
        chunks = []

        for i in range(0, len(input_ids), stride):
            chunk_ids = input_ids[i:i + max_tokens]
            if not chunk_ids:
                break
            chunks.append(chunk_ids)
            if i + max_tokens >= len(input_ids):
                break  # Don't start new chunks past the end

        return chunks


    def encode(self, text):
        text = text.replace("\n", " ")
        token_chunks = self.chunk_text_to_tokens(text)

        if (self.name == "text-embedding-3-small" or
                self.name == "text-embedding-ada-002" or
                self.name == "text-embedding-3-large"):
            embeddings = []
            for chunk in token_chunks:
                embedding = self.client.embeddings.create(input=chunk, model=self.name).data[0].embedding
                embeddings.append(embedding)
            return self.reconstruct(embeddings) # i.e., mean pool
        elif self.st_compat:
            embeddings = []
            for chunk in token_chunks:
                embedding = self.client.encode(chunk)
                embeddings.append(embedding)
            return self.reconstruct(embeddings) # i.e., mean pool
        return None

    def encode_pre_recon(self, text):
        text = text.replace("\n", " ")
        token_chunks = self.chunk_text_to_tokens(text)

        if (self.name == "text-embedding-3-small" or
                self.name == "text-embedding-ada-002" or
                self.name == "text-embedding-3-large"):
            embeddings = []
            for chunk in token_chunks:
                embedding = self.client.embeddings.create(input=chunk, model=self.name).data[0].embedding
                embeddings.append(embedding)
            return self.reconstruct(embeddings).tolist(), embeddings # i.e., mean pool
        elif self.st_compat:
            embeddings = []
            for chunk in token_chunks:
                embedding = self.client.encode(chunk)
                embeddings.append(embedding.tolist())
            return self.reconstruct(embeddings).tolist(), embeddings # i.e., mean pool
        return None

    def count_tokens(self, text):
        text = text.replace("\n", " ")
        chunks = self.chunk_text_to_tokens(text)
        return sum([len(chunk) for chunk in chunks])
