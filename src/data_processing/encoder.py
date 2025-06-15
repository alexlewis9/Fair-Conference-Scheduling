import logging
import time

import numpy as np
import tiktoken
import voyageai
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
logger = logging.getLogger(__name__)

def mean_pool(chunks):
    return np.mean(np.array(chunks), axis=0)


def process_reconstruct(reconstruct):
    if reconstruct == 'mean_pool':
        return mean_pool
    return None


class Encoder:
    def __init__(self, name, provider, stride=0, max_tokens=0, reconstruct ='mean_pool'):
        self.name = name
        load_dotenv()
        self.openai = False
        self.claude = False
        self.st_compat = False
        self.gemini = False
        if provider == 'openai':
            self.client = OpenAI()
            self.max_tokens = 8192 if max_tokens == 0 else max_tokens
            self.tokenizer = tiktoken.encoding_for_model(name) # Used to splitting texts exclusively
            self.openai = True
        elif provider == 'sent_trans':
            self.st_compat = True
            self.client = SentenceTransformer(name, trust_remote_code=True)
            self.max_tokens = self.client.get_max_seq_length() if max_tokens == 0 else max_tokens
            self.tokenizer = self.client.tokenizer  # Used to splitting texts exclusively
        elif provider == 'anthropic':
            self.client = voyageai.Client()
            self.max_tokens = 16000 if self.name == 'voyage-law-2' or self.name == 'voyage-code-2' else 32000
            self.tokenizer = AutoTokenizer.from_pretrained(f'voyageai/{self.name}')
            self.claude = True
        elif provider == 'gemini': # text-embedding-004
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.max_tokens = 8192 if self.name == 'gemini-embedding-exp-03-07' else 2048
            self.tokenizer = None
            self.gemini = True
        else:
            raise ValueError(
                "Invalid provider. Please choose from 'anthropic' or 'openai' if your model is from either of them,"
                " or 'sent_trans' if it's Sentence Transformer compatible.")
        self.stride = stride if stride != 0 else self.max_tokens
        self.reconstruct = process_reconstruct(reconstruct)

    def chunk_text_to_tokens(self, text: str, max_tokens: int = 0, stride= 0, disallowed_special = ()) -> list[str]:
        """
        Splits a long text into overlapping chunks of tokenized size with stride.
        Returns a list of token chunks.
        """
        max_tokens = self.max_tokens if max_tokens == 0 else max_tokens
        stride = self.stride if stride == 0 else stride
        enc = self.tokenizer
        if self.openai:
            input_ids = enc.encode(text, disallowed_special=disallowed_special)
        else:
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

    def chunk_by_token_counts(self, text, stride, max_token=0):
        """
        Split text into chunks of max_tokens size, with stride.
        Returns a list of token chunks.
        Developed for models without Tokenizer, but a function to return the total counts of tokens.
        """
        max_token = self.max_tokens if max_token == 0 else max_token
        words = text.split()
        if not words:
            return []

        chunks = []
        idx = 0
        overlap_words = []

        while idx < len(words):
            # Current window includes overlap from previous chunk
            available_words = overlap_words + words[idx:]

            # Binary search for maximum words that fit
            # Each word is at least 1 token, so max possible words is max_token
            left, right = 1, min(len(available_words), max_token)
            best = 1

            while left <= right:
                mid = (left + right) // 2
                # Join words into string for tokenization
                text = ' '.join(available_words[:mid])
                tokens = self.count_tokens_gemini(text)

                if tokens <= max_token:
                    best = mid
                    left = mid + 1
                else:
                    right = mid - 1

            # Create chunk as string
            chunk_words = available_words[:best]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

            # Move index forward by new words consumed (excluding overlap)
            new_words = best - len(overlap_words)
            idx += new_words

            # Calculate overlap for next chunk
            overlap_size = int(len(chunk_words) * stride)
            overlap_words = chunk_words[-overlap_size:] if overlap_size > 0 else []

        return chunks

    def encode(self, text, verbose=False, disallowed_special=(), max_retries = 3, retry_delay = 75):
        text = text.replace('\n', ' ')
        embeddings = []
        if self.openai:
            logger.info(f"chunking")
            token_chunks = self.chunk_text_to_tokens(text, disallowed_special=disallowed_special)
            logger.info(f"chunked") if not verbose else logger.info(f"chunked: {len(token_chunks)}")
            for chunk in token_chunks:
                embedding = self.client.embeddings.create(input=chunk, model=self.name).data[0].embedding
                embeddings.append(embedding)

        elif self.st_compat:
            logger.info(f"chunking")
            token_chunks = self.chunk_text_to_tokens(text, disallowed_special=disallowed_special)
            logger.info(f"chunked") if not verbose else logger.info(f"chunked: {len(token_chunks)}")
            token_chunks = [self.tokenizer.encode(chunk) for chunk in token_chunks]
            for chunk in token_chunks:
                embedding = self.client.encode(chunk)
                embeddings.append(embedding.tolist())

        elif self.claude:
            logger.info(f"chunking")
            token_chunks = self.chunk_text_to_tokens(text, disallowed_special=disallowed_special)
            logger.info(f"chunked") if not verbose else logger.info(f"chunked: {len(token_chunks)}")
            token_chunks = [self.tokenizer.encode(chunk) for chunk in token_chunks]
            for chunk in token_chunks:
                embedding = self.client.embed(self.tokenizer.decode(chunk), model=self.name).embeddings[0]
                embeddings.append(embedding.tolist())

        elif self.gemini:
            logger.info(f"chunking")
            chunks = self.chunk_by_token_counts(text, self.stride) # IN PERCENTAGE, been testing 10% in OpenAI
            logger.info(f"chunked") if not verbose else logger.info(f"chunked: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                for attempt in range(max_retries):
                    try:
                        logger.info(f"embedding chunk {i + 1}") if not verbose else logger.info(f"embedding chunk {i + 1}- (Attempt {attempt + 1})")
                        result = self.client.models.embed_content(
                            model = self.name,
                            contents=chunk
                        )
                        embeddings.append(result.embeddings[0].values)
                        logger.info(f"embedded chunk {i + 1}") if not verbose else logger.info(f"embedded chunk {i + 1}- (Attempt {attempt + 1})")
                        break
                    except Exception as e:
                        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                            code = e.response.status_code
                        elif hasattr(e, 'status_code'):
                            code = e.status_code
                        else:
                            code = getattr(e, 'code', None)
                        if code == 429:
                            logger.warning(
                                f"Rate Limit exceeded. Sleeping for {retry_delay} seconds... (Attempt {attempt + 1})")
                            time.sleep(retry_delay)
                        else:
                            raise e


        logger.info(f"finished embedding chunks")
        reconstructed = self.reconstruct(embeddings).tolist()
        logger.info(f"reconstructed")

        return reconstructed, embeddings  # i.e., mean pool

    def count_tokens(self, text):
        text = text.replace("\n", " ")
        chunks = self.chunk_text_to_tokens(text)
        return sum([len(chunk) for chunk in chunks])

    def count_tokens_gemini(self, text):
        for i in range(3):
            try:
                return self.client.models.count_tokens(
                    model='gemini-embedding-exp-03-07',
                    contents=text
                ).total_tokens
            except Exception as e:
                time.sleep(10)
                if i == 2:
                    raise e
        return 0