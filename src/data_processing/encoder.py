import logging

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

    def chunk_by_token_counts(self, text, max_tokens=0, stride_percent=0):
        """
        Split text into chunks of max_tokens size, with stride.
        Returns a list of token chunks.
        Developed for models without Tokenizer, but a function to return the total counts of tokens.
        """
        max_tokens = self.max_tokens if max_tokens == 0 else max_tokens
        stride_percent = self.stride if stride_percent == 0 else stride_percent

        def count_tokens(textt):
            return self.client.models.count_tokens(
                model='gemini-embedding-exp-03-07',
                contents=textt
            ).total_tokens

        words = text.split()
        n_words = len(words)
        chunks = []

        i = 0
        stride_size = 0

        while i < n_words:

            print('new chunk!')
            # Start with an overestimate that 1 word = 1 tok. IRL, 1 word = 2 or 3 toks.
            est_words = max_tokens
            high = min(i + est_words, n_words)
            low = i
            current = high

            best = i
            step = max(1, (high - low) // 2)

            while step > 0:
                candidate = ' '.join(words[i:current])
                tokens = count_tokens(candidate)

                if tokens <= max_tokens:
                    # Valid chunk, try to go longer
                    best = current
                    current = min(current + step, n_words)
                else:
                    # Too long, try to shorten
                    current = max(current - step, i + 1)

                step = step // 2  # Reduce step size, like a damped pendulum

            if best == i:
                raise ValueError("A single word or minimal span exceeds max_tokens.")

            chunk = ' '.join(words[i:best])
            chunks.append(chunk)

            if stride_size == 0:
                stride_size = max(1, int((best - i) * stride_percent))

            i = best - stride_size

            candidate_last = ' '.join(words[i:])
            tok_count = count_tokens(candidate_last)
            if tok_count <= max_tokens:
                chunks.append(candidate_last)
                break

        return chunks


    def encode(self, text, verbose=False, disallowed_special=()):
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
            chunks = self.chunk_by_token_counts(text, stride_percent = self.stride) # IN PERCENTAGE, been testing 10% in OpenAI
            for chunk in chunks:
                result = self.client.models.embed_content(
                    model = self.name,
                    contents=chunk
                )
                embeddings.append(result.embeddings[0].values)

        logger.info(f"finished embedding chunks")
        reconstructed = self.reconstruct(embeddings).tolist()
        logger.info(f"reconstructed")

        return reconstructed, embeddings  # i.e., mean pool

    def count_tokens(self, text):
        text = text.replace("\n", " ")
        chunks = self.chunk_text_to_tokens(text)
        return sum([len(chunk) for chunk in chunks])
