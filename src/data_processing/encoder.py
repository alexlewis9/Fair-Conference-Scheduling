
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(self, name):
        self.name = name
        if (name == "text-embedding-3-small" or
         name == "text-embedding-ada-002" or
         name == "text-embedding-3-large"):
            load_dotenv()
            self.client = OpenAI()
        elif name == 'gte-Qwen2-7B-instruct':
            self.client = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct",
                                              trust_remote_code=True)
        else:
            raise ValueError(
                "Invalid model name. Please choose from 'text-embedding-3-small',"
                " 'text-embedding-ada-002', 'text-embedding-3-large', or 'gte-Qwen2-7B-instruct'.")


    def encode(self, text):
        text = text.replace("\n", " ")
        if (self.name == "text-embedding-3-small" or
                self.name == "text-embedding-ada-002" or
                self.name == "text-embedding-3-large"):
            # return self.client.embeddings.create(input=text, model=self.name).data[0].embedding
            # TODO: uncomment this for prod
            return ['test']
        elif self.name == 'gte-Qwen2-7B-instruct':
            return self.client.encode(text)
        return None