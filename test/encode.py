import numpy as np

from src import Encoder


def test_chunking():
    long_text = "This is a sample sentence. " * 10000  # Long enough to force multiple chunks
    encoder = Encoder(name="text-embedding-3-small", stride=1000)
    emb = encoder.encode(long_text)
    print("Embedding shape:", np.array(emb).shape)


if __name__ == "__main__":
    test_chunking()