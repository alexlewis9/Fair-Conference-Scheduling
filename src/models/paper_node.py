from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import ast
from . import Node

@dataclass
class PaperNode(Node):
    id: str
    emb: np.ndarray

    # Optional metadata
    title: str = ''
    authors: List[str] = None
    year: int = 0
    session: str = ''
    abstract: str = ''
    publisher: str = ''
    forum_content: str = ''
    paper_content: str = ''
    pdf_url: str = ''
    openreview_url: str = ''
    url: str = ''

    def __post_init__(self):
        if not isinstance(self.emb, np.ndarray):
            raise TypeError("`emb` must be a numpy array.")
        if self.authors is None:
            self.authors = []
        super().__init__(self.id, self.emb)

    @classmethod
    def from_row(cls, row: Dict[str, Any], emb_column: str = 'emb') -> "PaperNode":
        """
        Create a PaperNode from a single CSV row.
        Automatically parses string embeddings like '[0.1, 0.2, 0.3]'.
        """
        if emb_column not in row:
            raise KeyError(f"Missing embedding column: '{emb_column}'")

        metadata = dict(row)
        raw_emb = metadata.pop(emb_column)
        metadata['emb'] = cls._parse_embedding(raw_emb)

        # Type coercion: convert 'authors' to a list if stringified
        if isinstance(metadata.get("authors"), str):
            try:
                metadata["authors"] = ast.literal_eval(metadata["authors"])
            except Exception:
                metadata["authors"] = [metadata["authors"]]

        return cls(**metadata)

    @classmethod
    def load_from_csv(cls, path: str, emb_column: str = 'emb') -> List["PaperNode"]:
        """
        Load PaperNode objects from a CSV file.
        """
        df = pd.read_csv(path)
        return [cls.from_row(row, emb_column) for _, row in df.iterrows()]

    @staticmethod
    def _parse_embedding(raw: Any) -> np.ndarray:
        if isinstance(raw, str):
            return np.array(ast.literal_eval(raw))
        elif isinstance(raw, (list, np.ndarray)):
            return np.array(raw)
        else:
            raise ValueError(f"Unsupported embedding format: {type(raw)}")

    def __repr__(self):
        return f"<PaperNode id={self.id} title={self.title[:30]}... authors={len(self.authors)} year={self.year}>"
