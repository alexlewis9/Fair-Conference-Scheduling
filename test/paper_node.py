# Load from CSV
from src import PaperNode
from src import DATA_DIR
import os

papers = PaperNode.load_from_csv(os.path.join(DATA_DIR, "test", "papers.csv"), emb_column="emb_v2")

# Preview
for paper in papers:
    print(paper)
#target: <PaperNode id=gFR4QwK53h title=Gene Regulatory Network Infere... authors=6 year=2024>